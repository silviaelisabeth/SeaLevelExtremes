import logging
import random
import warnings

import numpy as np
import statsmodels.api as sm
from joblib import Parallel, delayed
from pandas import DataFrame
from scipy import linalg, stats
from scipy.optimize import minimize
from scipy.stats import chi2, genextreme, norm
from statsmodels.tools.numdiff import approx_hess

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger("mp_gev_analysis")


def print_and_append_notes(message:str, ls_notes:list, print_msg: bool):
    if print_msg:
        print(message)

    logger.info(message)
    ls_notes.append(message)
    return ls_notes


def print_report(result, loc_ex):
    print(f'GEV Analysis Overview for location ID: {loc_ex}')

    print('\nStationary GEV analysis')
    gev_stat = result['stationary']
    print(f'µ, mm:\t\t{gev_stat["location"]*1000:.2f} ± {gev_stat["location_std"]*1000:.2e}')
    print(f'scale, mm:\t{gev_stat["scale"]*1000:.3f} ± {gev_stat["scale_std"]*1000:.2e}')
    print(f'shape:\t\t{gev_stat["shape"]:.3f} ± {gev_stat["shape_std"]:.3e}')


    print('\nNON-Stationary GEV analysis')
    gev_nonstat = result['nonstationary']
    params_ns = gev_nonstat['params_hat']
    params_ns_std = gev_nonstat['params_std']
    print(f'µ0, mm:\t\t{params_ns[0]*1000:.2f} ± {params_ns_std[1]*1000:.2e}')
    print(f'µ1, mm/yr:\t {params_ns[1]*1000:.3f} ± {params_ns_std[0]*1000:.2e}')
    print(f'scale, mm:\t {params_ns[2]*1000:.3f} ± {params_ns_std[2]*1000:.2e}')
    print(f'shape:\t\t{params_ns[3]:.3f} ± {params_ns_std[3]:.3e}')

    print('\nMODEL COMPARISON')
    print(result['model_comparison']['LRT']['interpretation'])
    print(f'p_value {result["model_comparison"]["LRT"]["p_value"]:.3e}')
    print(f'delta_LL {result["model_comparison"]["LRT"]["delta_LL"]:.3e}')
    print('-------------------------------------------------------------')


def extract_annual_maxima_at_location(
    data_hindcast: DataFrame, lon: float, lat: float, model: str | None = None
    ) -> DataFrame:
    """
    Extract annual maxima for a specific model-location combination.
    For each sim_year (actual calendar year), take the maximum across all leads and ensemble members.
    """
    if model:
        subset = data_hindcast[
            (data_hindcast['model'] == model) &
            (data_hindcast['lon'] == lon) &
            (data_hindcast['lat'] == lat)
        ].copy()
    else:
        subset = data_hindcast[
            (data_hindcast['lon'] == lon) &
            (data_hindcast['lat'] == lat)
        ].copy()

    if subset.empty:
        return DataFrame(columns=['year', 'annual_max'])

    return (subset
            .sort_values(['sim_year', 'model'])
            .reset_index(drop=True)
            .rename(columns={'sim_year': 'year', 'storm_surge': 'annual_max'})
            ).dropna()


def gev_neg_loglik(params, data):
    """
    Negative log-likelihood for a GEV (SciPy genextreme) with correct support.
    `params` = (c, loc, scale) where c = -xi (SciPy's shape convention).
    """
    c, loc, scale = params

    if scale <= 0:
        return np.inf
    # Support check:
    # If c > 0 (Weibull), support x <= loc + scale/c.
    # If c < 0 (Fréchet), support x >= loc + scale/c.
    if c > 0:
        if np.any(data > loc + scale/c):
            return np.inf
    elif c < 0:
        if np.any(data < loc + scale/c):
            return np.inf
    
    logpdf = stats.genextreme.logpdf(data, c, loc, scale)
    if np.any(~np.isfinite(logpdf)):
        return np.inf
    return -np.sum(logpdf)


def fit_gev_mle(data, init_shape=0.0):
    """
    Fit GEV by maximum likelihood. Returns fitted parameters and log-lik.
    Uses scipy.optimize.minimize starting from (init_shape, mean, std).
    Setting init_shape=0 often stabilizes the fit【43†L540-L548】.
    """
    init = np.array([ -init_shape, np.mean(data), np.std(data, ddof=1) ])
    bounds = [(-10, 10), (None, None), (1e-6, None)]
    
    res = minimize(lambda p: gev_neg_loglik(p, data), init, bounds=bounds, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError("GEV MLE did not converge: " + res.message)
    
    c_ml, loc_ml, scale_ml = res.x
    shape_ml = -c_ml  # convert to xi
    loglik = -res.fun

    n = len(data)
    aic = 2*3 - 2*loglik
    bic = np.log(n)*3 - 2*loglik
    dist_type = "Weibull" if c_ml > 0 else "Fréchet" if c_ml < 0 else "Gumbel"
    
    return {
        "shape": shape_ml, "location": loc_ml, "scale": scale_ml,
        "log_likelihood": loglik, "aic": aic, "bic": bic,
        "dist_type": dist_type, "tail": ("bounded" if c_ml>0 else "heavy"),
        "n_obs": n
    }


def gev_hessian(params, data, eps=1e-5):
    """
    Compute the Hessian of the negative log-likelihood via central differences.
    Returns the observed information matrix. Parameter `eps` is a relative step.
    """
    n = len(params)
    f0 = gev_neg_loglik(params, data)
    if not np.isfinite(f0):
        raise ValueError("Initial log-likelihood is not finite.")
    
    H = np.zeros((n, n))
    
    for i in range(n):
        pi = params[i]
        hi = eps * max(1.0, abs(pi))
        ei = np.zeros(n); ei[i] = hi
        f_plus  = gev_neg_loglik(params + ei, data)
        f_minus = gev_neg_loglik(params - ei, data)
        if np.isfinite(f_plus) and np.isfinite(f_minus):
            H[i,i] = (f_plus - 2*f0 + f_minus) / (hi**2)
        else:
            H[i,i] = np.nan
        for j in range(i+1, n):
            pj = params[j]
            hj = eps * max(1.0, abs(pj))
            ej = np.zeros(n); ej[j] = hj
            f_pp = gev_neg_loglik(params + ei + ej, data)
            f_pm = gev_neg_loglik(params + ei - ej, data)
            f_mp = gev_neg_loglik(params - ei + ej, data)
            f_mm = gev_neg_loglik(params - ei - ej, data)
            if all(np.isfinite([f_pp, f_pm, f_mp, f_mm])):
                H[i,j] = (f_pp - f_pm - f_mp + f_mm) / (4*hi*hj)
                H[j,i] = H[i,j]
            else:
                H[i,j] = H[j,i] = np.nan
    return 0.5*(H + H.T)


def gev_bootstrap(data, B=500, seed=None):
    """
    Parametric bootstrap: resample with replacement and refit GEV.
    Returns stddev of shape, loc, scale (bootstrap estimates)【40†L19-L24】.
    """
    rng = np.random.default_rng(seed)

    boots = []
    for _ in range(B):
        sample = rng.choice(data, size=len(data), replace=True)
        try:
            result = fit_gev_mle(sample, init_shape=0.0)
            boots.append((result["shape"], result["location"], result["scale"], result['n_obs']))
        except Exception:
            continue
    boots = np.array(boots)
    
    if boots.size == 0:
        raise RuntimeError("Bootstrap failed: no successful fits.")

    shape_std = np.std(boots[:,0], ddof=1)
    loc_std   = np.std(boots[:,1], ddof=1)
    scale_std = np.std(boots[:,2], ddof=1)
    n_obs = np.std(boots[:,3], ddof=1)

    return {"shape_std": shape_std, "location_std": loc_std, "scale_std": scale_std, "n_obs": n_obs}


def compute_cov_matrix_v1(
    gev_params: dict, data: np.ndarray, is_nonstationary:bool, years: np.ndarray = None
    ) -> np.ndarray:
    """
    Compute the covariance matrix of fitted GEV parameters using a numerically stable Hessian.
    
    Parameters
    ----------
    gev_params : dict
        Fitted GEV parameters. Should contain:
        - 'shape', 'location', 'scale' for stationary
        - or 'mu0', 'mu1', 'sigma', 'xi' for non-stationary location trend
    data : np.ndarray
        Observed annual maxima used for fitting
    years : np.ndarray, optional
        Years array (required if non-stationary model with trend)
    
    Returns
    -------
    cov_matrix : np.ndarray
        Covariance matrix of parameters, or None if Hessian is singular.
    """
    messages = ''
    if is_nonstationary:
        # Non-stationary location μ(t) = μ0 + μ1 * t
        t = (years - gev_params['years_mean']) / gev_params['years_std']

        def neg_loglik(theta):
            mu0, mu1, log_sigma, xi = theta
            sigma = np.exp(log_sigma)
            mu_t = mu0 + mu1 * t
            c = -xi  # SciPy convention
            z = 1 + xi * (data - mu_t)/sigma

            # Penalize invalid parameters (support violation or sigma <= 0)
            if np.any(z <= 1e-10) or sigma <= 0:
                return 1e10

            ll = stats.genextreme.logpdf(data, c, loc=mu_t, scale=sigma)
            if not np.all(np.isfinite(ll)):
                return 1e10

            return -np.sum(ll)

        theta_hat = np.array([
            gev_params.get('mu0', gev_params['params_hat'][0]),
            gev_params.get('mu1', gev_params['params_hat'][1]),
            np.log(gev_params.get('sigma', gev_params['params_hat'][2])),
            gev_params.get('xi', gev_params['params_hat'][3])
        ])

    else:
        # Stationary GEV: μ, σ, ξ
        def neg_loglik(theta):
            xi, mu, log_sigma = theta
            sigma = np.exp(log_sigma)
            z = 1 + xi * (data - mu)/sigma
            if np.any(z <= 1e-10) or sigma <= 0:
                return 1e10
            ll = stats.genextreme.logpdf(data, -xi, loc=mu, scale=sigma)
            if not np.all(np.isfinite(ll)):
                return 1e10
            return -np.sum(ll)

        theta_hat = np.array([
            gev_params['shape'],
            gev_params['location'],
            np.log(gev_params['scale'])
        ])

    try:
        if 'trend_in' in gev_params and gev_params['trend_in'] == 'location':
            mu0, mu1, log_sigma, xi = theta_hat
            sigma = np.exp(log_sigma)
            mu_t = mu0 + mu1 * t
            z = 1 + xi * (data - mu_t)/sigma
        else:
            xi, mu, log_sigma = theta_hat
            sigma = np.exp(log_sigma)
            z = 1 + xi * (data - mu)/sigma
        messages += f'\t\t\tMLE min(z):{np.min(z)}'
    except Exception:
        pass


    hessian_ = approx_hess(theta_hat, neg_loglik)
    eigvals = linalg.eigvals(hessian_)
    messages += f'\n\t\t\tHessian eigenvalues: {eigvals}'

    try:
        cov_matrix = linalg.inv(hessian_)
    except linalg.LinAlgError:
        print("Warning: Hessian not invertible; returning None")
        messages += "\n\t\t\tWarning: Hessian not invertible; returning None"
        return None, list, messages

    return cov_matrix, eigvals, messages


def fit_stationary_gev_incl_uncertainty(loc_id, data, B, seed, ls_notes, print_msg):
    cov = None
    std_errors = None

    stationary = fit_gev_mle(data, init_shape=0.0)
    ls_notes = print_and_append_notes(
        message=f'\t\tLoc {loc_id} | MLE results: {stationary}', ls_notes=ls_notes, print_msg=print_msg
        )

    c_ml = -stationary["shape"]
    params_ml = np.array([c_ml, stationary["location"], stationary["scale"]])
    H = gev_hessian(params_ml, data, eps=1e-5)

    if np.any(np.isnan(H)) or np.linalg.det(H) <= 0:
        ls_notes = print_and_append_notes(
            message=f'\nLoc {loc_id} | Hessian invalid or singular; switching to bootstrap for CIs.',
            ls_notes=ls_notes, print_msg=print_msg
        )

        bs = gev_bootstrap(data, B=B, seed=seed)

        stationary["shape_std"] = bs["shape_std"]
        stationary["location_std"] = bs["location_std"]
        stationary["scale_std"] = bs["scale_std"]

    else:
        cov = np.linalg.inv(H)
        std_errors = np.sqrt(np.diag(cov))
        stationary["shape_std"] = std_errors[0]
        stationary["location_std"] = std_errors[1]
        stationary["scale_std"] = std_errors[2]
        stationary['cov'] = cov
        stationary['std_errors'] = std_errors

    message = (
        f'\t\tLoc {loc_id} | Stationary GEV fit (n={stationary["n_obs"]}):'
        + f'\n\t\t\t  shape (ξ) = {stationary["shape"]:.4f} ± {stationary["shape_std"]:.4f}'
        + f'\n\t\t\t  location = {stationary["location"]:.4f} ± {stationary["location_std"]:.4f}'
        + f'\n\t\t\t  scale    = {stationary["scale"]:.4f} ± {stationary["scale_std"]:.4f}'
    )
    if cov is not None:
        message += f"\n\t\t\t  covariance = {cov}"
        message += f"\n\t\t\t  standard errors = {std_errors}"

    ls_notes = print_and_append_notes(message=message, ls_notes=ls_notes, print_msg=print_msg)
    return stationary, ls_notes
    

def fit_non_stationary_gev_incl_uncertainty(
    loc_id, data, stationary, uncertainty, years, ls_notes, print_msg, seed=None
    ):
    if uncertainty is None:
        uncertainty = 'delta'

    t = (years - years.mean()) / years.std()   
    x0 = [stationary['location'], 0.0, stationary['scale'], stationary['shape']]
    
    def neg_loglik(params):
        mu0, mu1, sigma, xi = params
        mu_t = mu0 + mu1 * t
        sigma_t = np.full_like(t, sigma)
        
        if np.any(sigma_t <= 0):
            return np.inf
        
        z = (data - mu_t) / sigma_t
        if abs(xi) < 1e-10:  # Gumbel case
            ll = -np.sum(np.log(sigma_t)) - np.sum(z) - np.sum(np.exp(-z))
        else:
            term = 1 + xi * z
            if np.any(term <= 0):
                return np.inf
            ll = -np.sum(np.log(sigma_t)) - np.sum((1 + 1/xi) * np.log(term)) - np.sum(term**(-1/xi))
        
        return -ll
        
    try:
        result = minimize(neg_loglik, x0, method='Nelder-Mead')
        params_hat = result.x

    except Exception as e:
        message = f"Loc {loc_id} | Non-stationary MLE fitting failed: {e}"
        ls_notes = print_and_append_notes(message=message, ls_notes=ls_notes, print_msg=print_msg)

        return None
    
    message = f'\t\t\tLoc {loc_id} | Nonstationary GEV - MLE {params_hat}'
    ls_notes = print_and_append_notes(message=message, ls_notes=ls_notes, print_msg=print_msg)


    nonstationary = {
        'params_hat': params_hat, 'n_obs': len(data), 'years_mean': years.mean(), 'years_std': years.std()
        }
    
    if uncertainty == "delta":
        message = f"\t\t\tLoc {loc_id} | Computing delta method for non-stationary GEV..."
        ls_notes = print_and_append_notes(message=message, ls_notes=ls_notes, print_msg=print_msg)

        try:
            cov_ns, hessian_eigenvalues, messages = compute_cov_matrix_v1(
                gev_params=nonstationary, data=data, years=years, is_nonstationary=True
                )
            nonstationary['Hessian eigenvalues'] = hessian_eigenvalues
            ls_notes = print_and_append_notes(message=messages, ls_notes=ls_notes, print_msg=print_msg)

        except:
            cov_ns = None
            message = f'Loc {loc_id} | Hessian inversion failed; Delta-method SD not computed.'
            ls_notes = print_and_append_notes(message=messages, ls_notes=ls_notes, print_msg=print_msg)

        if cov_ns is not None:

            nonstationary['cov'] = cov_ns
            try:
                diag_vals = np.diag(cov_ns)

                if np.any(diag_vals < 0):
                    ls_notes = print_and_append_notes(
                        message=f"Loc {loc_id} | Negative variance in cov diagonal: {diag_vals}",
                        ls_notes=ls_notes, print_msg=print_msg
                    )
                    raise ValueError(f"Loc {loc_id} | Negative variance in cov diagonal: {diag_vals}")

                params_std = np.sqrt(diag_vals)

                if np.any(np.isnan(params_std)):
                    ls_notes = print_and_append_notes(
                        message=f"Loc {loc_id} | NaN in params_std after sqrt", ls_notes=ls_notes, print_msg=print_msg
                    )
                    raise ValueError("Loc {loc_id} | NaN in params_std after sqrt")
            except Exception as e:
                params_std = None
                ls_notes = print_and_append_notes(
                        message=f"Loc {loc_id} | params_std failed: {e}", ls_notes=ls_notes, print_msg=print_msg
                    )

            nonstationary['params_std'] = params_std
        else:
            ls_notes = print_and_append_notes(
                message=f"Loc {loc_id} | Delta Method failed, falling back to Fisher Information",
                ls_notes=ls_notes, print_msg=print_msg
                )
            uncertainty = "fisher"

    if uncertainty == "fisher":
        ls_notes = print_and_append_notes(
            message=f"Loc {loc_id} | Computing Fisher Information for non-stationary GEV...",
            ls_notes=ls_notes, print_msg=print_msg
            )
                    
        try:
            cov = _compute_fisher_info_generic(neg_loglik, params_hat)
            if cov is not None and np.all(np.isfinite(cov)):
                std_errors = np.sqrt(np.diag(cov))
                if np.all(np.isfinite(std_errors)):
                    nonstationary['params_std'] = std_errors
                else:
                    ls_notes = print_and_append_notes(
                        message=f"Loc {loc_id} | Fisher Information failed (not all values computed), falling back to bootstrap",
                        ls_notes=ls_notes, print_msg=print_msg
                    )
                    uncertainty = "bootstrap"
                    B = 300
            else:
                ls_notes = print_and_append_notes(
                    message=f"Loc {loc_id} | Fisher Information failed, falling back to bootstrap",
                    ls_notes=ls_notes, print_msg=print_msg
                    )
                uncertainty = "bootstrap"
                B = 150
        except Exception as e:
            ls_notes = print_and_append_notes(
                message=f"Loc {loc_id} | Fisher Information exception: {e}, falling back to bootstrap",
                ls_notes=ls_notes, print_msg=print_msg
                )
            uncertainty = "bootstrap"
            B = 300
    
    if uncertainty == "bootstrap":
        ls_notes = print_and_append_notes(
            message=f"Loc {loc_id} | Computing bootstrap uncertainty for non-stationary GEV...",
            ls_notes=ls_notes, print_msg=print_msg
            )
        
        if seed is not None:
            random.seed(seed)
        
        #mu0, mu1, _, _ = params_hat
        #t_std = (years - years.mean()) / years.std()
        #mu_t = mu0 + mu1 * t_std

        mu0_samples, mu1_samples, sigma_samples, xi_samples = [], [], [], []

        for _ in range(B):
            res_b = minimize(neg_loglik, params_hat, method='Nelder-Mead')
            p_b = res_b.x
            mu0_samples.append(p_b[0])
            mu1_samples.append(p_b[1])
            sigma_samples.append(p_b[2])
            xi_samples.append(p_b[3])

        nonstationary.update({
            'mu0_samples': np.array(mu0_samples),
            'mu1_samples': np.array(mu1_samples),
            'sigma_samples': np.array(sigma_samples),
            'xi_samples': np.array(xi_samples),
            'mu0_std': np.std(mu0_samples, ddof=1),
            'mu1_std': np.std(mu1_samples, ddof=1),
            'sigma_std': np.std(sigma_samples, ddof=1),
            'xi_std': np.std(xi_samples, ddof=1),
        })
    
    return nonstationary, ls_notes


def fit_pooled_gev_with_uncertainty(
    loc_id, data, year: int|None= None, years: list | None = None, uncertainty_ns:str|None ='delta', 
    B:int=300, seed:int=42, print_msg=False
    ):
    ls_notes = []
    n = len(data)

    # ---- STEP 1: Get stationary ----
    ls_notes = print_and_append_notes(
        message=f'\t\tLoc {loc_id} | Compute stationary GEV incl uncertainty', ls_notes=ls_notes, print_msg=print_msg
        )   

    if n < 10:
        message = (
            f'\n\t Loc {loc_id} | Warning: Only {n} observations'
            + f' in {year} if year else '
            + '. Need at least 10 for reliable GEV fit.'
        )
        ls_notes = print_and_append_notes(message=message, ls_notes=ls_notes, print_msg=print_msg)

    try:
        stationary, ls_notes = fit_stationary_gev_incl_uncertainty(
            loc_id=loc_id, data=data, B=B, seed=seed, ls_notes=ls_notes, print_msg=print_msg, 
            )
        pooled_gev = dict({'stationary': stationary})

        # ---- STEP 2: Get NON-stationary ----
        message = f'\t\tLoc {loc_id} | Compute non-stationary GEV incl uncertainty'
        ls_notes = print_and_append_notes(message=message, ls_notes=ls_notes, print_msg=print_msg)

        if n < 20:
            message = f'Loc {loc_id} | Warning: Non-stationary GEV needs ≥20 observations but has {n}, '
            message += 'skipping non-stationary GEV...'
            ls_notes = print_and_append_notes(message=message, ls_notes=ls_notes, print_msg=print_msg)
            return {'stationary': stationary, 'nonstationary': None}, ls_notes
    
        nonstationary, ls_notes = fit_non_stationary_gev_incl_uncertainty(
            loc_id=loc_id, data=data, stationary=stationary, years=years, uncertainty=uncertainty_ns, 
            ls_notes=ls_notes, print_msg=print_msg
            )
        pooled_gev.update({'nonstationary': nonstationary})

    except Exception as e:
        message = f"Failed to conduct GEV fitting due to error: {e}"
        ls_notes.append(message)
        return {'stationary': None, 'nonstationary': None}, ls_notes
    
    return pooled_gev, ls_notes


def likelihood_ratio_test(LL_s, LL_ns, df):
    delta_LL = 2 * (LL_ns - LL_s)
    if delta_LL < 0:
        return delta_LL, 1.0, '→ stationary model is sufficient, trend in μ is not significant'
    else:
        p = 1 - chi2.cdf(delta_LL, df)
        if p< 0.05:
            return delta_LL, p, '→ non-stationary model is significantly better \n→ μ(t) trend matters'
        else:
            return delta_LL, p, '→ adding a trend doesn’t improve the fit'


def compare_stationary_nonstationary(stationary, nonstationary, data_loc):
    """_summary_
        μ (mu) → location parameter
        σ (sigma) → scale parameter
        ξ (xi) → shape parameter
    Args:
        stationary (_type_): _description_
        nonstationary (_type_): _description_
        data_loc (_type_): _description_

    Returns:
        _type_: _description_
    """
    n = stationary['n_obs']

    # Stationary
    mu_s = stationary['location']
    sigma_s = stationary['scale']
    xi_s = stationary['shape']

    z_s = (data_loc.annual_max - mu_s)/sigma_s
    if abs(xi_s) < 1e-10:
        LL_s = -np.sum(np.log(sigma_s)) - np.sum(z_s) - np.sum(np.exp(-z_s))
    else:
        term = 1 + xi_s*z_s
        LL_s = -n * np.log(sigma_s) - np.sum((1 + 1/xi_s) * np.log(term)) - np.sum(term**(-1/xi_s))
    k_s = 3  # stationary: μ, σ, ξ
    AIC_s = 2*k_s - 2*LL_s
    BIC_s = k_s*np.log(n) - 2*LL_s

    # Non-stationary (location trend)
    mu0 = nonstationary['params_hat'][0]
    mu1 = nonstationary['params_hat'][1]
    sigma_ns = nonstationary['params_hat'][2] 
    xi_ns = nonstationary['params_hat'][3] 

    t_scaled = (data_loc.year - nonstationary['years_mean']) / nonstationary['years_std']  # scaled years

    mu_t = mu0 + mu1*t_scaled
    z_ns = (data_loc.annual_max - mu_t)/sigma_ns
    if abs(xi_ns) < 1e-10:
        LL_ns = -np.sum(np.log(sigma_ns)) - np.sum(z_ns) - np.sum(np.exp(-z_ns))
    else:
        term = 1 + xi_ns*z_ns
        LL_ns = -n * np.log(sigma_ns) - np.sum((1 + 1/xi_ns) * np.log(term)) - np.sum(term**(-1/xi_ns))

    k_ns = 4  # μ0, μ1, σ, ξ
    AIC_ns = 2*k_ns - 2*LL_ns
    BIC_ns = k_ns*np.log(n) - 2*LL_ns

    # Likelihood ratio test
    df = k_ns - k_s
    delta_LL, p_value, interpretation = likelihood_ratio_test(LL_s=LL_s, LL_ns=LL_ns, df=df)

    return dict({
            'stationary': {'LL': LL_s, 'k': k_s, 'AIC': AIC_s, 'BIC': BIC_s},
            'nonstationary': {'LL': LL_ns, 'k': k_ns, 'AIC': AIC_ns, 'BIC': BIC_ns},
            'LRT': {'delta_LL': delta_LL, 'df': df, 'p_value': p_value, 'interpretation': interpretation}
        })


def compute_return_levels_delta(stationary, nonstationary, T, t_eval, confidence_level_pc:float = 0.9):
    t_eval = np.asarray(t_eval)
    
    # ---- Stationary ----
    mu = stationary['location']
    sigma = stationary['scale']
    xi = stationary['shape']
    cov_theta = stationary['cov']  # 3x3 covariance matrix for (mu, sigma, xi)
    
    if abs(xi) < 1e-10:
        z_T = mu + sigma * -np.log(-np.log(1 - 1/T))
        grad = np.array([1, -np.log(-np.log(1 - 1/T)), 0])
    else:
        f = (-np.log(1 - 1/T))**(-xi) - 1
        z_T = mu + sigma/xi * f
        df_dxi = -(-np.log(1 - 1/T))**(-xi) * np.log(-np.log(1 - 1/T))
        grad = np.array([1, f/xi, -sigma/xi**2 * f + sigma/xi * df_dxi])
    
    var_zT = grad @ cov_theta @ grad
    z = norm.ppf(1 - (1-confidence_level_pc)/2) 
    ci_lower = z_T - z * np.sqrt(var_zT)
    ci_upper = z_T + z * np.sqrt(var_zT)
    
    # ---- Non-stationary ----
    mu0, mu1, sigma, xi = nonstationary['params_hat']
    cov_theta_ns = nonstationary['cov']  # 4x4 covariance matrix for (mu0, mu1, sigma, xi)
    years_mean = nonstationary['years_mean']
    years_std = nonstationary['years_std']
    
    t_scaled = (t_eval - years_mean) / years_std
    mu_t = mu0 + mu1 * t_scaled
    
    if abs(xi) < 1e-10:
        z_T_ns = mu_t + sigma * -np.log(-np.log(1 - 1/T))
        grad_ns = np.array([1, t_scaled, -np.log(-np.log(1 - 1/T)), 0])
    else:
        f = (-np.log(1 - 1/T))**(-xi) - 1
        z_T_ns = mu_t + sigma/xi * f
        df_dxi = -(-np.log(1 - 1/T))**(-xi) * np.log(-np.log(1 - 1/T))
        grad_ns = np.array([1, t_scaled, f/xi, -sigma/xi**2*f + sigma/xi*df_dxi])
    
    var_zT_ns = grad_ns @ cov_theta_ns @ grad_ns
    ci_lower_ns = z_T_ns - 1.96 * np.sqrt(var_zT_ns)
    ci_upper_ns = z_T_ns + 1.96 * np.sqrt(var_zT_ns)
    
    return {
        'stationary': {'z_T': z_T, 'lower': ci_lower, 'upper': ci_upper},
        'nonstationary': {'z_T': z_T_ns, 'lower': ci_lower_ns, 'upper': ci_upper_ns}
    }


def compute_return_levels_delta_stat(stationary, T, confidence_level_pc:float = 0.9):
    mu = stationary['location']
    sigma = stationary['scale']
    xi = stationary['shape']
    cov_theta = stationary['cov']  # 3x3 covariance matrix for (mu, sigma, xi)
    
    if abs(xi) < 1e-10:
        z_T = mu + sigma * -np.log(-np.log(1 - 1/T))
        grad = np.array([1, -np.log(-np.log(1 - 1/T)), 0])
    else:
        f = (-np.log(1 - 1/T))**(-xi) - 1
        z_T = mu + sigma/xi * f
        df_dxi = -(-np.log(1 - 1/T))**(-xi) * np.log(-np.log(1 - 1/T))
        grad = np.array([1, f/xi, -sigma/xi**2 * f + sigma/xi * df_dxi])
    
    var_zT = grad @ cov_theta @ grad
    z = norm.ppf(1 - (1-confidence_level_pc)/2) 
    ci_lower = z_T - z * np.sqrt(var_zT)
    ci_upper = z_T + z * np.sqrt(var_zT)
    
    return {'z_T': z_T, 'lower': ci_lower, 'upper': ci_upper}


def compute_return_levels_delta_nonstat(nonstationary, T, t_eval, confidence_level_pc:float = 0.9):
    t_eval = np.asarray(t_eval)
    
    mu0, mu1, sigma, xi = nonstationary['params_hat']
    cov_theta_ns = nonstationary['cov']  # 4x4 covariance matrix for (mu0, mu1, sigma, xi)
    years_mean = nonstationary['years_mean']
    years_std = nonstationary['years_std']
    
    t_scaled = (t_eval - years_mean) / years_std
    mu_t = mu0 + mu1 * t_scaled
    
    if abs(xi) < 1e-10:
        z_T_ns = mu_t + sigma * -np.log(-np.log(1 - 1/T))
        grad_ns = np.array([1, t_scaled, -np.log(-np.log(1 - 1/T)), 0])
    else:
        f = (-np.log(1 - 1/T))**(-xi) - 1
        z_T_ns = mu_t + sigma/xi * f
        df_dxi = -(-np.log(1 - 1/T))**(-xi) * np.log(-np.log(1 - 1/T))
        grad_ns = np.array([1, t_scaled, f/xi, -sigma/xi**2*f + sigma/xi*df_dxi])
    
    var_zT_ns = grad_ns @ cov_theta_ns @ grad_ns
    z = norm.ppf(1 - (1-confidence_level_pc)/2) 
    ci_lower_ns = z_T_ns - z * np.sqrt(var_zT_ns)
    ci_upper_ns = z_T_ns + z * np.sqrt(var_zT_ns)
    
    return {'z_T': z_T_ns, 'lower': ci_lower_ns, 'upper': ci_upper_ns}


def return_levels_bootstrap_stationary(stationary, T, confidence_level=0.9):
    mu = stationary['location']
    sigma = stationary['scale']
    xi = stationary['shape']

    y = -np.log(1 - 1/T)

    zT = np.where(
        np.abs(xi) < 1e-10,
        mu + sigma * -np.log(y),
        mu + sigma/xi * (y**(-xi) - 1)
    )

    alpha = 1 - confidence_level

    return {
        "z_T": np.mean(zT),
        "lower": np.quantile(zT, alpha/2),
        "upper": np.quantile(zT, 1-alpha/2)
    }


def return_levels_bootstrap_nonstationary(nonstationary, T, t_eval, confidence_level=0.9):
    years_mean = nonstationary['years_mean']
    years_std = nonstationary['years_std']
    
    mu0, mu1, sigma, xi = nonstationary['params_hat']

    t_eval = np.asarray(t_eval)
    t_scaled = (t_eval - years_mean) / years_std

    y = -np.log(1 - 1/T)
    
    B = len(mu0)
    nt = len(t_eval)
    zT = np.zeros((B, nt))

    for i in range(B):
        mu_t = mu0[i] + mu1[i] * t_scaled
        if abs(xi[i]) < 1e-10:
            zT[i,:] = mu_t + sigma[i] * -np.log(y)
        else:
            zT[i,:] = mu_t + sigma[i]/xi[i] * (y**(-xi[i]) - 1)

    alpha = 1 - confidence_level

    return {
        "z_T": np.mean(zT, axis=0),
        "lower": np.quantile(zT, alpha/2, axis=0),
        "upper": np.quantile(zT, 1-alpha/2, axis=0)
    }


def compute_all_return_levels(stationary, nonstationary, return_periods, ls_t_eval, confidence_level_pc:float=0.9):
    dic_rl = {}
    for T in return_periods:
        dic_return_levels_t_eval = dict()
        for t_eval in ls_t_eval:
            if 'cov' in stationary.keys():
                rl_t_stat = compute_return_levels_delta_stat(
                    stationary=stationary, T=T, confidence_level_pc=confidence_level_pc
                )
            else:
                rl_t_stat = return_levels_bootstrap_stationary(
                    stationary=stationary, T=T, confidence_level=confidence_level_pc
                    )

            if 'cov' in nonstationary.keys():
                rl_t_ns = compute_return_levels_delta_nonstat(
                    nonstationary=nonstationary, T=T, t_eval=t_eval, confidence_level_pc=confidence_level_pc
                )
            else:
                rl_t_ns = return_levels_bootstrap_nonstationary(
                    nonstationary=nonstationary, T=T, t_eval=t_eval, confidence_level=confidence_level_pc
                    )

            dic_return_levels_t_eval[t_eval] = {'stationary': rl_t_stat, 'nonstationary': rl_t_ns}
        dic_rl[T] = dic_return_levels_t_eval
        
    df_rl = DataFrame([
            {**dic_rl[T][t_eval][key], 't_eval': t_eval, 'return_period': T, 'model': key}
            for T in return_periods
            for t_eval in dic_rl[T].keys()
            for key in ['stationary', 'nonstationary']
        ])
    return df_rl


def safe_return_level_backup(T, mu, sigma, xi):
    """
    Compute GEV return level for return period T (years).
    Handles xi ≈ 0 and prevents log(0) errors.
    """
    T = np.array(T, dtype=float)
    T = np.maximum(T, 1 + 1e-10)  # avoid T <= 1

    if abs(xi) < 1e-6:  # Gumbel limit
        return mu - sigma * np.log(-np.log(1 - 1/T))
    else:
        u = -np.log(1 - 1/T)
        u = np.maximum(u, 1e-10)
        return mu + sigma/xi * (u**(-xi) - 1)


def safe_return_level(T, mu, sigma, xi):
    """
    Compute GEV return level for return period T (years).
    Vectorized, numerically stable.
    """
    T = np.atleast_1d(T).astype(float)
    T = np.maximum(T, 1 + 1e-10)  # avoid T <= 1

    c = -xi
    
    z = genextreme.ppf(1 - 1/T, c=c, loc=mu, scale=sigma)
    z = np.clip(z, -1e5, 1e5)
    return z


def safe_return_period_backup(z, mu, sigma, xi, max_T=1e4):
    """
    Compute return period corresponding to level z.
    Return period >= 1 year (annual maxima assumption).
    """
    term = 1 + xi*(z - mu)/sigma
    term = np.maximum(term, 1e-10)

    F = np.exp(-term**(-1/xi))
    F = np.clip(F, 0, 1 - 1e-10)  
    
    T = 1/(1 - F)
    T = np.maximum(T, 1.0)  
    
    return np.minimum(T, max_T)


def safe_return_period(z, mu, sigma, xi, max_T=1e4):
    """
    Compute return period corresponding to level z.
    Returns vectorized T, clipped to [1, max_T].
    """
    z = np.atleast_1d(z).astype(float)
    mu = np.atleast_1d(mu).astype(float)

    c = -xi
    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        F = genextreme.cdf(z, c=c, loc=mu, scale=sigma)

    F = np.clip(F, 0, 1 - 1e-10)
    T = 1 / (1 - F)
    T = np.clip(T, 1.0, max_T)
    return T



def filter_unstable_samples(samples):
    filtered = []
    for p in samples:
        _, _, sigma, xi = p

        if sigma <= 0:
            continue
        if abs(xi) > 0.5:      # avoid unrealistic heavy tails
            continue

        filtered.append(p)

    return np.array(filtered)

    
def mu_year(year, mu0, mu1, mean_year, std_year):
    """
    Compute location parameter μ(t) for standardized time.
    params = [mu0, mu1, sigma, xi]
    """
    t_std = (year - mean_year) / std_year
    return mu0 + mu1 * t_std


def rp_evolution(params, years, mean_year, std_year, ref_year, T_ref, max_T=1e4):
    """
    Compute evolution of return period of the reference T_ref surge.
    Returns RP array and 1961 return level.
    """
    mu0, mu1, sigma, xi = params

    mu_ref = mu_year(year=ref_year, mu0=mu0, mu1=mu1, mean_year=mean_year, std_year=std_year)

    # 50-year return level in reference year
    z_ref = safe_return_level(T_ref, mu_ref, sigma, xi)

    # vectorized computation for all years
    mu_all = mu_year(year=years, mu0=mu0, mu1=mu1, mean_year=mean_year, std_year=std_year)
    rp_all = safe_return_period(z_ref, mu_all, sigma, xi, max_T=max_T)

    return rp_all, z_ref


def rp_uncertainty_monte_carlo_backup(params, cov, years, mean_year, std_year, ref_year, T_ref, nsim=5000, cap_rp=1e4):
    """
    Monte Carlo sampling from parameter covariance to get RP uncertainty.
    Returns mean, 5th percentile, 95th percentile of RP evolution.
    """
    samples_ = np.random.multivariate_normal(params, cov, nsim)
    samples = filter_unstable_samples(samples_)
    
    sims = []
    for p in samples:
        _, _, sigma, xi = p
        if sigma <= 0 or abs(xi) > 0.5:
            continue
        
        rp, _ = rp_evolution(p, years, mean_year, std_year, ref_year, T_ref, cap_rp)
        rp = np.minimum(rp, cap_rp)
        
        sims.append(rp)

    sims = np.array(sims)
    
    # remove extreme simulations
    threshold = np.percentile(sims, 99.5)
    sims[sims > threshold] = np.nan

    mean = sims.mean(axis=0)
    low = np.percentile(sims, 5, axis=0)
    high = np.percentile(sims, 95, axis=0)

    return DataFrame(
        [mean, low, high], 
        columns=years, index=['return_period_mean', 'return_period_lower', 'return_period_upper']
        ).T


def rp_uncertainty_monte_carlo(params, cov, years, mean_year, std_year, ref_year, T_ref, nsim=5000, max_T=1e4):
    """
    Monte Carlo sampling to estimate RP uncertainty.
    Returns DataFrame with mean, 5th, 95th percentile of RP evolution.
    """
    # Sample parameter space
    samples = np.random.multivariate_normal(params, cov, nsim)

    # Filter unstable samples
    mask_valid = (samples[:, 2] > 0) & (np.abs(samples[:, 3]) <= 0.5)
    samples = samples[mask_valid]

    # Preallocate
    sims = np.empty((len(samples), len(years)))
    sims[:] = np.nan

    # Vectorized computation
    for i, p in enumerate(samples):
        rp, _ = rp_evolution(p, years, mean_year, std_year, ref_year, T_ref, max_T=max_T)
        sims[i] = rp

    # Compute statistics ignoring NaNs
    mean_rp = np.nanmean(sims, axis=0)
    low_rp = np.nanpercentile(sims, 5, axis=0)
    high_rp = np.nanpercentile(sims, 95, axis=0)

    return DataFrame({
        'year': years,
        'return_period_mean': mean_rp,
        'return_period_lower': low_rp,
        'return_period_upper': high_rp
    }).set_index('year')
    

def rp_uncertainty_monte_carlo_fast(params, cov, years, mean_year, std_year, ref_year, T_ref, nsim=5000, max_T=1e4):
    samples = np.random.multivariate_normal(params, cov, nsim)

    mu0_s = samples[:,0]
    mu1_s = samples[:,1]
    sigma_s = samples[:,2]
    xi_s = samples[:,3]

    # ---- filter unrealistic samples ----
    mask = (sigma_s > 0) & (np.abs(xi_s) < 0.5)
    mu0_s, mu1_s, sigma_s, xi_s = mu0_s[mask], mu1_s[mask], sigma_s[mask], xi_s[mask]
    ns = len(mu0_s)

    # ---- reference location parameter ----
    mu_ref = mu_year(year=ref_year, mu0=mu0_s, mu1=mu1_s, mean_year=mean_year, std_year=std_year)

    # ---- reference return level ----
    u = -np.log(1 - 1/T_ref)

    small_xi = np.abs(xi_s) < 1e-6
    z_ref = np.empty(ns)

    # Gumbel case
    z_ref[small_xi] = mu_ref[small_xi] - sigma_s[small_xi] * np.log(-np.log(1 - 1/T_ref))

    # general case
    z_ref[~small_xi] = (
        mu_ref[~small_xi]
        + sigma_s[~small_xi]/xi_s[~small_xi]
        * (u**(-xi_s[~small_xi]) - 1)
    )

    # ---- compute μ(t) for all years ----
    t = (years - mean_year) / std_year
    mu_all = mu0_s[:,None] + mu1_s[:,None] * t[None,:]

    # ---- return periods ----
    xi_safe = np.clip(xi_s[:, None], -0.1, 0.1)         # optional: tighter bounds if needed
    small_xi = np.abs(xi_safe) < 1e-6
    
    term = 1 + xi_safe[:,None] * (z_ref[:,None] - mu_all)
    term = term / sigma_s[:, None]
    # prevent invalid values
    term = np.maximum(term, 1e-6)

    exp_term = np.where(
        small_xi[:, None],
        np.exp(-term),                                  # Gumbel-like safe
        np.minimum(term**(-1/xi_safe[:, None]), 1e10)   # cap extreme
    )
    F = np.clip(np.exp(-exp_term), 0, 1-1e-10)
    
    rp = 1/(1-F)
    rp = np.clip(rp, 1, max_T)

    # ---- statistics ----
    mean = np.mean(rp, axis=0)
    low = np.percentile(rp, 5, axis=0)
    high = np.percentile(rp, 95, axis=0)

    return DataFrame({
        "return_period_mean": mean,
        "return_period_lower": low,
        "return_period_upper": high
    }, index=years)
    
    
def pooled_gev_per_single_location(
    loc_id,
    location_data,
    return_periods,
    ls_t_eval,
    location_info,
    ref_year_rp=1961, 
    t_ref_rp=50,
    min_years=10,
    confidence_level_pc:float=0.9
):
    """
    Run stationary / non-stationary GEV analysis for a single location.
    Designed for multiprocessing with joblib.
    """

    lon_loc = location_data.lon.unique()[0]
    lat_loc = location_data.lat.unique()[0]

    logger.info(f'GEV analysis for location {loc_id} · {location_info}')

    # -------------------------------------------------------
    # Extract annual maxima
    annual_max = extract_annual_maxima_at_location(location_data, lon=lon_loc, lat=lat_loc)

    if len(annual_max) < min_years:
        logger.info(
            'WARNING - not enough data (<%s) for location %s (lon|lat · %s|%s)', min_years, loc_id, lon_loc, lat_loc
        )
        print(
            'WARNING - not enough data (<%s) for location %s (lon|lat · %s|%s)', min_years, loc_id, lon_loc, lat_loc
        )

    years = annual_max['year'].values
    data = annual_max['annual_max'].values

    # -------------------------------------------------------
    # Fit GEV with uncertainty
    pooled_gev, _ = fit_pooled_gev_with_uncertainty(
        loc_id=loc_id,
        data=data,
        years=years,
        print_msg=False,
    )

    logger.info(
        f"\tLoc {loc_id} | → Stationary GEV done (success {pooled_gev.get('stationary') is not None})"
        f"\tLoc {loc_id} | → Non-stationary GEV done (success {pooled_gev.get('nonstationary') is not None})"
        )

    stationary = pooled_gev.get("stationary")
    nonstationary = pooled_gev.get("nonstationary")
    if stationary is None or nonstationary is None:
        logger.info(
            f'\tLoc {loc_id} | → either stationary or non-stationary GEV missing; skipping location...'
        )
        result = {
            'location_info': location_info,
            'LatLon': (lat_loc, lon_loc),
            'data': annual_max,
            'stationary': pooled_gev['stationary'],
            'nonstationary': pooled_gev['nonstationary'],
            'model_comparison': None,
            'return_levels': None
        }
        return loc_id, result

    # -------------------------------------------------------
    logger.info(f'\tLoc {loc_id} | Compare Models...')
    comparison = compare_stationary_nonstationary(
        pooled_gev['stationary'],
        pooled_gev['nonstationary'],
        annual_max
    )

    # -------------------------------------------------------
    logger.info(f'\tLoc {loc_id} | Compute Return Levels...')
    df_all_return_levels = compute_all_return_levels(
        pooled_gev['stationary'], 
        pooled_gev['nonstationary'],
        return_periods,
        ls_t_eval,
        confidence_level_pc
    )
    
    # -------------------------------------------------------
    logger.info(f'\tLoc {loc_id} | Compute Return Period of the {t_ref_rp}-year event at {ref_year_rp}...')
    rp_ns = rp_uncertainty_monte_carlo(
        params=pooled_gev['nonstationary']['params_hat'], cov=pooled_gev['nonstationary']['cov'], years=years, 
        mean_year=pooled_gev['nonstationary']['years_mean'], std_year=pooled_gev['nonstationary']['years_std'],
        ref_year=ref_year_rp, T_ref=t_ref_rp
        )
    pooled_gev['nonstationary'].update({'return_period': rp_ns})

    logger.info(
        f"Loc {loc_id} | Return Period Overview\n"
        f" mean: {rp_ns.min().return_period_mean:.2f} – {rp_ns.max().return_period_mean:.2f} years\n"
        f" lower: {rp_ns.min().return_period_lower:.2f} – {rp_ns.max().return_period_lower:.2f} years\n"
        f" upper: {rp_ns.min().return_period_upper:.2f} – {rp_ns.max().return_period_upper:.2f} years\n\n"
    )
    

    # -------------------------------------------------------
    result = {
        'location_info': location_info,
        'LatLon': (lat_loc, lon_loc),
        'data': annual_max,
        'stationary': pooled_gev['stationary'],
        'nonstationary': pooled_gev['nonstationary'],
        'model_comparison': comparison,
        'return_levels': df_all_return_levels,
    }

    return loc_id, result


# --------------- ANNUAL-STATIONARY --------------------
def mle_fitting(data, n, ls_notes):
    c, location, scale = stats.genextreme.fit(data)
    shape = -c

    logpdf = stats.genextreme.logpdf(data, c, location, scale)
    if not np.all(np.isfinite(logpdf)):
        ls_notes.append("Non-finite log-likelihood (GEV support violation)")
    ll = np.sum(logpdf)

    n_params = 3
    aic = 2 * n_params - 2 * ll
    bic = np.log(n) * n_params - 2 * ll

    if abs(shape) < 0.05:
        dist_type, tail = "Gumbel (Type I)", "Exponential"
    elif shape > 0:
        dist_type, tail = "Fréchet (Type II)", "Heavy (polynomial)"
    else:
        dist_type, tail = "Weibull (Type III)", "Light (bounded)"

    return {
        "shape": shape,
        "location": location,
        "scale": scale,
        "n_obs": n,
        "log_likelihood": ll,
        "aic": aic,
        "bic": bic,
        "dist_type": dist_type,
        "tail_behavior": tail,
    }, ls_notes


def fit_annual_gev_mle(df, ls_notes, col_data='storm_surge', col_year='sim_year'):
    """
    Fit stationary GEV MLEs for each year in the dataframe.
    Returns a dataframe with columns: ['year', 'mu', 'sigma', 'xi']
    """
    annual_max_per_year = df.groupby(col_year)[col_data].apply(list)
    records = []

    for year, data in annual_max_per_year.items():
        result, ls_notes = mle_fitting(data, len(data), ls_notes)
        records.append((year, result))
    
    flattened = []
    for year, vals in records:
        row = {'year': year, **vals} 
        flattened.append(row)

    return DataFrame(flattened).sort_values('year').reset_index(drop=True)


def regress_location_trend(annual_df):
    """
    Fits linear trend: mu ~ standardized year
    Returns dict with mu0, mu1, SEs for both, plus mean and std of years
    """
    years = annual_df['year'].values
    years_mean = years.mean()
    years_std = years.std(ddof=0)  # same as np.std with population formula

    x_std = (years - years_mean) / years_std

    X = sm.add_constant(x_std)  # add intercept
    y = annual_df['location'].values
    model = sm.OLS(y, X).fit()

    return {
        'mu0': model.params[0],
        'mu1': model.params[1],
        'mu0_se': model.bse[0],
        'mu1_se': model.bse[1],
        'years_mean': years_mean,
        'years_std': years_std
    }
    

def fit_location(loc_id, df, ls_notes):
    """
    Fits annual stationary GEV MLEs and computes mu trend regression
    """
    logger.info('Loc {loc_id} | compute annual GEV using MLE...')
    annual_df = fit_annual_gev_mle(df, ls_notes=ls_notes)
    logger.info('Loc {loc_id} | compute location trend regression...')
    trend_results = regress_location_trend(annual_df)

    return loc_id, {
        'annual_mle': annual_df,
        'mu_trend': trend_results
    }


def fit_all_locations(dic_data_per_location, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(fit_location)(loc_id, df, None) for loc_id, df in dic_data_per_location.items()
    )
    return dict(results)


def annual_stationary_trend(annual_df, factor_m_to_mm=1000):
    """
    Compute weighted linear trend of annual GEV location parameter (mu) with standardized years.
    
    Parameters:
        annual_df : pd.DataFrame
            Must contain columns ['year', 'location', 'n_obs']
        factor_m_to_mm : float
            Factor to multiply mu (default converts m -> mm)
    
    Returns:
        dict with:
            mu_trend : dict with mu0, mu1, mu0_se, mu1_se
            mu_fit : np.array of fitted values (in mm)
            mu_ci_upper, mu_ci_lower : np.array of 95% CI (in mm)
            years_std : standardized years used
    """
    
    x = annual_df['year'].values
    y = annual_df['location'].values
    weights = annual_df['n_obs'].values
    
    x_mean = np.mean(x)
    x_std_val = np.std(x, ddof=1)
    x_std = (x - x_mean) / x_std_val
    
    X = sm.add_constant(x_std)
    
    model = sm.WLS(y, X, weights=weights).fit()
    
    mu0_mm = model.params[0] * factor_m_to_mm
    mu1_mm = model.params[1] * factor_m_to_mm
    mu0_se_mm = model.bse[0] * factor_m_to_mm
    mu1_se_mm = model.bse[1] * factor_m_to_mm
    
    mu_fit = mu0_mm + mu1_mm * x_std
    
    mu_ci_upper = mu_fit + 1.96 * mu1_se_mm * x_std
    mu_ci_lower = mu_fit - 1.96 * mu1_se_mm * x_std
    
    return {
        'mu_trend': {
            'mu0': mu0_mm,
            'mu1': mu1_mm,
            'mu0_se': mu0_se_mm,
            'mu1_se': mu1_se_mm
        },
        'mu_fit': mu_fit,
        'mu_ci_upper': mu_ci_upper,
        'mu_ci_lower': mu_ci_lower,
        'years_std': x_std,
        'x_mean': x_mean,
        'x_std_val': x_std_val
    }


def prepare_for_regression(
    annual_stationary, nonstationary, years_mean, years_std, hindcast_start, hindcast_end, confidence_level_pc, 
    factor_m_to_mm
    ):
    
    years_ = np.arange(hindcast_start, hindcast_end+1)
    years_autoscaled = (years_ - years_mean) / years_std

    x_ans = annual_stationary['annual_mle'].year.values
    y_ans = annual_stationary['annual_mle'].location.values*factor_m_to_mm
    weights_ans = annual_stationary['annual_mle'].n_obs.values

    # ------------ stationary -------------------
    results_reg_annual_stat = annual_stationary_trend(annual_stationary['annual_mle'])

    intercept_ans = results_reg_annual_stat['mu_trend']['mu0']
    slope_ans = results_reg_annual_stat['mu_trend']['mu1']

    # ------------ non-stationary -------------------
    mu0_ns = nonstationary['params_hat'][0]*factor_m_to_mm
    mu1_ns = nonstationary['params_hat'][1]*factor_m_to_mm

    mu_ns = mu0_ns + mu1_ns * years_autoscaled

    z = norm.ppf(1 - (1-confidence_level_pc)/2) 
    cov = nonstationary['cov'][:2, :2] * factor_m_to_mm**2
    mu_var = cov[0,0] + 2 * cov[0,1] * years_autoscaled + cov[1,1] * years_autoscaled**2  # shape (67,)
    mu_ns_ci_upper = mu_ns + z * np.sqrt(mu_var)
    mu_ns_ci_lower = mu_ns - z * np.sqrt(mu_var)
    
    return years_, {
        'stationary': (x_ans, y_ans, weights_ans, results_reg_annual_stat, slope_ans, intercept_ans), 
        'nonstationary': (mu_ns, mu1_ns, mu0_ns, mu_ns_ci_lower, mu_ns_ci_upper),
        }


