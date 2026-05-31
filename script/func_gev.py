import logging
import random
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from pandas import DataFrame
from scipy import linalg, stats
from scipy.optimize import minimize
from scipy.stats import chi2, genextreme, norm
from statsmodels.tools.numdiff import approx_hess

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger("mp_gev_analysis")

BOUNDS_SHAPE = (-2, 2)
BOUNDS_LOCATION = (None, None)
BOUNDS_SCALE = (1e-6, None)


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
    init = np.array([-init_shape, np.mean(data), np.std(data, ddof=1)])
    bounds = [BOUNDS_SHAPE, BOUNDS_LOCATION, BOUNDS_SCALE]
    
    res = minimize(lambda p: gev_neg_loglik(p, data), init, bounds=bounds, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError("GEV MLE did not converge: " + res.message)
    
    c_ml, loc_ml, scale_ml = res.x
    shape_ml = -c_ml               
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
    gev_params: dict, data: np.ndarray, is_nonstationary:bool, years_scaled: bool, years: np.ndarray = None
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
        # Non-stationary location μ(t) = μ0 + μ1 * t; with centering around year_mean of years or without 
        if years_scaled is True:
            t = years - gev_params['years_mean']
        else:
            t = years
            
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
    loc_id, data, x0, uncertainty, years, years_scaled, ls_notes, print_msg, B, seed=None
    ):
    """_summary_

    Args:
        loc_id (_type_): _description_
        data (_type_): _description_
        x0 (_type_): _description_
        uncertainty (_type_): _description_
        years (_type_): _description_
        years_scaled (_type_): _description_
        ls_notes (_type_): _description_
        print_msg (_type_): _description_
        B (_type_): _description_
        seed (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if uncertainty is None:
        uncertainty = 'delta'

    t = years - years.mean() if years_scaled is True else years
    #x0 = [stationary['location'], 0.0, stationary['scale'], stationary['shape']]
    
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
                gev_params=nonstationary, data=data, years=years, years_scaled=years_scaled, is_nonstationary=True
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
                message=f"Loc {loc_id} | Delta Method failed, falling back to bootstrapping",
                ls_notes=ls_notes, print_msg=print_msg
                )
            uncertainty = "bootstrap"

    if uncertainty == "bootstrap":
        ls_notes = print_and_append_notes(
            message=f"Loc {loc_id} | Computing bootstrap uncertainty for non-stationary GEV...",
            ls_notes=ls_notes, print_msg=print_msg
            )
        
        if seed is not None:
            random.seed(seed)
        
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


def fit_pooled_stationary_gev_with_uncertainty(
    loc_id, data, year: int|None= None, B:int=300, seed:int=42, print_msg=False
    ):
    ls_notes = []
    n = len(data)

    ls_notes = print_and_append_notes(
        message=f'\t\tLoc {loc_id} | Compute stationary GEV incl uncertainty', 
        ls_notes=ls_notes, print_msg=print_msg
        )   

    if n < 10:
        message = (
            f'\n\t Loc {loc_id} | Warning: Only {n} observations in {year} if year else '
            + '. Need at least 10 for reliable GEV fit.'
        )
        ls_notes = print_and_append_notes(message=message, ls_notes=ls_notes, print_msg=print_msg)

    try:
        stationary, ls_notes = fit_stationary_gev_incl_uncertainty(
            loc_id=loc_id, data=data, B=B, seed=seed, ls_notes=ls_notes, print_msg=print_msg, 
            )
        pooled_gev = dict({'stationary': stationary})

    except Exception as e:
        message = f"Failed to conduct stationary GEV fitting due to error: {e}"
        ls_notes.append(message)
        return {'stationary': None}, ls_notes
    
    return pooled_gev, ls_notes


def fit_pooled_nonstat_gev_with_uncertainty(
    loc_id, data, x0_stat, years_scaled: bool, year: int|None= None, years: list | None = None, 
    uncertainty_ns:str|None ='delta', B:int=300, seed:int=42, print_msg=False
    ):
    ls_notes = []
    n = len(data)
    
    if n < 10:
        message = (
            f'\n\t Loc {loc_id} | Warning: Only {n} observations'
            + f' in {year} if year else '
            + '. Need at least 10 for reliable GEV fit.'
        )
        ls_notes = print_and_append_notes(message=message, ls_notes=ls_notes, print_msg=print_msg)

    try:
        message = f'\t\tLoc {loc_id} | Compute non-stationary GEV incl uncertainty'
        ls_notes = print_and_append_notes(message=message, ls_notes=ls_notes, print_msg=print_msg)

        if n < 20:
            message = f'Loc {loc_id} | Warning: Non-stationary GEV needs ≥20 observations but has {n}, '
            message += 'skipping non-stationary GEV...'
            ls_notes = print_and_append_notes(message=message, ls_notes=ls_notes, print_msg=print_msg)
            return {'nonstationary': None}, ls_notes

        # x0_stat = [stationary['location'], 0.0, stationary['scale'], stationary['shape']]
        nonstationary, ls_notes = fit_non_stationary_gev_incl_uncertainty(
            loc_id=loc_id, data=data, x0=x0_stat, years=years, uncertainty=uncertainty_ns, 
            ls_notes=ls_notes, print_msg=print_msg, years_scaled=years_scaled, B=B, seed=seed
            )
        pooled_gev = dict({'nonstationary': nonstationary})

    except Exception as e:
        message = f"Failed to conduct GEV fitting due to error: {e}"
        ls_notes.append(message)
        return {'nonstationary': None}, ls_notes
    
    return pooled_gev, ls_notes


def fit_pooled_gev_with_uncertainty(
    loc_id, data, years_scaled: bool, year: int|None= None, years: list | None = None, uncertainty_ns:str|None ='delta', 
    B:int=300, seed:int=42, print_msg=False
    ):
    ls_notes = []
    n = len(data)

    # ---- STEP 1: Get stationary ----
    ls_notes = print_and_append_notes(
        message=f'\t\tLoc {loc_id} | Compute stationary GEV incl uncertainty', 
        ls_notes=ls_notes, print_msg=print_msg
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

        x0_stat = [stationary['location'], 0.0, stationary['scale'], stationary['shape']]
        
        nonstationary, ls_notes = fit_non_stationary_gev_incl_uncertainty(
            loc_id=loc_id, data=data, x0=x0_stat, years=years, uncertainty=uncertainty_ns, 
            ls_notes=ls_notes, print_msg=print_msg, years_scaled=years_scaled, B=B, seed=seed
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
    
    t_scaled = (t_eval - years_mean)
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
    t_scaled = (t_eval - years_mean)

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
    # T = np.clip(T, 1.0, max_T)
    T[T > max_T] = np.nan
    T[T < 1.0] = np.nan
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
    mean_rp = np.nanmedian(sims, axis=0)
    low_rp = np.nanpercentile(sims, 5, axis=0)
    high_rp = np.nanpercentile(sims, 95, axis=0)

    return DataFrame({
        'year': years,
        'return_period_median': mean_rp,
        'return_period_lower': low_rp,
        'return_period_upper': high_rp
    }).set_index('year')

    
def pooled_gev_per_single_location(
    loc_id,
    location_data,
    return_periods,
    ls_t_eval,
    location_info,
    B,
    seed,
    years_scaled,
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
        B=B, 
        seed=seed, 
        print_msg=False,
        years_scaled=years_scaled
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
        f" mean: {rp_ns.min().return_period_median:.2f} – {rp_ns.max().return_period_median:.2f} years\n"
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


def regress_location_trend(annual_df, years_scaled):
    """
    Fits linear trend: mu ~ standardized year
    Returns dict with mu0, mu1, SEs for both, plus mean and std of years
    """
    years = annual_df['year'].values
    years_mean = years.mean()
    years_std = years.std(ddof=0)  # same as np.std with population formula

    x_std = years - years_mean if years_scaled is True else years

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
    

def fit_location(loc_id, df, years_scaled, ls_notes):
    """
    Fits annual stationary GEV MLEs and computes mu trend regression
    """
    logger.info(f'Loc {loc_id} | compute annual GEV using MLE...')
    annual_df = fit_annual_gev_mle(df, ls_notes=ls_notes)
    logger.info(f'Loc {loc_id} | compute location trend regression...')
    trend_results = regress_location_trend(annual_df, years_scaled)

    return loc_id, {
        'annual_mle': annual_df,
        'mu_trend': trend_results
    }


def fit_all_locations(dic_data_per_location, years_scaled, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(fit_location)(loc_id, df, years_scaled, None) for loc_id, df in dic_data_per_location.items()
    )
    return dict(results)


def annual_stationary_trend(x_range_years, annual_df, confidence_level_pc, factor_m_to_mm):
    """_summary_
    Compute weighted linear trend of annual GEV location parameter (mu) with standardized years.

    Args:
        x_range_years (_type_): _description_
        annual_df (_type_): _description_
        confidence_level_pc (_type_): _description_
        factor_m_to_mm (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    y = annual_df['location'].values
    weights = annual_df['n_obs'].values
    X = sm.add_constant(x_range_years)
    
    # weighted least-square regression with parameters given in m and years (auto-scaled, if selected)
    model = sm.WLS(y, X, weights=weights).fit()
    
    mu0_mm = model.params[0] * factor_m_to_mm
    mu1_mm = model.params[1] * factor_m_to_mm
    mu0_se_mm = model.bse[0] * factor_m_to_mm
    mu1_se_mm = model.bse[1] * factor_m_to_mm
    print(f'annual stat LSR results {mu0_mm:.2f}mm + {mu1_mm:.2f}mm • t')

    mu_fit = mu0_mm + mu1_mm * x_range_years

    z = norm.ppf(1 - (1-confidence_level_pc)/2)
    mu_ci_upper = mu_fit + z * mu1_se_mm * x_range_years
    mu_ci_lower = mu_fit - z * mu1_se_mm * x_range_years
    
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
    }


def non_stationary_trend(x_range_years, nonstationary, confidence_level_pc, factor_m_to_mm):
    """_summary_

    Args:
        x_range_years (_type_): _description_
        nonstationary (_type_): _description_
        confidence_level_pc (_type_): _description_
        factor_m_to_mm (_type_): _description_

    Returns:
        _type_: _description_
    """
    mu0_ns = nonstationary['params_hat'][0]*factor_m_to_mm
    mu1_ns = nonstationary['params_hat'][1]*factor_m_to_mm

    mu_ns = mu0_ns + mu1_ns * x_range_years

    z = norm.ppf(1 - (1-confidence_level_pc)/2)
    cov = nonstationary['cov'][:2, :2] * factor_m_to_mm**2
    mu_var = cov[0,0] + 2 * cov[0,1] * x_range_years + cov[1,1] * x_range_years**2  
    mu_ns_ci_upper = mu_ns + z * np.sqrt(mu_var)
    mu_ns_ci_lower = mu_ns - z * np.sqrt(mu_var)
    
    print(f'non-stationary regression results {mu0_ns:.2f}mm + {mu1_ns:.2f}mm • t')
    return {
        'mu_trend': {
            'mu0': mu0_ns,
            'mu1': mu1_ns,
        },
        'mu_fit': mu_ns,
        'mu_ci_upper': mu_ns_ci_upper,
        'mu_ci_lower': mu_ns_ci_lower,
    }


# --------------- TREND-REGRESSION --------------------
def prepare_for_regression(
    annual_stationary, nonstationary, confidence_level_pc, factor_m_to_mm, years_scaled
    ):
    # ------------ general preparation -------------------    
    x_ans = annual_stationary['annual_mle'].year.values
    x_mean = np.mean(x_ans)
    x_std_val = np.std(x_ans, ddof=0)
    x_std = x_ans - x_mean              # possible for centering around mean but not standardization to keep the unit
    
    # decide for year standardization or not
    years_ = x_std if years_scaled is True else x_ans
    
    y_ans = annual_stationary['annual_mle'].location.values*factor_m_to_mm
    weights_ans = annual_stationary['annual_mle'].n_obs.values
    
    # ------------ stationary -------------------
    results_reg_annual_stat = annual_stationary_trend(
        x_range_years=years_, annual_df=annual_stationary['annual_mle'], 
        confidence_level_pc=confidence_level_pc, factor_m_to_mm=factor_m_to_mm
        )

    intercept_ans = results_reg_annual_stat['mu_trend']['mu0']
    slope_ans = results_reg_annual_stat['mu_trend']['mu1']

    # ------------ non-stationary -------------------
    results_reg_non_stat = non_stationary_trend(
        x_range_years=years_, nonstationary=nonstationary, 
        confidence_level_pc=confidence_level_pc, factor_m_to_mm=factor_m_to_mm
    )
    
    return {
        'years_regression': years_, 
        'scaled': years_scaled, 
        'mean': x_mean, 
        'std': x_std_val
        }, {
        'stationary': (
            x_ans, 
            y_ans, 
            weights_ans, 
            results_reg_annual_stat, 
            slope_ans, 
            intercept_ans
            ), 
        'nonstationary': (
            results_reg_non_stat['mu_fit'], 
            results_reg_non_stat['mu_trend']['mu1'],
            results_reg_non_stat['mu_trend']['mu0'],
            results_reg_non_stat['mu_ci_lower'],
            results_reg_non_stat['mu_ci_upper']
            ),
        }


# -------------- 50-year-event-evolution --------------------
def gev_mu_at_year(year, mu0, mu1, years_mean):
    """
    Location parameter for a non-stationary GEV with centered time.

    The model is:
        mu(t) = mu0 + mu1 * (year - years_mean)

    Important:
    Do not divide by years_std unless the model was fitted using standardized time.
    In the current workflow, years are centered but not standardized.
    """
    return mu0 + mu1 * (np.asarray(year, dtype=float) - years_mean)


def gev_return_level(T, mu, sigma, xi):
    """
    Return level z_T for return period T under a GEV distribution.

    Uses scipy's genextreme convention c = -xi.
    The survival probability is 1 / T.
    """
    if sigma <= 0:
        return np.nan

    return genextreme.isf(1.0 / T, c=-xi, loc=mu, scale=sigma)


def gev_return_period(z, mu, sigma, xi, max_return_period=1e4):
    """
    Equivalent return period of a fixed level z under GEV(mu, sigma, xi).

    Return period is:
        T = 1 / P(X > z)

    Uses survival function directly for numerical stability.
    """
    if sigma <= 0:
        return np.nan

    sf = genextreme.sf(z, c=-xi, loc=mu, scale=sigma)

    if not np.isfinite(sf) or sf <= 0:
        return np.nan

    sf = np.clip(sf, 1.0 / max_return_period, 1.0)
    T = 1.0 / sf

    if T < 1.0 or not np.isfinite(T):
        return np.nan

    return T


def return_period_evolution_1961_50yr(
    nonstationary,
    years=None,
    ref_year=1961,
    ref_return_period=50,
    max_return_period=1e4,
):
    """
    Compute the return-period evolution of the 1961 50-year event.

    Parameters
    ----------
    nonstationary : dict
        Dictionary containing fitted non-stationary GEV results.
        Required keys:
            - params_hat = [mu0, mu1, sigma, xi]
            - years_mean

    years : array-like, optional
        Years over which to evaluate the equivalent return period.
        If None, uses 1961--2026.

    ref_year : int
        Base year used to define the historical event level.

    ref_return_period : float
        Return period used to define the historical reference level.

    max_return_period : float
        Maximum return period used to avoid unstable numerical tail values.

    Returns
    -------
    pandas.DataFrame
        Columns:
            - year
            - mu_t
            - return_period
            - z_ref
            - mu_ref
            - delta_mu_from_ref
    """
    if years is None:
        years = np.arange(1961, 2027)
    else:
        years = np.asarray(years, dtype=float)

    mu0, mu1, sigma, xi = np.asarray(nonstationary["params_hat"], dtype=float)
    years_mean = float(nonstationary["years_mean"])

    # 1. GEV location in the base year
    mu_ref = gev_mu_at_year(ref_year, mu0, mu1, years_mean)

    # 2. Physical storm-surge level corresponding to the base-year 50-year event
    z_ref = gev_return_level(ref_return_period, mu_ref, sigma, xi)

    # 3. Internal consistency check: in the reference year, this must return approximately 50 years
    rp_check = gev_return_period(z_ref, mu_ref, sigma, xi, max_return_period=max_return_period)
    if not np.isclose(rp_check, ref_return_period, rtol=1e-4, atol=1e-4):
        raise RuntimeError(
            f"Return-period consistency check failed: expected {ref_return_period}, got {rp_check}"
        )

    # 4. Re-evaluate return period of the same fixed level in all years
    mu_t = gev_mu_at_year(years, mu0, mu1, years_mean)
    rp_t = np.array([
        gev_return_period(z_ref, mu, sigma, xi, max_return_period=max_return_period)
        for mu in mu_t
    ])

    return pd.DataFrame({
        "year": years.astype(int),
        "mu_t": mu_t,
        "return_period": rp_t,
        "z_ref": z_ref,
        "mu_ref": mu_ref,
        "delta_mu_from_ref": mu_t - mu_ref,
    })
    

def return_period_evolution_1961_50yr_mc(
    nonstationary,
    years=None,
    ref_year=1961,
    ref_return_period=50,
    confidence_level=0.90,
    nsim=5000,
    seed=42,
    max_return_period=1e4,
    max_abs_xi=0.8,
):
    """
    Monte Carlo uncertainty propagation for the return-period evolution
    of the 1961 50-year event.

    The function samples [mu0, mu1, sigma, xi] from the fitted covariance matrix,
    filters unstable samples, and recomputes the equivalent return-period curve.

    Returns
    -------
    pandas.DataFrame
        Columns:
            - year
            - return_period_median
            - return_period_lower
            - return_period_upper
            - z_ref_median
            - n_valid_samples
    """
    if years is None:
        years = np.arange(1961, 2027)
    else:
        years = np.asarray(years, dtype=float)

    params_hat = np.asarray(nonstationary["params_hat"], dtype=float)
    cov = np.asarray(nonstationary["cov"], dtype=float)
    years_mean = float(nonstationary["years_mean"])

    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(params_hat, cov, size=nsim)

    # Filter physically/numerically unstable samples
    valid = (
        np.isfinite(samples).all(axis=1)
        & (samples[:, 2] > 0)                 # sigma > 0
        & (np.abs(samples[:, 3]) <= max_abs_xi)  # avoid extreme tail instability
    )
    samples = samples[valid]

    sims = np.full((len(samples), len(years)), np.nan)
    z_refs = np.full(len(samples), np.nan)

    for i, (mu0, mu1, sigma, xi) in enumerate(samples):
        mu_ref = gev_mu_at_year(ref_year, mu0, mu1, years_mean)
        z_ref = gev_return_level(ref_return_period, mu_ref, sigma, xi)
        z_refs[i] = z_ref

        # Consistency check for each sample
        rp_check = gev_return_period(z_ref, mu_ref, sigma, xi, max_return_period=max_return_period)
        if not np.isfinite(rp_check) or not np.isclose(rp_check, ref_return_period, rtol=1e-3, atol=1e-3):
            continue

        mu_t = gev_mu_at_year(years, mu0, mu1, years_mean)
        sims[i, :] = [
            gev_return_period(z_ref, mu, sigma, xi, max_return_period=max_return_period)
            for mu in mu_t
        ]

    alpha = 1.0 - confidence_level

    valid_rows = np.isfinite(sims).any(axis=1)
    if valid_rows.sum() == 0:
        return pd.DataFrame({
            "year": years.astype(int),
            "return_period_median": np.nan,
            "return_period_lower": np.nan,
            "return_period_upper": np.nan,
            "z_ref_median": np.nan,
            "n_valid_samples": 0,
        })
        
    sim_valid = sims[valid_rows]
    z_refs_valid = z_refs[valid_rows]
    return pd.DataFrame({
        "year": years.astype(int),
        "return_period_median": np.nanmedian(sim_valid, axis=0),
        "return_period_lower": np.nanquantile(sim_valid, alpha / 2, axis=0),
        "return_period_upper": np.nanquantile(sim_valid, 1 - alpha / 2, axis=0),
        "z_ref_median": np.nanmedian(z_refs_valid),
        "n_valid_samples": np.sum(np.isfinite(sim_valid).any(axis=1)),
    })


# -------------- Return Period of 50-year event FOR ALL SITES --------------------
def _compute_rp2026_one_site(
    site_id,
    nonstat,
    location_geo_info,
    eval_year,
    ref_year,
    ref_return_period,
    nsim,
    seed,
):
    if nonstat is None:
        return None

    try:
        rp_df = return_period_evolution_1961_50yr_mc(
            nonstationary=nonstat,
            years=np.array([eval_year]),
            ref_year=ref_year,
            ref_return_period=ref_return_period,
            nsim=nsim,
            seed=None if seed is None else seed + int(site_id),
        )

        lat, lon = location_geo_info.get(site_id, (None, None))

        return {
            "site_id": site_id,
            "lat": lat,
            "lon": lon,
            "return_period_2026": float(rp_df["return_period_median"].iloc[0]),
            "return_period_lower": float(rp_df["return_period_lower"].iloc[0]),
            "return_period_upper": float(rp_df["return_period_upper"].iloc[0]),
            "z_ref_median": float(rp_df["z_ref_median"].iloc[0]),
        }

    except Exception as e:
        return {
            "site_id": site_id,
            "lat": None,
            "lon": None,
            "return_period_2026": np.nan,
            "return_period_lower": np.nan,
            "return_period_upper": np.nan,
            "z_ref_median": np.nan,
            "error": str(e),
        }


def compute_return_period_2026_all_sites_parallel(
    nonstat_all,
    location_geo_info,
    eval_year=2026,
    ref_year=1961,
    ref_return_period=50,
    nsim=1000,
    n_jobs=-1,
    seed=42,
):
    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        batch_size=50,
        verbose=10,
    )(
        delayed(_compute_rp2026_one_site)(
            site_id,
            nonstat,
            location_geo_info,
            eval_year,
            ref_year,
            ref_return_period,
            nsim,
            seed,
        )
        for site_id, nonstat in nonstat_all.items()
    )

    rows = [r for r in results if r is not None]
    return pd.DataFrame(rows)