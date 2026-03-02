import logging
import os
import random
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import func_plotting as dbplt
import func_utils as ut
import numpy as np
import statsmodels.api as sm
from IPython.display import Markdown, display
from joblib import Parallel, delayed
from numpy import (arange, array, asarray, diag, exp, finfo, float64,
                   full_like, generic, inf, isfinite, isnan, linalg, linspace,
                   log, mean, nan, ndarray, ones_like, random, repeat, sqrt,
                   std, vstack, zeros, zeros_like)
from numpy.linalg import inv
from pandas import DataFrame, concat, to_numeric
from scipy import linalg, optimize, stats
from scipy.optimize import approx_fprime, minimize
from scipy.stats import chi2, genextreme, norm
from statsmodels.tools.numdiff import approx_hess
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger("GEVanalysis_v2")
_LOCATION_LABELS = None


def extract_annual_maxima_unique(
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

    return (
        subset
        .groupby('sim_year')['storm_surge']
        .max()
        .reset_index()
        .rename(columns={'sim_year': 'year', 'storm_surge': 'annual_max'})
        .sort_values('year')
    )


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


def fit_stationary_gev(
    data: ndarray, year:int | None = None, print_msg:bool=False
    ) -> tuple[dict | None, list[str]]:
    """
    Fit stationary GEV using Maximum Likelihood Estimation.
    
    This assumes GEV parameters are constant over time.
    RECOMMENDED: Use when you have 60+ years × 2 members = 120 points
    
    Parameters:
    -----------
    data : np.ndarray
        Annual maxima values
        
    Returns:
    --------
    dict : GEV parameters and diagnostics
    """
    ls_notes = []
    if len(data) < 10:
        if year is not None:
            message = f"Warning: Only {len(data)} observations in {year}. Need at least 10 for reliable GEV fit."
        else:
            message = f"Warning: Only {len(data)} observations. Need at least 10 for reliable GEV fit."
        if print_msg:
            logger.info(message)
        ls_notes.append(message)
        return None, ls_notes

    try:
        c, loc, scale = stats.genextreme.fit(data)
        shape = -c

        logpdf = stats.genextreme.logpdf(data, c, loc, scale)
        if not np.all(isfinite(logpdf)):
            ls_notes.append("Non-finite log-likelihood (GEV support violation)")
        ll = np.sum(logpdf)

        n_params = 3
        aic = 2 * n_params - 2 * ll
        bic = log(len(data)) * n_params - 2 * ll

        if abs(shape) < 0.05:
            dist_type, tail = "Gumbel (Type I)", "Exponential"
        elif shape > 0:
            dist_type, tail = "Fréchet (Type II)", "Heavy (polynomial)"
        else:
            dist_type, tail = "Weibull (Type III)", "Light (bounded)"

        return {
            'shape': shape,
            'location': loc,
            'scale': scale,
            'n_obs': len(data),
            'log_likelihood': ll,
            'aic': aic,
            'bic': bic,
            'dist_type': dist_type,
            'tail_behavior': tail
        }, ls_notes
    except Exception as e:
        logger.debug('Failed to conduct GEV fitting due to error: %s',e)
        return None, ls_notes


def fit_nonstationary_gev(
    years: ndarray, data: ndarray, trend_params: str = 'location', print_msg:bool=False
    ) -> tuple[dict, list]:
    """
    Fit non-stationary GEV where parameters vary linearly with time.
    
    This allows detection of trends in extreme values.
    RECOMMENDED: Use to test if sea level rise affects extremes
    
    Parameters:
    -----------
    years : np.ndarray
        Years corresponding to each observation
    data : np.ndarray
        Annual maxima values
    trend_params : str
        Which parameters have trends: 'location', 'scale', or 'both'
        
    Returns:
    --------
    dict : Non-stationary GEV parameters and diagnostics
    """
    ls_notes = []
    if len(data) < 20:
        message = f"Warning: Non-stationary GEV needs ≥20 obs. Have {len(data)}."
        if print_msg:
            logger.info(message)
        ls_notes.append(message)
        return None, ls_notes

    t = (years - years.mean()) / years.std()

    def neg_log_likelihood(params):
        """Negative log-likelihood for optimization."""
        if trend_params == 'location':
            mu0, mu1, sigma, xi = params
            mu_t = mu0 + mu1 * t
            sigma_t = full_like(t, sigma)
        elif trend_params == 'scale':
            mu, sigma0, sigma1, xi = params
            mu_t = full_like(t, mu)
            sigma_t = sigma0 + sigma1 * t
        elif trend_params == 'both':
            mu0, mu1, sigma0, sigma1, xi = params
            mu_t = mu0 + mu1 * t
            sigma_t = sigma0 + sigma1 * t
        else:
            raise ValueError("trend_params must be 'location', 'scale', or 'both'")

        if np.any(sigma_t <= 0):
            return inf

        z = (data - mu_t) / sigma_t

        if abs(xi) < 1e-10:  # Gumbel case
            ll = -np.sum(log(sigma_t)) - np.sum(z) - np.sum(exp(-z))
        else:
            term = 1 + xi * z
            if np.any(term <= 0):
                return inf
            ll = (-np.sum(log(sigma_t)) -
                    (1 + 1/xi) * np.sum(log(term)) -
                    np.sum(term**(-1/xi))
                    )
        return -ll

    stationary, message = fit_stationary_gev(data)
    ls_notes.append(message)
    if stationary is None:
        return None, ls_notes

    try:
        if trend_params == 'location':
            x0 = [stationary['location'], 0.0, stationary['scale'], stationary['shape']]
            result = minimize(neg_log_likelihood, x0, method='Nelder-Mead')
            mu0, mu1, sigma, xi = result.x
            params_out = {
                'mu0': mu0, 'mu1': mu1, 'sigma': sigma, 'xi': xi,
                'trend_in': 'location'
            }
            n_params = 4
            
        elif trend_params == 'scale':
            x0 = [stationary['location'], stationary['scale'], 0.0, stationary['shape']]
            result = minimize(neg_log_likelihood, x0, method='Nelder-Mead')
            mu, sigma0, sigma1, xi = result.x
            params_out = {
                'mu': mu, 'sigma0': sigma0, 'sigma1': sigma1, 'xi': xi,
                'trend_in': 'scale'
            }
            n_params = 4

        else:
            x0 = [stationary['location'], 0.0, stationary['scale'], 0.0, stationary['shape']]
            result = minimize(neg_log_likelihood, x0, method='Nelder-Mead')
            mu0, mu1, sigma0, sigma1, xi = result.x
            params_out = {
                'mu0': mu0, 'mu1': mu1, 'sigma0': sigma0, 'sigma1': sigma1, 'xi': xi,
                'trend_in': 'both'
            }
            n_params = 5

        ll = -result.fun
        aic = 2 * n_params - 2 * ll
        bic = log(len(data)) * n_params - 2 * ll
        
        # adding confidence interval to location calculation
        mu_pred, mu_lower, mu_upper = compute_mle_ci([mu0, mu1, sigma, xi], years, data)
        params_out.update({
            'n_obs': len(data),
            'log_likelihood': ll,
            'aic': aic,
            'bic': bic,
            'years_mean': years.mean(),
            'years_std': years.std()
        })
        params_out.update({'CI': {'mu_pred': mu_pred, 'mu_lower': mu_lower, 'mu_upper': mu_upper}})
        
        return params_out, ls_notes
        
    except Exception as e:
        logger.info('Non-stationary GEV fitting error: %s', e)
        return None, ls_notes


def compare_models(stationary: dict, nonstationary: dict) -> dict:
    """
    Compare stationary vs non-stationary GEV using likelihood ratio test.
    
    Returns:
    --------
    dict : Test results and recommendation
    """
    if stationary is None or nonstationary is None:
        return None
    
    # likelihood ratio statistic
    lr_statistic = 2 * (nonstationary['log_likelihood'] - stationary['log_likelihood'])
    
    if nonstationary['trend_in'] in ['location', 'scale']:
        df = 1  
    else:  
        df = 2  
    
    p_value = 1 - stats.chi2.cdf(lr_statistic, df)
    
    delta_aic = nonstationary['aic'] - stationary['aic']
    
    if p_value < 0.05:
        decision = "Non-stationary model is significantly better (p < 0.05)"
        recommendation = "Use non-stationary model - trend detected!"
    elif delta_aic < -2:
        decision = "Non-stationary preferred by AIC (ΔAIC < -2)"
        recommendation = "Use non-stationary model"
    else:
        decision = "No strong evidence for non-stationarity"
        recommendation = "Use stationary model (simpler)"
    
    return {
        'lr_statistic': lr_statistic,
        'df': df,
        'p_value': p_value,
        'delta_aic': delta_aic,
        'delta_bic': nonstationary['bic'] - stationary['bic'],
        'decision': decision,
        'recommendation': recommendation
    }


def calculate_return_levels(
    gev_params: dict, return_periods: list, year: float | None = None,
    cov_matrix: ndarray | None = None,) -> dict:
    """
    Calculate return levels from GEV parameters, optionally with uncertainty.
    """
    if gev_params is None:
        return None
    
    if 'trend_in' in gev_params:
        if year is None:
            year = gev_params['years_mean']
        t = (year - gev_params['years_mean']) / gev_params['years_std']
        
        if gev_params['trend_in'] == 'location':
            mu = gev_params['mu0'] + gev_params['mu1'] * t
            sigma = gev_params['sigma']
            xi = gev_params['xi']
        elif gev_params['trend_in'] == 'scale':
            mu = gev_params['mu']
            sigma = gev_params['sigma0'] + gev_params['sigma1'] * t
            xi = gev_params['xi']
        else:
            mu = gev_params['mu0'] + gev_params['mu1'] * t
            sigma = gev_params['sigma0'] + gev_params['sigma1'] * t
            xi = gev_params['xi']
    else:
        mu = gev_params['location']
        sigma = gev_params['scale']
        xi = gev_params['shape']
    
    results = {}
    z = 1 - 1 / array(return_periods)
    
    for T, p in zip(return_periods, z):
        if abs(xi) < 1e-10:  # Gumbel
            z_T = mu - sigma * log(-log(p))
        else:
            z_T = mu + (sigma / xi) * ((-log(p))**(-xi) - 1)
        
        if cov_matrix is not None:
            if 'trend_in' in gev_params and gev_params['trend_in'] == 'location':
                dz_dmu0 = 1
                dz_dmu1 = t
                dz_dsigma = ((-log(p))**(-xi) - 1)/xi if abs(xi) >= 1e-10 else -log(-log(p))
                dz_dxi = (-sigma/xi**2 * ((-log(p))**(-xi)-1) +
                          (sigma/xi) * ((-log(p))**(-xi) * log(-log(p)))) if abs(xi) >= 1e-10 else 0
                grad = array([dz_dmu0, dz_dmu1, dz_dsigma, dz_dxi])
            else:
                dz_dmu = 1
                dz_dsigma = ((-log(p))**(-xi) - 1)/xi if abs(xi) >= 1e-10 else -log(-log(p))
                dz_dxi = (-sigma/xi**2 * ((-log(p))**(-xi)-1) +
                          (sigma/xi) * ((-log(p))**(-xi) * log(-log(p)))) if abs(xi) >= 1e-10 else 0
                grad = array([dz_dmu, dz_dsigma, dz_dxi])

            var_z = grad.T @ cov_matrix @ grad
            ci_lower = z_T - 1.96 * sqrt(var_z) if var_z >= 0 else z_T
            ci_upper = z_T + 1.96 * sqrt(var_z) if var_z >= 0 else z_T
            results[f'{T}-year'] = {'return_level': z_T, 'CI_lower': ci_lower, 'CI_upper': ci_upper}
        else:
            results[f'{T}-year'] = {'return_level': z_T}
    
    return results


def calculate_return_levels_wo_ci(gev_params: dict, return_periods: list, year: float | None = None) -> dict:
    """
    Calculate return levels from GEV parameters.
    
    Parameters:
    -----------
    gev_params : dict
        GEV parameters (stationary or non-stationary)
    return_periods : list
        Return periods in years
    year : float, optional
        For non-stationary: year to calculate return level
        
    Returns:
    --------
    dict : Return levels
    """
    if gev_params is None:
        return None
    
    if 'trend_in' in gev_params:
        if year is None:
            year = gev_params['years_mean']
        
        t = (year - gev_params['years_mean']) / gev_params['years_std']
        
        if gev_params['trend_in'] == 'location':
            mu = gev_params['mu0'] + gev_params['mu1'] * t
            sigma = gev_params['sigma']
            xi = gev_params['xi']
        elif gev_params['trend_in'] == 'scale':
            mu = gev_params['mu']
            sigma = gev_params['sigma0'] + gev_params['sigma1'] * t
            xi = gev_params['xi']
        else:  # both
            mu = gev_params['mu0'] + gev_params['mu1'] * t
            sigma = gev_params['sigma0'] + gev_params['sigma1'] * t
            xi = gev_params['xi']
    else:
        mu = gev_params['location']
        sigma = gev_params['scale']
        xi = gev_params['shape']
    
    return_levels = {}
    for t_rp in return_periods:
        p = 1 - 1/t_rp
        
        if abs(xi) < 1e-10:  # Gumbel
            z_p = mu - sigma * log(-log(p))
        else:
            z_p = mu + (sigma / xi) * ((-log(p))**(-xi) - 1)
        
        return_levels[f'{t_rp}-year'] = z_p
    
    return return_levels


def compute_cov_matrix(gev_params: dict, data: ndarray, years: ndarray = None) -> ndarray:
    """
    Compute the covariance matrix of fitted GEV parameters using numerical Hessian.

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
        Covariance matrix of parameters
    """
    
    # Define parameter vector theta and negative log-likelihood
    if 'trend_in' in gev_params and gev_params['trend_in'] == 'location':
        # Non-stationary μ(t) = μ0 + μ1 * t
        t = (years - gev_params['years_mean']) / gev_params['years_std']
        def neg_loglik(theta):
            mu0, mu1, sigma, xi = theta
            mu_t = mu0 + mu1 * t
            c = -xi  # SciPy convention
            return -np.sum(stats.genextreme.logpdf(data, c, loc=mu_t, scale=sigma))
        theta_hat = array([
            gev_params['mu0'], gev_params['mu1'], 
            gev_params['sigma'], gev_params['xi']
            ])
    else:
        # Stationary GEV: μ, σ, ξ
        def neg_loglik(theta):
            xi, mu, sigma = theta
            c = -xi
            return -np.sum(stats.genextreme.logpdf(data, c, loc=mu, scale=sigma))
        theta_hat = array([gev_params['shape'], gev_params['location'], gev_params['scale']])
    
    # Numerical Hessian via finite differences
    epsilon = sqrt(finfo(float).eps)
    def grad(theta):
        return optimize.approx_fprime(theta, neg_loglik, epsilon)
    
    hessian_ = optimize.approx_fprime(theta_hat, grad, epsilon)
    
    # Covariance = inverse of Hessian
    try:
        cov_matrix = linalg.inv(hessian_)
    except linalg.LinAlgError:
        logger.info("Warning: Hessian not invertible; returning None")
        return None
    
    return cov_matrix


def compute_cov_matrix_v1(gev_params: dict, data: ndarray, years: ndarray = None) -> ndarray:
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
    is_nonstationary = (
        ('trend_in' in gev_params and gev_params['trend_in'] == 'location') or
        ('trend_params' in gev_params and gev_params['trend_params'] == 'location')
    )
    if is_nonstationary:
        # Non-stationary location μ(t) = μ0 + μ1 * t
        t = (years - gev_params['years_mean']) / gev_params['years_std']

        def neg_loglik(theta):
            mu0, mu1, log_sigma, xi = theta
            sigma = exp(log_sigma)
            mu_t = mu0 + mu1 * t
            c = -xi  # SciPy convention
            z = 1 + xi * (data - mu_t)/sigma

            # Penalize invalid parameters (support violation or sigma <= 0)
            if np.any(z <= 1e-10) or sigma <= 0:
                return 1e10

            ll = stats.genextreme.logpdf(data, c, loc=mu_t, scale=sigma)
            if not np.all(isfinite(ll)):
                return 1e10

            return -np.sum(ll)

        theta_hat = array([
            gev_params.get('mu0', gev_params['params_hat'][0]),
            gev_params.get('mu1', gev_params['params_hat'][1]),
            log(gev_params.get('sigma', gev_params['params_hat'][2])),
            gev_params.get('xi', gev_params['params_hat'][3])
        ])

    else:
        # Stationary GEV: μ, σ, ξ
        def neg_loglik(theta):
            xi, mu, log_sigma = theta
            sigma = exp(log_sigma)
            z = 1 + xi * (data - mu)/sigma
            if np.any(z <= 1e-10) or sigma <= 0:
                return 1e10
            ll = stats.genextreme.logpdf(data, -xi, loc=mu, scale=sigma)
            if not np.all(isfinite(ll)):
                return 1e10
            return -np.sum(ll)

        theta_hat = array([
            gev_params['shape'],
            gev_params['location'],
            log(gev_params['scale'])
        ])

    try:
        if 'trend_in' in gev_params and gev_params['trend_in'] == 'location':
            mu0, mu1, log_sigma, xi = theta_hat
            sigma = exp(log_sigma)
            mu_t = mu0 + mu1 * t
            z = 1 + xi * (data - mu_t)/sigma
        else:
            xi, mu, log_sigma = theta_hat
            sigma = exp(log_sigma)
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
        return None

    return cov_matrix, eigvals, messages


def compute_mle_ci(params, years, data, trend_params='location', alpha=0.05):
    """
    Compute 95% CI for μ(t) from non-stationary GEV via delta method.
    """
    t = (years - years.mean()) / years.std()
    
    # Negative log-likelihood function (reuse your code)
    def neg_log_likelihood_local(p):
        if trend_params == 'location':
            mu0, mu1, sigma, xi = p
            mu_t = mu0 + mu1 * t
            sigma_t = full_like(t, sigma)
        else:
            raise NotImplementedError("CI for scale/both not yet implemented")
        if np.any(sigma_t <= 0):
            return inf
        z = (data - mu_t) / sigma_t
        if abs(xi) < 1e-10:
            ll = -np.sum(log(sigma_t)) - np.sum(z) - np.sum(exp(-z))
        else:
            term = 1 + xi * z
            if np.any(term <= 0):
                return inf
            ll = -np.sum(log(sigma_t)) - (1 + 1/xi) * np.sum(log(term)) - np.sum(term**(-1/xi))
        return -ll

    # ---- Approximate Hessian ----
    epsilon = sqrt(finfo(float).eps)
    grad = lambda p: approx_fprime(p, neg_log_likelihood_local, epsilon)
    p = array(params)
    n = len(p)
    hessian = zeros((n,n))
    for i in range(n):
        def fi(xi): 
            p_tmp = p.copy()
            p_tmp[i] = xi
            return grad(p_tmp)
        hessian[:,i] = approx_fprime([p[i]], fi, epsilon).flatten()
    
    try:
        cov = linalg.inv(hessian)  # covariance of MLE
    except linalg.LinAlgError:
        logger.info("Hessian is singular; cannot compute CI")
        return None, None

    # Delta method: Var(mu_t) = grad(mu_t)^T * cov * grad(mu_t)
    mu0, mu1 = params[0], params[1]
    mu_t = mu0 + mu1 * t
    grad_mu_t = vstack([ones_like(t), t]).T  # derivative wrt mu0, mu1
    var_mu_t = np.sum(grad_mu_t @ cov[:2,:2] * grad_mu_t, axis=1)  # only first 2 params
    std_mu_t = sqrt(var_mu_t)
    
    zscore = norm.ppf(1 - alpha/2)
    upper = mu_t + zscore * std_mu_t
    lower = mu_t - zscore * std_mu_t
    
    return mu_t, lower, upper


def convert_annual_return_levels_with_ci_into_dataframe(data: dict) -> DataFrame:
    data_clean = {}

    for year, year_data in data.items():
        if year_data is None:
            continue
        for period, metrics in year_data.items():
            for metric, value in metrics.items():
                if isinstance(value, float64) and (isnan(value) or value == None):
                    value = nan
                data_clean.setdefault((year, period, metric), value)

    rows = []
    for year, year_data in data.items():
        if year_data is None:
            continue
        row = {'Year': year}
        for period, metrics in year_data.items():
            for metric, value in metrics.items():
                if isinstance(value, float64) and (isnan(value)):
                    value = nan
                row[f'{period}_{metric}'] = float(value)
        rows.append(row)
    
    return (DataFrame(rows)
            .sort_values('Year')
            .reset_index(drop=True)
            .set_index('Year')
            )


def execute_and_store_stat_gev_per_year(
    results: dict, store_results:bool, return_periods:list, column_label_year:str
    ) -> dict:
    dic_notes = {}
    for en, site_id in enumerate(results.keys()):
        ls_notes = []
        grp_per_year = results[site_id]['data'].groupby(column_label_year)
        
        message = f"Conducting stationary GEV for siteID {site_id} (#{en+1} out of {len(results.keys())}) grouped per year..."
        logger.info(message)
        ls_notes.append(message)

        results_stat_per_year_at_location = dict()
        results_return_values_per_year = dict()
        en = 0
        for year, group in grp_per_year:
            en+=1
            logger.info('\t...%s (#%s out of %s years)', int(year), en, len(grp_per_year)) 

            data = (group
                    .sort_values('year')
                    .reset_index(drop=True)
                    .rename(columns={'storm_surge': 'annual_max'})
                    .dropna())
            annual_max_for_year = data['annual_max'].values
            
            gev_stationary, message = fit_stationary_gev(annual_max_for_year, int(year))
            logger.info(message)
            ls_notes.append(message)
            if gev_stationary:
                cov_stationary = compute_cov_matrix(gev_stationary, annual_max_for_year)
                rl_stationary = calculate_return_levels(
                    gev_params=gev_stationary, return_periods=return_periods, cov_matrix=cov_stationary
                )
            else:
                message = f'\tSkipping Return Level Calculation, no stationary GEV for {year}...'
                ls_notes.append(message)
                logger.info(message)
                rl_stationary = None

            results_stat_per_year_at_location[int(year)] = gev_stationary
            results_return_values_per_year[int(year)] = rl_stationary
        logger.info('\nDone!\n') 

        df_stat_gev_per_year = DataFrame.from_dict(results_stat_per_year_at_location).T
        results[site_id]['fit results']['gev_stationary']['analysis_per_year'] = df_stat_gev_per_year.dropna()

        df_stat_return_levels_per_year = convert_annual_return_levels_with_ci_into_dataframe(results_return_values_per_year)
        results[site_id]['fit results']['gev_stationary']['return_levels_per_year'] = df_stat_return_levels_per_year.dropna()
        
        if store_results:
            if 'file_path_report' in results[site_id].keys():
                file_path = results[site_id]['file_path_report']
            else:
                file_path = results[site_id]['file location']
            file_name = os.path.join(file_path, "statGEV_per_year.parquet")
            df_stat_gev_per_year.to_parquet(file_name)
            logger.info('Output stored in %s', file_name)
            
        dic_notes[site_id] = ls_notes       
    return results, dic_notes


def process_site_stat_gev(site_id, site_data, return_periods, store_results:bool = False):
    ls_notes = []
    
    columns = site_data['data']
    year_col = "year" if "year" in columns else "sim_year" if "sim_year" in columns else None
    grp_per_year = site_data['data'].groupby(year_col)
    
    message = f"Conducting stationary GEV for siteID {site_id} grouped per year..."
    logger.info(message)
    ls_notes.append(message)

    results_stat_per_year_at_location = dict()
    results_return_values_per_year = dict()

    for en, (year, group) in enumerate(grp_per_year, start=1):
        logger.info('\tsiteID %s...%s (#%s out of %s years)',site_id, int(year), en, len(grp_per_year))
        data = (group
                .sort_values(year_col)
                .reset_index(drop=True)
                .rename(columns={'storm_surge': 'annual_max'})
                .dropna())
        annual_max_for_year = data['annual_max'].values
        
        gev_stationary, message = fit_stationary_gev(annual_max_for_year, int(year))
        ls_notes.append(message)
        if gev_stationary:
            cov_stationary = compute_cov_matrix(gev_stationary, annual_max_for_year)
            rl_stationary = calculate_return_levels(
                gev_params=gev_stationary, return_periods=return_periods, cov_matrix=cov_stationary
            )
        else:
            message = f'\tskipping Return Level Calculation, no stationary GEV for {year}...'
            ls_notes.append(message)
            logger.info(message)
            rl_stationary = None

        results_stat_per_year_at_location[int(year)] = gev_stationary
        results_return_values_per_year[int(year)] = rl_stationary

    df_stat_gev_per_year = DataFrame.from_dict(results_stat_per_year_at_location).T
    df_stat_return_levels_per_year = convert_annual_return_levels_with_ci_into_dataframe(results_return_values_per_year)

    if store_results:
        file_path = site_data.get('file_path_report') or site_data.get('file location')
        if not file_path:
            fallback_dir = Path(tempfile.gettempdir()) / f"site_{site_id}_output"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            file_path = str(fallback_dir)
            logger.info('[WARNING] No file path found for site %s, using fallback: %s', site_id, file_path)
        
        file_name = os.path.join(file_path, "statGEV_per_year.parquet")
        df_stat_gev_per_year.to_parquet(file_name)
        logger.info('\t→ Output stored as %s\n', file_name)

    site_result = {
        'fit results': {
            'gev_stationary': {
                'analysis_per_year': df_stat_gev_per_year.dropna(),
                'return_levels_per_year': df_stat_return_levels_per_year.dropna()
            }
        }
    }
    return site_id, site_result, ls_notes


def execute_and_store_stat_gev_per_year_mp(results: dict, store_results: bool, return_periods: list) -> dict:
    parallel_results = Parallel(n_jobs=-1)(
        delayed(process_site_stat_gev)(site_id, results[site_id], return_periods, store_results)
        for site_id in tqdm(list(results.keys()))
    )
        
    dic_notes = {}
    for site_id, site_result, ls_notes in parallel_results:
        results[site_id]['fit results'] = site_result['fit results']
        dic_notes[site_id] = ls_notes

    return results, dic_notes


def weighted_least_square_regression_annual_location(global_statgev_scale, global_statgev_shape, df):
    df['n_obs'] = to_numeric(df.n_obs, errors='coerce') 
    df = df.dropna(subset=['n_obs'])  
    df['n_obs'] = df.n_obs.astype(int)

    df['var_mu'] = (global_statgev_scale ** 2) / df.n_obs * 1 / (1 - global_statgev_shape) ** 2
    weights = 1.0 / df.var_mu.values

    year_mean = df['year'].mean()
    X = sm.add_constant(df['year'] - year_mean)  # centered
    y = df.location.astype(float)

    wls_delta = sm.WLS(y, X, weights=weights).fit()
    
    
    year_grid = linspace(df['year'].min(), df['year'].max(), 200)
    X_pred = sm.add_constant(year_grid - year_mean)  
    y_pred = wls_delta.predict(X_pred)

    return df, wls_delta, weights, y_pred, year_grid, year_mean


def weighted_least_square_regression_for_site_mp(site_id, dic_location):
    """Perform the WLS regression for one site, return the results needed for later plotting/saving."""
    df_stat_gev_per_year = dic_location['fit results']['gev_stationary']['analysis_per_year'].dropna()
    df = df_stat_gev_per_year.reset_index().rename(columns={'index': 'year'})

    global_statgev_shape = dic_location['fit results']['gev_stationary']['shape']
    global_statgev_scale = dic_location['fit results']['gev_stationary']['scale']

    df, wls_delta, weights, y_pred, year_grid, year_mean = weighted_least_square_regression_annual_location(
        global_statgev_scale, global_statgev_shape, df
    )

    return site_id, {
        'df': df,
        'wls_delta': wls_delta,
        'weights': weights,
        'y_pred': y_pred,
        'year_grid': year_grid,
        'year_mean': year_mean
    }


def analyze_per_location(
    data_hindcast:DataFrame, site_id: int, lat: float, lon: float, location_info: str, return_periods: list
    )-> tuple[dict, list]:
    """
    Complete analysis for one model-location combination.
    Fits both stationary and non-stationary GEV.
    """
    ls_notes = []
    annual_max = extract_annual_maxima_at_location(data_hindcast, lon=lon, lat=lat)
    
    if len(annual_max) < 10:
        return None

    years = annual_max['year'].values
    data = annual_max['annual_max'].values

    logger.info("\tConducting stationary GEV...")
    gev_stationary, warnings_ = fit_stationary_gev(data)
    ls_notes.append(warnings_)
    logger.info('\t → Stationary GEV done (success %s)',gev_stationary != None)
    
    logger.info("\tContinuing with non-stationary GEV...")
    gev_nonstat_loc, warnings_ = fit_nonstationary_gev(years, data, 'location')
    ls_notes.append(warnings_)
    logger.info('\t → Non-stationary GEV done (success %s)', gev_nonstat_loc != None)
    
    comparison = compare_models(gev_stationary, gev_nonstat_loc)
    
    logger.info('Return Levels Stationary GEV...')
    cov_stationary = compute_cov_matrix(gev_stationary, data)
    rl_stationary = calculate_return_levels(
        gev_params=gev_stationary, return_periods=return_periods, cov_matrix=cov_stationary
        )

    logger.info('Return Levels NON-Stationary GEV...')
    if gev_nonstat_loc:
        cov_nonstat = compute_cov_matrix(gev_nonstat_loc, data, years=years)
        rl_nonstat_start = calculate_return_levels(
            gev_nonstat_loc, return_periods, year=years.min(), cov_matrix=cov_nonstat
            )
        rl_nonstat_end = calculate_return_levels(
            gev_nonstat_loc, return_periods, year=years.max(), cov_matrix=cov_nonstat
            )
    else:
        rl_nonstat_start = None
        rl_nonstat_end = None
    
    return {'location info': {'site_id': site_id, 'lat': lat, 'lon': lon, 'description': location_info},
            'hindcast period': (int(annual_max.year.min()), int(annual_max.year.max())), 
            'data': annual_max,
            'fit results': {
                'gev_stationary': gev_stationary, 
                'gev_nonstationary': gev_nonstat_loc, 
                },
            'model_comparison': comparison,
            'return_levels': {
                'stationary': rl_stationary,
                'nonstationary_start': {'year': int(years.min()), 'values': rl_nonstat_start},
                'nonstationary_end': {'year': int(years.max()),'values': rl_nonstat_end},
                }
    }, ls_notes


def process_location(site_id, dic_location, display_results, save_regression_summary):
    df_stat_gev_per_year = dic_location['fit results']['gev_stationary']['analysis_per_year'].dropna()
    df = df_stat_gev_per_year.reset_index().rename(columns={'index': 'year'})

    global_statgev_shape = dic_location['fit results']['gev_stationary']['shape']
    global_statgev_scale = dic_location['fit results']['gev_stationary']['scale']

    df, wls_delta, weights, y_pred, year_grid, year_mean = weighted_least_square_regression_annual_location(
        global_statgev_scale, global_statgev_shape, df
    )
    dic_location['WLSdelta'] = dict({'summary': wls_delta, 'weights': weights})

    fig = dbplt.plot_gev_mu_trend(
        df=df,
        weights=weights,
        year_grid=year_grid,
        year_mean=year_mean,
        y_pred=y_pred,
        wls_delta=wls_delta,
        nonstat_years=dic_location['data'].year.values.astype(int),
        nonstat=dic_location['fit results']['gev_nonstationary'],
        display_results=display_results,
        colors_reg=['#333333FF', '#C88D35FF'],
        markers_color='#99E3DDFF'
    )

    if save_regression_summary and ('file location' in dic_location or 'file_path_report' in dic_location):
        save_path = dic_location.get('file location', dic_location.get('file_path_report'))
        with open(save_path + '/WLSdelta_summary.html', 'w') as f:
            f.write(wls_delta.summary().as_html())
        
        lat = str(dic_location['location info']['lat'].round(3))
        lon = str(dic_location['location info']['lon'].round(3))
        country = dic_location['location info']['description'].split(',')[-1].strip()  
        file_name = f"/GEVTrendAnalysis_location_{site_id}_{country}_{lat}|{lon}.png"
        fig.savefig(save_path + file_name, dpi=300, bbox_inches='tight')

    return site_id, dic_location


def create_gev_written_report_per_location(
    result_location: dict, location_in_example: tuple[str, float, float], ls_messages:list[str], print_msg:bool=True
    ) -> list[str]:
    
    ls_messages = ut.adding_plot_and_text("\n" + "="*100, ls_messages, print_msg)
    ls_messages = ut.adding_plot_and_text(
        "POOLED RESULTS for location (lon|lat): "
        f"{result_location['location info']['lon']:.5f}|{result_location['location info']['lat']:.5f}", 
        ls_messages, print_msg
        )
    ls_messages = ut.adding_plot_and_text("="*100, ls_messages, print_msg)
    
    ls_messages = ut.adding_plot_and_text(
        f"\nClosest location identified: {location_in_example}", ls_messages, print_msg
        )
    
    # -------------------------------------------------------------------------
    ls_messages = ut.adding_plot_and_text(
        f"Hindcast period: {result_location['hindcast period'][0]}-{result_location['hindcast period'][1]} "
        f"({len(result_location['data'].year.unique())} unique years)", ls_messages, print_msg
        )
    ls_messages = ut.adding_plot_and_text(
        "Observations per year: ~2 (from ensemble members)", ls_messages, print_msg
        )
    ls_messages = ut.adding_plot_and_text(
        f"Total data points for GEV: {result_location['fit results']['gev_stationary']['n_obs']}", ls_messages, 
        print_msg
        )
    
    # -------------------------------------------------------------------------
    ls_messages = ut.adding_plot_and_text("\nSTATIONARY GEV", ls_messages, print_msg)
    stat = result_location['fit results']['gev_stationary']
    ls_messages = ut.adding_plot_and_text(f"  μ (location) = {stat['location']:.3f}", ls_messages, print_msg)
    ls_messages = ut.adding_plot_and_text(f"  σ (scale) = {stat['scale']:.3f}", ls_messages, print_msg)
    ls_messages = ut.adding_plot_and_text(f"  ξ (shape) = {stat['shape']:.3f}", ls_messages, print_msg)
    ls_messages = ut.adding_plot_and_text(f"  Type: {stat['dist_type']}", ls_messages, print_msg)
    ls_messages = ut.adding_plot_and_text("  Return Level", ls_messages, print_msg)
    
    for x, v in result_location['return_levels']['stationary'].items():
        aep = 100/int(x.split('-')[0])
        ls_messages = ut.adding_plot_and_text(
            f"  \tFor {x} period: {v:.3f} m ({aep}% annual exceedance probability)", ls_messages, print_msg
            )
        
    # -------------------------------------------------------------------------
    if result_location['fit results']['gev_nonstationary']:
        ls_messages = ut.adding_plot_and_text("\nNON-STATIONARY GEV", ls_messages, print_msg)
        nonstat = result_location['fit results']['gev_nonstationary']
        ls_messages = ut.adding_plot_and_text(
            f"  μ(t) = {nonstat['mu0']:.3f} + {nonstat['mu1']:.4f}·t", ls_messages, print_msg
            )
        ls_messages = ut.adding_plot_and_text(f"  Trend = {nonstat['mu1']:.4f} m/year", ls_messages, print_msg)
        ls_messages = ut.adding_plot_and_text("  Return Level Evolution", ls_messages, print_msg)
        for x,y in zip(
            result_location['return_levels']['nonstationary_start']['values'].items(), 
            result_location['return_levels']['nonstationary_end']['values'].items()
            ):

            period = x[0]
            aep = 100/int(period.split('-')[0])
            ls_messages = ut.adding_plot_and_text(
                f"  \tFor {period} period: {x[1]:.3f}m - {y[1]:.3f}m ({aep}% annual exceedance probability)",
                ls_messages, print_msg
            )
    # -------------------------------------------------------------------------       
    if result_location['model_comparison']:
        ls_messages = ut.adding_plot_and_text("\nMODEL COMPARISON", ls_messages, print_msg)
        comp = result_location['model_comparison']
        ls_messages = ut.adding_plot_and_text(f"  p-value: {comp['p_value']:.4f}", ls_messages, print_msg)
        ls_messages = ut.adding_plot_and_text(f"  Decision: {comp['decision']}", ls_messages, print_msg)
        ls_messages = ut.adding_plot_and_text(f"  → {comp['recommendation']}", ls_messages, print_msg)

    return ls_messages


def create_gev_report_per_location(
    result_location: dict, plot_period_evolution:list[str], export_report:bool=True, path_export:str | None=None, 
    print_msg:bool | None=False, display_results:bool | None=True
    ) -> None:
    ls_messages = []
    
    if result_location:
        location_info = result_location['location info']
        location_label = location_info['description']
        
        display(Markdown(f"""<pre><strong>    Create GEV analysis report for {location_label}...</strong></pre>"""))
                
        today_ = str(datetime.today().date().isoformat())
        country = location_label.split(',')[-1].strip()
        lat_str = str(round(float(result_location['location'][0]), 3))
        lon_str = str(round(float(result_location['location'][1]),3))
        
        ls_messages = create_gev_written_report_per_location(
            result_location=result_location, location_in_example=location_label, ls_messages=ls_messages,
            print_msg=print_msg
            )
        
        if export_report and path_export:
            save_path = path_export + today_
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True) 

            full_file_path = save_dir / f"GEVanalysis_{country}_{lat_str}|{lon_str}_{today_}.parquet"   
            
            df_messages = DataFrame({'messages': ls_messages}) 
            df_messages.to_parquet(full_file_path, index=False)
        else:
            save_path = None
            
        # -----------------------------------------------------------------------------------------------------------
        _ = dbplt.plot_pooled_analysis(
            result=result_location, 
            site_id=result_location['location'], 
            location_info=result_location['location info'][0],
            periods_evolution = plot_period_evolution,
            box_parameters_x=0.05, box_parameters_y=0.95,
            width_bar_returns=0.35,
            leg_comparison_x=0.35, leg_comparison_y=0.65, linespace=1.5,
            save_path = save_path,
            color_markers='#99E3DDFF', colors_trends='#1D141BFF', 
            colors_models=['#B887ADFF', '#008A80FF'],
            colors_return_levels=['#008A80FF','#CAA5C2FF'],
            bbox_color='#F5F5F5FF', axes_color='#333333', 
            linestyle_trends=['dashdot', 'dashed', 'solid'], 
            fontsize=12, figsize=(15, 7.5),
            display_results=display_results
        )
        display(Markdown(f"""<pre><strong>    Report Completed.</strong></pre>"""))
        return full_file_path


def store_report_stationary_gev_per_year(site_data:dict) -> None:
    if 'location info' in site_data:
        site_data['location info'] = str(site_data['location info'])

    annual_max_path = Path(site_data['file_path_report']).with_name(Path(site_data['file_path_report']).stem + "_data.parquet")
    site_data['annual_maxima'].to_parquet(annual_max_path, index=False)
    site_data['file_path_annual_max'] = str(annual_max_path)

    analysis_clean = {}
    for year, d in site_data['gev_stationary']['analysis_per_year'].items():
        clean_d = {k: (v.tolist() if isinstance(v, (ndarray, generic)) else v) for k, v in d.items()}
        analysis_clean[year] = clean_d

    analysis_path = Path(site_data['file_path_report']).with_name(Path(site_data['file_path_report']).stem + "_stat-gev_per_year.parquet")
    DataFrame.from_dict(analysis_clean, orient='index').to_parquet(analysis_path, index=True)
    site_data['file_path_analysis_per_year'] = str(analysis_path)

    metadata = {k: v for k, v in site_data.items() if k not in ['annual_maxima', 'gev_stationary']}
    metadata['gev_stationary'] = {k: v for k, v in site_data['gev_stationary'].items() if k != 'analysis_per_year'}

    for k, v in metadata.items():
        if not isinstance(v, (str, int, float, bool, dict, list)):
            metadata[k] = str(v)

    meta_path = Path(site_data['file_path_report']).with_name(Path(site_data['file_path_report']).stem + "_metadata.parquet")
    DataFrame({'metadata': [metadata]}).to_parquet(meta_path, index=False)
    site_data['file_path_metadata'] = str(meta_path)
    site_data['file_path_metadata'] = str(meta_path)
    site_data['file_path_metadata'] = str(meta_path)
    site_data['file_path_metadata'] = str(meta_path)
    site_data['file_path_metadata'] = str(meta_path)


# ----------------------------------------------
def set_location_labels(labels):
    global _LOCATION_LABELS
    _LOCATION_LABELS = labels
    

def mle_fitting(data, n, ls_notes):
    c, location, scale = stats.genextreme.fit(data)
    shape = -c

    logpdf = stats.genextreme.logpdf(data, c, location, scale)
    if not np.all(isfinite(logpdf)):
        ls_notes.append("Non-finite log-likelihood (GEV support violation)")
    ll = np.sum(logpdf)

    n_params = 3
    aic = 2 * n_params - 2 * ll
    bic = log(n) * n_params - 2 * ll

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


def stationary_gev_bootstrapping(data, B=500, seed=None):
    """
    Estimate uncertainty of GEV location parameter using parametric bootstrap.
    
    Parameters
    ----------
    data : array
        Annual maxima
    B : int
        Number of bootstrap simulations
    seed : int or None
    
    Returns
    -------
    dict with:
        mu_hat
        mu_std
        mu_samples
    """
    if seed is not None:
        random.seed(seed)
    
    n = len(data)
    c_hat, mu_hat, sigma_hat = genextreme.fit(data)
    
    shape_samples = zeros(B)
    mu_samples = zeros(B)
    scale_samples = zeros(B)

    for b in range(B):
        synthetic = genextreme.rvs(
            c=c_hat,
            loc=mu_hat,
            scale=sigma_hat,
            size=n
        )
        
        c_b, mu_b, sigma_b = genextreme.fit(synthetic)  
        shape_samples[b] = -c_b
        mu_samples[b] = mu_b
        scale_samples[b] = sigma_b
        
    
    return {
        "mu_hat": mu_hat,
        "mu_std": std(mu_samples, ddof=1),
        "mu_samples": mu_samples,
        "shape_hat": -c_hat,
        "shape_std": std(shape_samples, ddof=1),
        "scale_hat": sigma_hat,
        "scale_std": std(scale_samples, ddof=1),
        "n_obs": n
    }


def nonstationary_bootstrapping(neg_loglik, params_hat, t, n, B=500, trend_params: str = 'location'):
    mu0_samples, mu1_samples, sigma_samples, xi_samples = [], [], [], []
    for b in range(B):
        if trend_params == 'location':
            mu0, mu1, sigma, xi = params_hat
            mu_t = mu0 + mu1 * t
            synthetic = genextreme.rvs(c=-xi, loc=mu_t, scale=sigma, size=n)
            res_b = minimize(neg_loglik, params_hat, method='Nelder-Mead')
            mu0_b, mu1_b, sigma_b, xi_b = res_b.x
        elif trend_params == 'scale':
            mu, sigma0, sigma1, xi = params_hat
            sigma_t = sigma0 + sigma1 * t
            synthetic = genextreme.rvs(c=-xi, loc=mu, scale=sigma_t, size=n)
            res_b = minimize(neg_loglik, params_hat, method='Nelder-Mead')
            mu, sigma0_b, sigma1_b, xi_b = res_b.x
        else:  # both
            mu0, mu1, sigma0, sigma1, xi = params_hat
            mu_t = mu0 + mu1 * t
            sigma_t = sigma0 + sigma1 * t
            synthetic = genextreme.rvs(c=-xi, loc=mu_t, scale=sigma_t, size=n)
            res_b = minimize(neg_loglik, params_hat, method='Nelder-Mead')
            mu0_b, mu1_b, sigma0_b, sigma1_b, xi_b = res_b.x
        
        mu0_samples.append(mu0_b if trend_params != 'scale' else mu)
        mu1_samples.append(mu1_b if trend_params != 'scale' else 0.0)
        sigma_samples.append(sigma_b if trend_params != 'scale' else (sigma0_b if trend_params=='scale' else sigma0_b))
        xi_samples.append(xi_b)
    return mu0_samples, mu1_samples, sigma_samples, xi_samples


def _gev_negloglik(params, data):
    c, loc, scale = params
    if scale <= 0:
        return inf
    return -np.sum(stats.genextreme.logpdf(data, c, loc, scale))



def _compute_fisher_info(data, c, loc, scale):
    ls_notes = []
    params = array([c, loc, scale])
    eps = sqrt(finfo(float).eps)

    n = len(params)
    hessian = zeros((n, n))

    for i in range(n):
        for j in range(n):
            shift = zeros(n)
            shift[i] += eps
            shift[j] += eps

            fpp = _gev_negloglik(params + shift, data)
            fpm = _gev_negloglik(params + shift * array([1, -1, 1]), data)
            fmp = _gev_negloglik(params + shift * array([-1, 1, 1]), data)
            fmm = _gev_negloglik(params - shift, data)

            values = [fpp, fpm, fmp, fmm]

            if not np.all(isfinite(values)):
                ls_notes.append(
                    f"Invalid likelihood values at params={params}, "
                    f"shifted values={values}"
                )

            hessian[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps**2)

    if not np.all(isfinite(hessian)):
        ls_notes.append(f"Hessian contains invalid values:\n{hessian}")

    try:
        cov = linalg.inv(hessian)
    except:
        cov = None
    return cov, ls_notes


def _compute_fisher_info_generic(neg_loglik, params_hat):
    """
    Numerical Fisher Information via Hessian of negative log-likelihood.
    Works for arbitrary number of parameters.
    """
    params_hat = array(params_hat, dtype=float)
    n = len(params_hat)
    eps = sqrt(finfo(float).eps)

    hessian = zeros((n, n))

    for i in range(n):
        for j in range(n):
            shift = zeros(n)
            shift[i] += eps
            shift[j] += eps

            fpp = neg_loglik(params_hat + shift)
            fpm = neg_loglik(params_hat + shift * array([1 if k != j else -1 for k in range(n)]))
            fmp = neg_loglik(params_hat + shift * array([1 if k != i else -1 for k in range(n)]))
            fmm = neg_loglik(params_hat - shift)

            hessian[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps**2)

    try:
        cov = linalg.inv(hessian)
    except:
        return None

    return cov


def fit_stationary_gev_incl_uncertainty(
    data: ndarray,
    year: int | None = None,
    print_msg: bool = False,
    uncertainty: str | None = None,
    B: int = 500,
    seed: int | None = None,
) -> tuple[dict | None, list[str]]:
    """
    Fit stationary GEV using MLE.
    Optionally estimate parameter uncertainty via parametric bootstrap or fisher.
    """
    if uncertainty is None:
        uncertainty = "fisher"
        
    ls_notes = []
    n = len(data)

    if n < 10:
        message = (
            f"Warning: Only {n} observations"
            + (f" in {year}" if year else "")
            + ". Need at least 10 for reliable GEV fit."
        )
        if print_msg:
            print(message)
        ls_notes.append(message)
        return None, ls_notes

    try:
        # ---- MLE FIT ----
        if print_msg:
            print('Compute maximum likelihood estimation')
        result, ls_notes = mle_fitting(data, n, ls_notes)

        # ---- OPTIONAL Fisher Information ---- 
        if uncertainty == "fisher" :
            message = 'Compute Fisher Information...'
            if print_msg:
                print(message)
            ls_notes.append(message)
            
            cov, messages = _compute_fisher_info(data, -result['shape'], result['location'], result['scale'])
            ls_notes.append(messages)
            message = f'\t\tHessian covariance computed: {cov}'
            if print_msg:
                print(message)
            ls_notes.append(message)
            
            if cov is not None and np.all(isfinite(cov)):
                result.update({"cov": cov})
                
                std_errors = sqrt(diag(cov))
                result.update({
                    "shape_std": std_errors[0],
                    "location_std": std_errors[1],
                    "scale_std": std_errors[2],
                })
            else:
                message = f'Failed to compute information, falling back to bootstrap (B={B})...'
                if print_msg:
                    print(message)
                ls_notes.append(message)
        
                uncertainty = "bootstrap"
                B = 150
                ls_notes.append('Compute Bootstrapping...')
                bootstrap_results = stationary_gev_bootstrapping(data, B=B, seed=seed)
                
                result.update({
                    "shape_std": bootstrap_results['shape_std'],
                    "location_std": bootstrap_results['mu_std'],
                    "scale_std": bootstrap_results['scale_std'],
                    "location_samples": bootstrap_results['mu_samples'],
                    "n_obs": bootstrap_results['n_obs']
                })
                
        # ---- OPTIONAL BOOTSTRAP (more accurate) ----
        if uncertainty == "bootstrap":
            ls_notes.append('Compute Bootstrapping...')
            bootstrap_results = stationary_gev_bootstrapping(data, B=B, seed=seed)
            
            result.update({
                "shape_std": bootstrap_results['shape_std'],
                "location_std": bootstrap_results['mu_std']*1000, # conversion to millimeter
                "scale_std": bootstrap_results['scale_std'],
                "location_samples": bootstrap_results['mu_samples'],
                "n_obs": bootstrap_results['n_obs']
            })
        result.update({'uncertainty': uncertainty})
        
        return result, ls_notes

    except Exception as e:
        message = f"Failed to conduct GEV fitting due to error: {e}"
        if print_msg:
            print(message)
        ls_notes.append(message)
        
        return None, ls_notes


def fit_pooled_gev_with_uncertainty(
    loc_id: int,
    years: ndarray,
    data: ndarray,
    trend_params: str = 'location',
    uncertainty: str | None = None,
    B: int = 500,
    seed: int | None = None,
    print_msg: bool = False
) -> tuple[dict | None, list[str]]:
    """
    Fit (non-)stationary GEV.
    Optionally compute parameter uncertainty via Fisher or bootstrap.
    """
    if uncertainty is None:
        uncertainty = "delta"
    
    n = len(data)
    
    # ---- STEP 1: Get stationary ----
    message = f'\t\tLoc {loc_id} | Compute stationary GEV incl uncertainty using {uncertainty}'
    logger.info(message)
    if print_msg:
        print(message)
    stationary, stat_notes = fit_stationary_gev_incl_uncertainty(data)
    logger.info(stat_notes)
    
    if stationary is None:
        message = f"Loc {loc_id} | Failed to compute stationary GEV, skipping..."
        if print_msg:
            print(message)
        logger.info(message)
        return {'stationary': None, 'nonstationary': None, 'notes': ['Stationary GEV failed']}
    
    # ---- STEP 2: Prepare for non-stationary GEV ----
    message = f'\t\tLoc {loc_id} | Compute non-stationary GEV incl uncertainty using {uncertainty}'
    logger.info(message)
    if print_msg:
        print(message)
    
    if n < 20:
        message = f"Loc {loc_id} | Warning: Non-stationary GEV needs ≥20 observations but has {n}, "
        message += f"skipping non-stationary GEV..."
        if print_msg:
            print(message)
        logger.info(message)
        return stationary
    
    t = (years - years.mean()) / years.std()
    
    if trend_params == 'location':
        x0 = [stationary['location'], 0.0, stationary['scale'], stationary['shape']]
    elif trend_params == 'scale':
        x0 = [stationary['location'], stationary['scale'], 0.0, stationary['shape']]
    else:
        x0 = [stationary['location'], 0.0, stationary['scale'], 0.0, stationary['shape']]
    
    # ---- STEP 3: Define negative log-likelihood for non-stationary case ----
    def neg_loglik(params):
        if trend_params == 'location':
            mu0, mu1, sigma, xi = params
            mu_t = mu0 + mu1 * t
            sigma_t = full_like(t, sigma)
        elif trend_params == 'scale':
            mu, sigma0, sigma1, xi = params
            mu_t = full_like(t, mu)
            sigma_t = sigma0 + sigma1 * t
        elif trend_params == 'both':
            mu0, mu1, sigma0, sigma1, xi = params
            mu_t = mu0 + mu1 * t
            sigma_t = sigma0 + sigma1 * t
        else:
            raise ValueError("trend_params must be 'location', 'scale', or 'both'")
        
        if np.any(sigma_t <= 0):
            return inf
        
        z = (data - mu_t) / sigma_t
        if abs(xi) < 1e-10:  # Gumbel case
            ll = -np.sum(log(sigma_t)) - np.sum(z) - np.sum(exp(-z))
        else:
            term = 1 + xi * z
            if np.any(term <= 0):
                return inf
            ll = -np.sum(log(sigma_t)) - np.sum((1 + 1/xi) * log(term)) - np.sum(term**(-1/xi))
        
        return -ll
        
    # ---- STEP 4: Fit MLE for non-stationary case ----
    try:
        result = minimize(neg_loglik, x0, method='Nelder-Mead')
        params_hat = result.x
    except Exception as e:
        message = f"Loc {loc_id} | Non-stationary MLE fitting failed: {e}"
        if print_msg:
            print(message)
        logger.info(message)
        return None
    
    message = f'\t\t\tLoc {loc_id} | Nonstationary GEV - MLE {params_hat}'
    logger.info(message)
    if print_msg:
        print(message)

    nonstationary = {
        'params_hat': params_hat, 'trend_params': trend_params, 'n_obs': n, 
        'years_mean': years.mean(), 'years_std': years.std()
        }
    
    # ---- STEP 5: Uncertainty Calculation for non-stationary case DEFAULT Fisher ----
    if uncertainty == "delta":
        message = "\t\t\tLoc {loc_id} | Computing delta method for non-stationary GEV..."
        logger.info(message)
        if print_msg:
            print(message)

        try:
            cov_ns, hessian_eigenvalues, messages = compute_cov_matrix_v1(gev_params=nonstationary, data=data, years=years)
            nonstationary['Hessian eigenvalues'] = hessian_eigenvalues
            logger.info(messages)
        except:
            cov_ns = None
            logger.info("Loc {loc_id} | Hessian inversion failed; Delta-method SD not computed.")

        if cov_ns is not None:
            nonstationary['cov_mu'] = cov_ns
            try:
                diag_vals = diag(cov_ns)
                if np.any(diag_vals < 0):
                    logger.debug(f"Loc {loc_id} | Negative variance in cov diagonal: {diag_vals}")
                    raise ValueError(f"Loc {loc_id} | Negative variance in cov diagonal: {diag_vals}")
                params_std = sqrt(diag_vals)
                if np.any(isnan(params_std)):
                    logger.debug("Loc {loc_id} | NaN in params_std after sqrt")
                    raise ValueError("Loc {loc_id} | NaN in params_std after sqrt")
            except Exception as e:
                params_std = None
                logger.debug(f"Loc {loc_id} | params_std failed: {e}")
                logger.info(f"Loc {loc_id} | params_std failed: {e}")
            nonstationary['params_std'] = params_std
        else:
            logger.info("Loc {loc_id} | Delta Method failed, falling back to Fisher Information")
            uncertainty = "fisher"
    
    if uncertainty == "fisher":
        logger.info("Loc {loc_id} | Computing Fisher Information for non-stationary GEV...")
        try:
            cov = _compute_fisher_info_generic(neg_loglik, params_hat)
            if cov is not None and np.all(isfinite(cov)):
                std_errors = sqrt(diag(cov))
                if np.all(isfinite(std_errors)):
                    nonstationary['params_std'] = std_errors
                else:
                    logger.info("Loc {loc_id} | Fisher Information failed (not all values computed), falling back to bootstrap")
                    uncertainty = "bootstrap"
                    B = 300
            else:
                logger.info("Loc {loc_id} | Fisher Information failed, falling back to bootstrap")
                uncertainty = "bootstrap"
                B = 150
        except Exception as e:
            logger.info(f"Loc {loc_id} | Fisher Information exception: {e}, falling back to bootstrap")
            uncertainty = "bootstrap"
            B = 300
    
    if uncertainty == "bootstrap":
        logger.infp("Loc {loc_id} | Computing bootstrap uncertainty for non-stationary GEV...")
        if seed is not None:
            random.seed(seed)
        
        mu0, mu1, _, _ = params_hat
        t_std = (years - years.mean()) / years.std()
        mu_t = mu0 + mu1 * t_std

        residuals = data - mu_t

        mu0_samples, mu1_samples, sigma_samples, xi_samples = [], [], [], []

        for _ in range(B):
            res_b = minimize(neg_loglik, params_hat, method='Nelder-Mead')
            p_b = res_b.x
            mu0_samples.append(p_b[0])
            mu1_samples.append(p_b[1])
            sigma_samples.append(p_b[2])
            xi_samples.append(p_b[3])

        nonstationary.update({
            'mu0_samples': array(mu0_samples),
            'mu1_samples': array(mu1_samples),
            'sigma_samples': array(sigma_samples),
            'xi_samples': array(xi_samples),
            'mu0_std': std(mu0_samples, ddof=1),
            'mu1_std': std(mu1_samples, ddof=1),
            'sigma_std': std(sigma_samples, ddof=1),
            'xi_std': std(xi_samples, ddof=1),
        })
    
    return {'stationary': stationary, 'nonstationary': nonstationary}


def likelihood_ratio_test(LL_s, LL_ns, df):
    delta_LL = 2 * (LL_ns - LL_s)
    if delta_LL < 0:
        return delta_LL, 1.0, '→ stationary model is sufficient, trend in μ is not significant'
    else:
        p = 1 - chi2.cdf(delta_LL, df)
        if p< 0.05:
            return delta_LL, p, '→ non-stationary model is significantly better → μ(t) trend matters'
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
        LL_s = -np.sum(log(sigma_s)) - np.sum(z_s) - np.sum(exp(-z_s))
    else:
        term = 1 + xi_s*z_s
        LL_s = -n * log(sigma_s) - np.sum((1 + 1/xi_s) * log(term)) - np.sum(term**(-1/xi_s))
    k_s = 3  # stationary: μ, σ, ξ
    AIC_s = 2*k_s - 2*LL_s
    BIC_s = k_s*log(n) - 2*LL_s

    # Non-stationary (location trend)
    mu0 = nonstationary['params_hat'][0]
    mu1 = nonstationary['params_hat'][1]
    sigma_ns = nonstationary['params_hat'][2] 
    xi_ns = nonstationary['params_hat'][3] 

    t_scaled = (data_loc.year - nonstationary['years_mean']) / nonstationary['years_std']  # scaled years

    mu_t = mu0 + mu1*t_scaled
    z_ns = (data_loc.annual_max - mu_t)/sigma_ns
    if abs(xi_ns) < 1e-10:
        LL_ns = -np.sum(log(sigma_ns)) - np.sum(z_ns) - np.sum(exp(-z_ns))
    else:
        term = 1 + xi_ns*z_ns
        LL_ns = -n * log(sigma_ns) - np.sum((1 + 1/xi_ns) * log(term)) - np.sum(term**(-1/xi_ns))

    k_ns = 4  # μ0, μ1, σ, ξ
    AIC_ns = 2*k_ns - 2*LL_ns
    BIC_ns = k_ns*log(n) - 2*LL_ns

    # Likelihood ratio test
    df = k_ns - k_s
    delta_LL, p_value, interpretation = likelihood_ratio_test(LL_s=LL_s, LL_ns=LL_ns, df=df)

    return dict({
            'stationary': {'LL': LL_s, 'k': k_s, 'AIC': AIC_s, 'BIC': BIC_s},
            'nonstationary': {'LL': LL_ns, 'k': k_ns, 'AIC': AIC_ns, 'BIC': BIC_ns},
            'LRT': {'delta_LL': delta_LL, 'df': df, 'p_value': p_value, 'interpretation': interpretation}
        })


def compute_return_levels_for_year(stationary, nonstationary, T, t_eval):
    t_eval = asarray(t_eval)
    
    #  stationary
    mu = stationary['location']
    sd_mu = stationary['location_std']
    scale = stationary['scale']
    shape = stationary['shape']
    
    factor = (-log(1 - 1/T))**(-shape) - 1
    z_T = mu + scale/shape * factor
    z_lower = z_T - 1.96 * sd_mu
    z_upper = z_T + 1.96 * sd_mu

    # non-stationary
    params_hat = nonstationary['params_hat']
    years_mean = nonstationary['years_mean']
    years_std = nonstationary['years_std']
    
    mu0, mu1, sigma, xi = params_hat
    t_scaled_eval = (t_eval - years_mean) / years_std
    
    #mu_t = nonstationary['mu0_samples'].mean() + nonstationary['mu1_samples'].mean()*t_scaled_eval
    mu_t = mu0 + mu1*t_scaled_eval
    #mu_samples_eval = nonstationary['mu0_samples'][:, None] + nonstationary['mu1_samples'][:, None]*t_scaled_eval
    
    if 'delta_sd_mu' in nonstationary:
        sd_mu_t = array([nonstationary['delta_sd_mu'](year) for year in t_eval])
    else:
        sd_mu_t = zeros_like(mu_t)  # fallback if Delta-method not available

    if abs(xi) < 1e-10:
        factor_ns = -log(-log(1 - 1/T))
        z_T_ns = mu_t + sigma * factor_ns
    else:
        factor_ns = (-log(1 - 1/T))**(-xi) - 1
        z_T_ns = mu_t + sigma/xi * factor_ns

    z_lower_ns = z_T_ns - 1.96 * sd_mu_t
    z_upper_ns = z_T_ns + 1.96 * sd_mu_t
    
    return dict({
        'stationary': {'z_T': z_T, 'lower': z_lower, 'upper': z_upper},
        'nonstationary': {'z_T': z_T_ns, 'lower': z_lower_ns, 'upper': z_upper_ns},
            })


def convert_return_level_format(return_periods, t_eval, all_return_levels):
    dfs = []

    for T in return_periods:
        n_eval = len(t_eval)

        # stationary 
        stat = all_return_levels[T]['stationary']
        df_stat = DataFrame({
            'z_T': repeat(stat['z_T'], n_eval),
            'lower': repeat(stat['lower'], n_eval),
            'upper': repeat(stat['upper'], n_eval),
            'model': 'stationary',
            't_eval': t_eval,
            'return_period': T
        })

        # nonstationary 
        ns = all_return_levels[T]['nonstationary']
        df_ns = DataFrame({
            'z_T': ns['z_T'],
            'lower': ns['lower'],
            'upper': ns['upper'],
            'model': 'nonstationary',
            't_eval': t_eval,
            'return_period': T
        })

        dfs.extend([df_stat, df_ns])

    result = concat(dfs, ignore_index=True)
    result.set_index(['t_eval', 'return_period', 'model'], inplace=True)
    return result


def compute_return_levels_delta(stationary, nonstationary, T, t_eval):
    t_eval = asarray(t_eval)
    
    # ---- Stationary ----
    mu = stationary['location']
    sigma = stationary['scale']
    xi = stationary['shape']
    cov_theta = stationary['cov']  # 3x3 covariance matrix for (mu, sigma, xi)
    
    if abs(xi) < 1e-10:
        z_T = mu + sigma * -log(-log(1 - 1/T))
        grad = array([1, -log(-log(1 - 1/T)), 0])
    else:
        f = (-log(1 - 1/T))**(-xi) - 1
        z_T = mu + sigma/xi * f
        df_dxi = -(-log(1 - 1/T))**(-xi) * log(-log(1 - 1/T))
        grad = array([1, f/xi, -sigma/xi**2 * f + sigma/xi * df_dxi])
    
    var_zT = grad @ cov_theta @ grad
    ci_lower = z_T - 1.96 * sqrt(var_zT)
    ci_upper = z_T + 1.96 * sqrt(var_zT)
    
    # ---- Non-stationary ----
    mu0, mu1, sigma, xi = nonstationary['params_hat']
    cov_theta_ns = nonstationary['cov_mu']  # 4x4 covariance matrix for (mu0, mu1, sigma, xi)
    years_mean = nonstationary['years_mean']
    years_std = nonstationary['years_std']
    
    t_scaled = (t_eval - years_mean) / years_std
    mu_t = mu0 + mu1 * t_scaled
    
    if abs(xi) < 1e-10:
        z_T_ns = mu_t + sigma * -log(-log(1 - 1/T))
        grad_ns = array([1, t_scaled, -log(-log(1 - 1/T)), 0])
    else:
        f = (-log(1 - 1/T))**(-xi) - 1
        z_T_ns = mu_t + sigma/xi * f
        df_dxi = -(-log(1 - 1/T))**(-xi) * log(-log(1 - 1/T))
        grad_ns = array([1, t_scaled, f/xi, -sigma/xi**2*f + sigma/xi*df_dxi])
    
    var_zT_ns = grad_ns @ cov_theta_ns @ grad_ns
    ci_lower_ns = z_T_ns - 1.96 * sqrt(var_zT_ns)
    ci_upper_ns = z_T_ns + 1.96 * sqrt(var_zT_ns)
    
    return {
        'stationary': {'z_T': z_T, 'lower': ci_lower, 'upper': ci_upper},
        'nonstationary': {'z_T': z_T_ns, 'lower': ci_lower_ns, 'upper': ci_upper_ns}
    }
    
    
def compute_all_return_levels(stationary, nonstationary, return_periods, ls_t_eval):
    dic_rl = {}
    for T in return_periods:
        dic_return_levels_t_eval = dict()
        for t_eval in ls_t_eval:
            rl_t = compute_return_levels_delta(
            stationary=stationary, nonstationary=nonstationary, T=T, t_eval=t_eval
            )
            dic_return_levels_t_eval[t_eval] = rl_t
        dic_rl[T] = dic_return_levels_t_eval
        
    df_rl = DataFrame([
        {**dic_rl[T][t_eval][key], 't_eval': t_eval, 'return_period': T, 'model': key}
        for T in return_periods
        for t_eval in dic_rl[T].keys()
        for key in ['stationary', 'nonstationary']
    ])
    return df_rl



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


def fit_stationary_gev_year(year, data, uncertainty='fisher', B:int=150, seed:int | None = None):
    try:
        stationary, stat_notes = fit_stationary_gev_incl_uncertainty(data=data, uncertainty=uncertainty, B=B, seed=seed)
        return year, stationary, stat_notes
    except Exception:
        return year, None
    

def fit_all_years_stationary(df_prepared, uncertainty='fisher', seed=None):
    results_per_year = {}
    annual_max_per_year = df_prepared.groupby('sim_year')['storm_surge'].apply(list)

    for year, data in annual_max_per_year.items():
        stationary, stat_notes = fit_stationary_gev_incl_uncertainty(
            data=data,
            uncertainty=uncertainty,
            B=150,
            seed=seed
        )
        results_per_year[year] = {
            "stationary": stationary,
            "notes": stat_notes
        }

    return results_per_year


def process_gev_per_location(
    loc_id,
    df_prepared,
    location_labels,
    return_periods,
    ls_t_eval,
    B,
    uncertainty="delta",
    min_years=10,
    seed=None,
):

    try:
        lon_loc = df_prepared.lon.unique()[0]
        lat_loc = df_prepared.lat.unique()[0]

        location_info = location_labels.get(
            (round(lon_loc, 6), round(lat_loc, 6)),
            "unknown location"
        )

        logger.info(f"GEV analysis for location {loc_id} · {location_info}")
        annual_max = extract_annual_maxima_at_location(df_prepared, lon=lon_loc, lat=lat_loc)

        if len(annual_max) < min_years:
            return loc_id, None

        years = annual_max["year"].values
        data = annual_max["annual_max"].values

        # -------------------------------------
        logger.info("... Annual maxima for location extracted, now GEV with pooled data.")
        pooled_gev, _ = fit_pooled_gev_with_uncertainty(
            loc_id=loc_id,
            years=years,
            data=data,
            uncertainty=uncertainty,
            trend_params="location",
            print_msg=False,
            B=B,
            seed=seed
        )

        if pooled_gev is None:
            return loc_id, None

        logger.info("... Comparing stationary and non-stationary GEV.")
        comparison = compare_stationary_nonstationary(
            pooled_gev["stationary"],
            pooled_gev["nonstationary"],
            annual_max
        )
        
        logger.info("... Compute return levels.")
        df_all_return_levels = compute_all_return_levels(
            pooled_gev["stationary"],
            pooled_gev["nonstationary"],
            return_periods,
            ls_t_eval
        )

        result = {
            "location_info": location_info,
            "LatLon": (lat_loc, lon_loc),
            "data": annual_max,
            "stationary": pooled_gev["stationary"],
            "nonstationary": pooled_gev["nonstationary"],
            "model_comparison": comparison,
            "return_levels": df_all_return_levels,
        }

        return loc_id, result

    except Exception:
        import traceback
        print(f"\nERROR at location {loc_id}")
        traceback.print_exc()
        return loc_id, None


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
    
    x_mean = mean(x)
    x_std_val = std(x, ddof=1)
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
    annual_stationary, nonstationary, years_mean, years_std, hindcast_start, hindcast_end, z_percentile, factor_m_to_mm
    ):
    
    years_ = arange(hindcast_start, hindcast_end+1)
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

    cov_mu = nonstationary['cov_mu'][:2, :2] * factor_m_to_mm**2
    mu_var = cov_mu[0,0] + 2 * cov_mu[0,1] * years_autoscaled + cov_mu[1,1] * years_autoscaled**2  # shape (67,)
    mu_ns_ci_upper = mu_ns + z_percentile * sqrt(mu_var)
    mu_ns_ci_lower = mu_ns - z_percentile * sqrt(mu_var)
    
    return years_, {
        'stationary': (x_ans, y_ans, weights_ans, results_reg_annual_stat, slope_ans, intercept_ans), 
        'nonstationary': (mu_ns, mu1_ns, mu0_ns, mu_ns_ci_lower, mu_ns_ci_upper),
        }

    
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
    annual_df = fit_annual_gev_mle(df, ls_notes=ls_notes)
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

