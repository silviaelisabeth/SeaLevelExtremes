import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import func_plotting as dbplt
import func_preparation as dbf
import func_utils as ut
import statsmodels.api as sm
from IPython.display import Markdown, display
from numpy import (any, array, exp, finfo, float64, full_like, generic, inf,
                   isfinite, isnan, linalg, linspace, log, nan, ndarray,
                   ones_like, sqrt, sum, vstack, zeros)
from pandas import DataFrame, to_numeric
from scipy import optimize, stats
from scipy.optimize import approx_fprime, minimize
from scipy.stats import norm


def extract_annual_maxima_unique(
    data_hindcast: DataFrame, lon: float, lat: float, model: Optional[str] = None
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
    data_hindcast: DataFrame, lon: float, lat: float, model: Optional[str] = None
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


def fit_stationary_gev(data: ndarray, year:Optional[int]=None) -> Tuple[Optional[dict], list[str]]:
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
        print(message)
        ls_notes.append(message)
        return None, ls_notes
    
    try:
        c, loc, scale = stats.genextreme.fit(data)
        shape = -c 
        
        logpdf = stats.genextreme.logpdf(data, c, loc, scale)
        if not all(isfinite(logpdf)):
            ls_notes.append("Non-finite log-likelihood (GEV support violation)")
        ll = sum(logpdf)

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
        print(f"Failed to conduct GEV fitting due to error: {e}")
        return None, ls_notes


def fit_nonstationary_gev(
    years: ndarray, data: ndarray, trend_params: str = 'location'
    ) -> Tuple[Dict, list]:
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
        print(message)
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
        
        
        if any(sigma_t <= 0):
            return inf
        
        z = (data - mu_t) / sigma_t
        
        if abs(xi) < 1e-10:  # Gumbel case
            ll = -sum(log(sigma_t)) - sum(z) - sum(exp(-z))
        else:
            term = 1 + xi * z
            if any(term <= 0):
                return inf
            ll = (-sum(log(sigma_t)) - 
                    (1 + 1/xi) * sum(log(term)) - 
                    sum(term**(-1/xi)))
        
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
        print(f"Non-stationary GEV fitting error: {e}")
        return None, ls_notes


def compare_models(stationary: Dict, nonstationary: Dict) -> Dict:
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
    gev_params: Dict, return_periods: list, year: Optional[float] = None,
    cov_matrix: Optional[ndarray] = None, alpha: float = 0.05
    ) -> Dict:
    """
    Calculate return levels from GEV parameters, optionally with uncertainty.
    """
    if gev_params is None:
        return None
    
    # compute mu, sigma, xi as before
    if 'trend_in' in gev_params:
        # non-stationary GEV
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
        # stationary GEV
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
        
        # include uncertainty if covariance matrix is given
        if cov_matrix is not None:
            if 'trend_in' in gev_params and gev_params['trend_in'] == 'location':
                # Gradient for non-stationary μ(t)
                dz_dmu0 = 1
                dz_dmu1 = t
                dz_dsigma = ((-log(p))**(-xi) - 1)/xi if abs(xi) >= 1e-10 else -log(-log(p))
                dz_dxi = (-sigma/xi**2 * ((-log(p))**(-xi)-1) +
                          (sigma/xi) * ((-log(p))**(-xi) * log(-log(p)))) if abs(xi) >= 1e-10 else 0
                grad = array([dz_dmu0, dz_dmu1, dz_dsigma, dz_dxi])
            else:
                # Stationary gradient
                dz_dmu = 1
                dz_dsigma = ((-log(p))**(-xi) - 1)/xi if abs(xi) >= 1e-10 else -log(-log(p))
                dz_dxi = (-sigma/xi**2 * ((-log(p))**(-xi)-1) +
                          (sigma/xi) * ((-log(p))**(-xi) * log(-log(p)))) if abs(xi) >= 1e-10 else 0
                grad = array([dz_dmu, dz_dsigma, dz_dxi])

            # Delta-method variance
            var_z = grad.T @ cov_matrix @ grad
            ci_lower = z_T - 1.96 * sqrt(var_z)
            ci_upper = z_T + 1.96 * sqrt(var_z)
            results[f'{T}-year'] = {'return_level': z_T, 'CI_lower': ci_lower, 'CI_upper': ci_upper}
        else:
            results[f'{T}-year'] = {'return_level': z_T}
    
    return results


def calculate_return_levels_wo_ci(
    gev_params: Dict, return_periods: list, year: Optional[float] = None
    ) -> Dict:
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
    for T in return_periods:
        p = 1 - 1/T
        
        if abs(xi) < 1e-10:  # Gumbel
            z_p = mu - sigma * log(-log(p))
        else:
            z_p = mu + (sigma / xi) * ((-log(p))**(-xi) - 1)
        
        return_levels[f'{T}-year'] = z_p
    
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
            return -sum(stats.genextreme.logpdf(data, c, loc=mu_t, scale=sigma))
        theta_hat = array([gev_params['mu0'], gev_params['mu1'], gev_params['sigma'], gev_params['xi']])
    else:
        # Stationary GEV: μ, σ, ξ
        def neg_loglik(theta):
            xi, mu, sigma = theta
            c = -xi
            return -sum(stats.genextreme.logpdf(data, c, loc=mu, scale=sigma))
        theta_hat = array([gev_params['shape'], gev_params['location'], gev_params['scale']])
    
    # Numerical Hessian via finite differences
    epsilon = sqrt(finfo(float).eps)
    def grad(theta):
        return optimize.approx_fprime(theta, neg_loglik, epsilon)
    
    H = optimize.approx_fprime(theta_hat, grad, epsilon)
    
    # Covariance = inverse of Hessian
    try:
        cov_matrix = linalg.inv(H)
    except linalg.LinAlgError:
        print("Warning: Hessian not invertible; returning None")
        return None
    
    return cov_matrix


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
        if any(sigma_t <= 0):
            return inf
        z = (data - mu_t) / sigma_t
        if abs(xi) < 1e-10:
            ll = -sum(log(sigma_t)) - sum(z) - sum(exp(-z))
        else:
            term = 1 + xi * z
            if any(term <= 0):
                return inf
            ll = -sum(log(sigma_t)) - (1 + 1/xi) * sum(log(term)) - sum(term**(-1/xi))
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
        print("Hessian is singular; cannot compute CI")
        return None, None

    # Delta method: Var(mu_t) = grad(mu_t)^T * cov * grad(mu_t)
    mu0, mu1 = params[0], params[1]
    mu_t = mu0 + mu1 * t
    grad_mu_t = vstack([ones_like(t), t]).T  # derivative wrt mu0, mu1
    var_mu_t = sum(grad_mu_t @ cov[:2,:2] * grad_mu_t, axis=1)  # only first 2 params
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
    
    return DataFrame(rows).sort_values('Year').reset_index(drop=True).set_index('Year')


def execute_and_store_stat_gev_per_year(results: dict, store_results:bool, return_periods:list) -> dict:
    dic_notes = {}
    for en, site_id in enumerate(results.keys()):
        ls_notes = []
        grp_per_year = results[site_id]['data'].groupby('year')
        
        message = f"Conducting stationary GEV for siteID {site_id} (#{en+1} out of {len(results.keys())}) grouped per year..."
        print(message)
        ls_notes.append(message)

        results_stat_per_year_at_location = dict()
        results_return_values_per_year = dict()
        en = 0
        for year, group in grp_per_year:
            en+=1
            print(f"\t...{int(year)} (#{en} out of {len(grp_per_year)} years)", end="\r") 

            data = (group
                    .sort_values('year')
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
                print(message)
                rl_stationary = None

            results_stat_per_year_at_location[int(year)] = gev_stationary
            results_return_values_per_year[int(year)] = rl_stationary
        print("\nDone!\n") 

        df_stat_gev_per_year = DataFrame.from_dict(results_stat_per_year_at_location).T
        results[site_id]['fit results']['gev_stationary']['analysis_per_year'] = df_stat_gev_per_year.dropna()

        df_stat_return_levels_per_year = convert_annual_return_levels_with_ci_into_dataframe(results_return_values_per_year)
        results[site_id]['fit results']['gev_stationary']['return_levels_per_year'] = df_stat_return_levels_per_year.dropna()
        
        if store_results:
            if 'file_path_report' in results[site_id].keys():
                file_path = results[site_id]['file_path_report']
            else:
                file_path = results[site_id]['file location']
                
            df_stat_gev_per_year.to_parquet(os.path.join(file_path, f"statGEV_per_year.parquet"))
        
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


def analyze_per_location(
    data_hindcast:DataFrame, site_id: int, lat: float, lon: float, location_info: str, return_periods: list
    )-> Tuple[dict, list]:
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

    print("\tConducting stationary GEV...")
    gev_stationary, warnings = fit_stationary_gev(data)
    ls_notes.append(warnings)
    print(f"\t → Stationary GEV done (success {gev_stationary != None})")
    print("\tContinuing with non-stationary GEV...")
    gev_nonstat_loc, warnings = fit_nonstationary_gev(years, data, 'location')
    ls_notes.append(warnings)
    print(f"\t → Non-stationary GEV done (success {gev_nonstat_loc != None})")
    comparison = compare_models(gev_stationary, gev_nonstat_loc)

    print('Return Levels Stationary GEV...')
    cov_stationary = compute_cov_matrix(gev_stationary, data)
    rl_stationary = calculate_return_levels(
        gev_params=gev_stationary, return_periods=return_periods, cov_matrix=cov_stationary
        )

    print('Return Levels NON-Stationary GEV...')
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
    result_location: dict, plot_period_evolution:list[str], export_report:bool=True, path_export:Optional[str]=None, 
    print_msg:Optional[bool]=False, display_results:Optional[bool]=True
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
            result_display=result_location, location_in_example=location_label, ls_messages=ls_messages,
            print_msg=print_msg
            )
        
        if export_report and path_export: 
            save_path = path_export + today_
            Path(save_path).mkdir(parents=True, exist_ok=True)  
            full_file_path = save_path + f"/GEVanalysis_{country}_{lat_str}|{lon_str}_{today_}.parquet"   
            
            df_messages = DataFrame({'messages': ls_messages}) 
            df_messages.to_parquet(full_file_path, index=False)
            #with open(full_file_path, 'w') as f:
            #    f.write('\n'.join(ls_messages))
        else:
            save_path = None
            
        # -----------------------------------------------------------------------------------------------------------
        dbplt.plot_pooled_analysis(
            result=result_location, 
            lat_lon_tuple=result_location['location'], 
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


def store_report_stationary_gev_per_year(site_data:dict):
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
