from typing import Dict, Optional

from numpy import any, exp, full_like, inf, log, ndarray, sum
from pandas import DataFrame
from scipy import stats
from scipy.optimize import minimize


# !!!ToDO: check extract_annual_maxima - not again selecting max from the annual max dataset!
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

def extract_annual_maxima(
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
            )
    

def fit_stationary_gev(data: ndarray) -> Dict:
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
    if len(data) < 10:
        print(f"Warning: Only {len(data)} observations. Need at least 10 for reliable GEV fit.")
        return None
    
    try:
        c, loc, scale = stats.genextreme.fit(data)
        shape = -c 
        
        ll = sum(stats.genextreme.logpdf(data, c, loc, scale))

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
        }
    except Exception as e:
        print(f"Failed to conduct GEV fitting due to error: {e}")
        return None


def fit_nonstationary_gev(
    years: ndarray, data: ndarray, trend_params: str = 'location'
    ) -> Dict:
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
    if len(data) < 20:
        print(f"Warning: Non-stationary GEV needs ≥20 obs. Have {len(data)}.")
        return None
    
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
    
    stationary = fit_stationary_gev(data)
    if stationary is None:
        return None
    
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
        
        params_out.update({
            'n_obs': len(data),
            'log_likelihood': ll,
            'aic': aic,
            'bic': bic,
            'years_mean': years.mean(),
            'years_std': years.std()
        })
        
        return params_out
        
    except Exception as e:
        print(f"Non-stationary GEV fitting error: {e}")
        return None


def compare_models(stationary: Dict, nonstationary: Dict) -> Dict:
    """
    Compare stationary vs non-stationary GEV using likelihood ratio test.
    
    Returns:
    --------
    dict : Test results and recommendation
    """
    if stationary is None or nonstationary is None:
        return None
    
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


def analyze_location_per_model(
    data_hindcast:DataFrame, model: str, lat: float, lon: float, location_info: str, return_periods: list
    )-> Dict:
        """
        Complete analysis for one model-location combination.
        Fits both stationary and non-stationary GEV.
        """
        annual_max = extract_annual_maxima(data_hindcast, model=model, lon=lon, lat=lat)

        if len(annual_max) < 10:
                return None

        years = annual_max['year'].values
        data = annual_max['annual_max'].values

        print("\t\tconducting stationary GEV...")
        gev_stationary = fit_stationary_gev(data)
        print(f"\t\t\tstationary GEV done (success {gev_stationary != None}); continuing with non-stationary GEV...")
        gev_nonstat_loc = fit_nonstationary_gev(years, data, 'location')
        print(f"\t\t\tnon-stationary GEV done (success {gev_nonstat_loc != None}).")
        comparison = compare_models(gev_stationary, gev_nonstat_loc)

        rl_stationary = calculate_return_levels(gev_stationary, return_periods)
        rl_nonstat_start = calculate_return_levels(
                gev_nonstat_loc, return_periods, year=years.min()
        ) if gev_nonstat_loc else None

        rl_nonstat_end = calculate_return_levels(
                gev_nonstat_loc, return_periods, year=years.max()
        ) if gev_nonstat_loc else None

        return {
                'model': model,
                'location': (lat, lon),
                'location info': location_info,
                'annual_maxima': annual_max,
                'gev_stationary': gev_stationary,
                'gev_nonstationary': gev_nonstat_loc,
                'model_comparison': comparison,
                'return_levels_stationary': rl_stationary,
                'return_levels_nonstationary_start': {
                    'year': int(years.min()),
                    'values': rl_nonstat_start
                    },
                'return_levels_nonstationary_end': {
                    'year': int(years.max()),
                    'values': rl_nonstat_end
                    }
        }


def analyze_per_location(
        data_hindcast:DataFrame, lat: float, lon: float, location_info: str, return_periods: list
        )-> Dict:
        """
        Complete analysis for one model-location combination.
        Fits both stationary and non-stationary GEV.
        """
        annual_max = extract_annual_maxima(data_hindcast, lon=lon, lat=lat)
        
        if len(annual_max) < 10:
                return None

        years = annual_max['year'].values
        data = annual_max['annual_max'].values

        print("\t\tconducting stationary GEV...")
        gev_stationary = fit_stationary_gev(data)
        print(f"\t\t\tstationary GEV done (success {gev_stationary != None}); continuing with non-stationary GEV...")
        gev_nonstat_loc = fit_nonstationary_gev(years, data, 'location')
        print(f"\t\t\tnon-stationary GEV done (success {gev_nonstat_loc != None}).")
        comparison = compare_models(gev_stationary, gev_nonstat_loc)

        rl_stationary = calculate_return_levels(gev_stationary, return_periods)
        rl_nonstat_start = calculate_return_levels(
                gev_nonstat_loc, return_periods, year=years.min()
        ) if gev_nonstat_loc else None

        rl_nonstat_end = calculate_return_levels(
                gev_nonstat_loc, return_periods, year=years.max()
        ) if gev_nonstat_loc else None

        return {'location': (lat, lon),
                'location info': location_info,
                'annual_maxima': annual_max,
                'gev_stationary': gev_stationary,
                'gev_nonstationary': gev_nonstat_loc,
                'model_comparison': comparison,
                'return_levels_stationary': rl_stationary,
                'return_levels_nonstationary_start': {
                        'year': int(years.min()),
                        'values': rl_nonstat_start
                        },
                'return_levels_nonstationary_end': {
                        'year': int(years.max()),
                        'values': rl_nonstat_end
                        },
                'data from model(s)':None,
        }
