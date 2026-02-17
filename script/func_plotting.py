import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import func_utils as ut
import matplotlib
import matplotlib.gridspec as gridspec
import pydeck as pdk
import seaborn as sns
from IPython.display import display
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import arange, array, nan, ndarray, sqrt
from pandas import DataFrame
from statsmodels.regression.linear_model import RegressionResultsWrapper

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger("gev_analysis")

rcParams['font.family'] = [
    'Noto Sans',
    'Noto Sans Arabic',
    'Noto Sans Tifinagh',
    'Noto Sans CJK JP',
    'Noto Sans Devanagari'
]
sns.set_style('whitegrid')


def create_map_location_missing_valid_data(
    missing_locations:DataFrame,
    df_valid:DataFrame,
    radius_marker_m:int=3500, 
    color_missing:str='#980019FF',
    color_valid:str='#7BAA80FF',
    dir_export:str='../output/exploration',
    store_map:bool=False,
    ) -> None:

    missing_locations['info'] = "Missing data (n_obs = 0)"

    layer_missing = pdk.Layer(
        "ScatterplotLayer", data=missing_locations, get_position='[lon, lat]', get_radius=radius_marker_m,
        radius_scale=1, radius_min_pixels=1, radius_max_pixels=7, get_fill_color=ut.hex_to_rgba(color_missing),
        pickable=True,
    )

    layer_valid = pdk.Layer(
        "ScatterplotLayer", data=df_valid, get_position='[lon, lat]', get_radius=radius_marker_m, radius_scale=1,
        radius_min_pixels=1, radius_max_pixels=7, get_fill_color=ut.hex_to_rgba(color_valid), pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=missing_locations['lat'].mean(), longitude=missing_locations['lon'].mean(), zoom=4
    )

    deck = pdk.Deck(
        layers=[layer_valid, layer_missing], initial_view_state=view_state,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"text": "Lat: {lat:.5f}\nLon: {lon:.5f}\nInfo: {info}"}
    )

    if store_map:
        save_dir = Path(dir_export)
        save_dir.mkdir(parents=True, exist_ok=True) 
        file_name = save_dir / f"map_missingValidData_{round(radius_marker_m/1000,1)}kmRadius.html"
        deck.to_html(file_name, notebook_display=False, open_browser=False)
        logger.info(f"Map saved as {file_name}. You can open it in your browser and interact with it.")
    
    return deck    


def plot_gev_mu_trend(
    df: DataFrame,
    weights: list,
    year_grid: ndarray,
    year_mean: float,
    y_pred: ndarray,
    wls_delta:RegressionResultsWrapper,
    nonstat_years: array,
    nonstat: dict,
    lat:float,
    lon:float,
    site_id:int,
    display_results: bool = True,
    fontsize: float = 11,
    figsize: tuple[float, float] = (13, 3.5), 
    axes_color: str = '#333333',
    markers_color: str = "#99E3DDFF",
    colors_reg: list = ['#333333FF', '#7F6C7BFF']  
    ) -> Figure:
    """
    Plot GEV μ trend per location with WLS delta-method fit and optional non-stationary MLE comparison.
    """

    intercept, slope = wls_delta.params

    mask = df['location'].notna() & df['var_mu'].notna()
    df_plot = df[mask].copy()
    weights_plot = array(weights)[mask]

    # annual stationary GEV - CI
    cov = wls_delta.cov_params().values  
    t_centered = year_grid - year_mean
    pred_var_t = cov[0,0] + t_centered**2 * cov[1,1] + 2 * t_centered * cov[0,1]
    pred_std_t = sqrt(pred_var_t)
    y_upper = y_pred + 1.96 * pred_std_t
    y_lower = y_pred - 1.96 * pred_std_t
    
    # ----------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        df_plot['year'], df_plot['location'], s=weights_plot/1000, 
        alpha=0.6, c=markers_color, label='Annual estimates'
    ) 
    for _, row in df_plot.iterrows():
        ax.text(row.year, row.location, f"{row.n_obs}", fontsize=8, alpha=0.6)
    
    ax.plot(
        year_grid, y_pred, color='black', 
        label=f'Annual stationary μ(t)\nslope={slope:.5f}, intercept={intercept:.4f} (centered {int(year_mean)})'
    )
    ax.fill_between(year_grid, y_lower, y_upper, color=colors_reg[0], alpha=0.15, label='95% CI (annual stationary)')
    
    if nonstat is not None and isinstance(nonstat, dict) and nonstat.get('CI') is not None and 'mu_pred' in nonstat['CI']:
        ax.plot(
            nonstat_years,
            nonstat['CI']['mu_pred'],
            color=colors_reg[1],
            linewidth=1.5,
            label=(
                f"Non-stationary μ(t)\nslope={nonstat['mu1']:.5f}, intercept={nonstat['mu0']:.4f} (centered {int(year_mean)})"
            ),
            alpha=0.8
        )
        ax.fill_between(
            nonstat_years, nonstat['CI']['mu_lower'], nonstat['CI']['mu_upper'], color=colors_reg[1], 
            alpha=0.15, label='95% CI (non-stationary)'
        )
    else:
        logger.info("nonstat['CI'] is None or missing 'mu_pred'")

    leg = ax.legend(loc=0, edgecolor=axes_color, borderpad=.65, fontsize=fontsize*0.75)
    leg.get_frame().set_linewidth(.5)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.axhline(y=ax.get_ylim()[0], color=axes_color, linewidth=1.2, zorder=10)
    ax.axvline(x=ax.get_xlim()[0], color=axes_color, linewidth=1.2, zorder=10)

    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)

    ax.grid(True, alpha=0.3, color='lightgrey')
    ax.set_title(
        f'GEV μ Trend with Fixed Scale & Shape for lat|lon {lat:.3f}|{lon:.3f} (siteID {site_id})', 
        fontsize=fontsize*1.25
        )
    ax.set_xlabel('Year', fontsize=fontsize)
    ax.set_ylabel('GEV location parameter', fontsize=fontsize)

    plt.tight_layout()
    fig.canvas.draw()
    plt.show() if display_results else plt.close(fig)

    return fig


def create_parameter_summary(
    stat: Optional[dict], 
    nonstat: Optional[dict], 
    comp: Optional[dict], 
    box_x: float, 
    box_y: float, 
    ax: Axes, 
    fontsize: float = 9, 
    linespace: float = 1.5,
    bbox: Optional[dict]=dict(boxstyle='round', facecolor='#F5F5F5FF', alpha=0.5)
    ) -> None:
    ax.axis('off')
    
    info_text = r"STATIONARY GEV" "\n"
    if stat:
        info_text += f"  μ = {stat['location']:.4f}\n"
        info_text += f"  σ = {stat['scale']:.4f}\n"
        info_text += f"  ξ = {stat['shape']:.4f}\n"
        info_text += f"  Type: {stat['dist_type']}\n"
        info_text += f"  AIC = {stat['aic']:.1f}\n"
    
    if nonstat:
        info_text += "\n"r"NON-STATIONARY GEV" "\n"
        info_text += f"  μ(t) = {nonstat['mu0']:.4f} + {nonstat['mu1']:.4f}·t\n"
        info_text += f"  σ = {nonstat['sigma']:.4f}\n"
        info_text += f"  ξ = {nonstat['xi']:.4f}\n"
        info_text += f"  AIC = {nonstat['aic']:.1f}\n"
    
    if comp:
        info_text += "\n" r"SIGNIFICANCE TEST" "\n"
        if 'p_value' in comp.keys():
            info_text += f"  p-value = {comp['p_value']:.5f}\n" 
        if 'delta_aic' in comp.keys():
            info_text += f"  ΔAIC = {comp['delta_aic']:.1f}\n" 
    
    ax.text(
        box_x, box_y, info_text, transform=ax.transAxes, linespacing=linespace, fontsize=fontsize, 
        verticalalignment='top', family='sans serif', bbox=bbox
        )
    

def plot_annual_max_with_trends(
    ax:Axes, 
    loc_id:int,
    annual_max:DataFrame, 
    nonstat: Optional[dict],
    comp: Optional[dict],
    return_levels:dict, 
    colors_trends:str, 
    linestyle_trends:list[str], 
    axes_color:str, 
    color_markers:str='#99E3DDFF', 
    ms:int=6, 
    fontsize:float=10, 
    ls_periods:list = ['10-year', '50-year', '100-year']
    ) -> bool:
    """
    Plot annual maxima with stationary and non-stationary GEV return levels.
    Includes CI if return_levels contains them.
    """
    skip_non_stat = False

    # --- Plot annual maxima ---
    ax.plot(
        annual_max['year'], annual_max['annual_max'], 'o', color=color_markers, markersize=ms, label='Annual max'
    )

    # --- Plot stationary return levels ---
    if return_levels.get('stationary'):
        for en, period in enumerate(ls_periods):
            if period not in return_levels['stationary']:
                logger.info(f" - Warning: period {period} not found in return levels stationary, skipping...")
                continue

            lvl = return_levels['stationary'][period]
            if isinstance(lvl, dict):
                y = lvl['return_level']
                ax.fill_between(
                    [annual_max['year'].min(), annual_max['year'].max()], 
                    lvl.get('CI_lower', y), lvl.get('CI_upper', y), color=colors_trends, alpha=0.15
                )
            else:
                y = lvl
            ax.axhline(y=y, linestyle=linestyle_trends[en], color=colors_trends, label=f'{period} (stationary)')

    # --- Plot non-stationary μ(t) if significant ---
    pval = comp.get('p_value') if isinstance(comp, dict) else None
    if nonstat and pval is not None and pval < 0.05:
        if 'CI' in nonstat:
            ax.fill_between(
                annual_max['year'], nonstat['CI']['mu_lower'], nonstat['CI']['mu_upper'], 
                color='red', alpha=0.15, label='95% CI (non-stationary μ)'
            )
        ax.plot(
            annual_max['year'], nonstat['CI']['mu_pred'], 'r-', linewidth=1.5, label='Non-stationary μ(t)', alpha=0.8
            )

    else:
        pval_str = "None" if pval is None else f"{pval:.2f}"
        logger.info(
            f"\t WARNING! Skipping non-stationary with model comparison {pval_str} for location {loc_id} "
            f"(threshold for non-stationary 0.05)"
        )
        skip_non_stat = True

    # --- Labels and title ---
    years_unique = annual_max.year.unique()
    ax.set_xlabel('Year', fontsize=fontsize*0.9)
    ax.set_ylabel('Storm Surge (m)', fontsize=fontsize*0.9)
    ax.set_title(
        f'Annual Maximum Storm Surge (n={len(annual_max)} samples; {len(years_unique)} unique years b/w '
        f'{int(years_unique.min())}-{int(years_unique.max())})', 
        fontsize=fontsize
    )

    # --- Legend & Styling ---
    leg = ax.legend(loc=0, edgecolor=axes_color, borderpad=.65, fontsize=fontsize*0.75)
    leg.get_frame().set_linewidth(.5)

    ax.grid(True, alpha=0.3, color='lightgrey')
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.axhline(y=ax.get_ylim()[0], color=axes_color, linewidth=1.2, zorder=10)
    ax.axvline(x=ax.get_xlim()[0], color=axes_color, linewidth=1.2, zorder=10)
    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)

    return skip_non_stat


def plot_model_comparison(
    ax:Axes, 
    comp: Optional[dict],
    stat: Optional[dict],
    nonstat: Optional[dict],
    colors_models:list[str], 
    models_names:list[str] =['Stationary', 'Non-Stationary'],
    axes_color:str = '#333333', 
    bbox_color:str = '#F5F5F5FF', 
    leg_x:float = 0.15, 
    leg_y:float = 0.5, 
    fontsize:float = 10
    ) -> None:
    aics = [stat['aic'], nonstat['aic']]
    
    bars = ax.bar(models_names, aics, color=colors_models, alpha=0.7, edgecolor=None)
    ax.set_ylabel('AIC (lower is better)', fontsize=fontsize*0.9)
    ax.set_title(f'Model Comparison p = {comp["p_value"]:.4f}', fontsize=fontsize*0.9)
    
    for bar, aic in zip(bars, aics):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{aic:.1f}', ha='center', va='bottom')
    
    ax.text(
        leg_x, leg_y, r">Recommendation · " f"\n{comp['recommendation']}",  transform=ax.transAxes, 
        fontsize=fontsize*0.75, verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor=bbox_color, edgecolor=axes_color, linewidth=0.5, alpha=0.3),
        )
    
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.axhline(y=ax.get_ylim()[0], color=axes_color, linewidth=1.2, zorder=10)
    ax.axvline(x=ax.get_xlim()[0], color=axes_color, linewidth=1.2, zorder=10)

    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)

    ax.grid(True, alpha=0.3, color='lightgrey')


def plot_level_evolution(
    ax:Axes, 
    return_levels:dict, 
    periods:list[str], 
    color_levels:list[str] = ['#008A80FF','#CAA5C2FF'], 
    width:float = 0.35, 
    fontsize:float = 10, 
    axes_color:str = '#333333', 
    skip_non_stat:bool=False
    ) -> None:
    """
    Plot evolution of return levels for non-stationary GEV including CI as shaded bars.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    return_levels : dict
        Dictionary with keys 'nonstationary_start' and 'nonstationary_end', each containing:
            - 'year': year of prediction
            - 'values': dict of return levels per period, either scalar or dict with 'return_level' and 'CI_lower', 'CI_upper'
    periods : list[str]
        List of return period labels, e.g. ['10-year', '50-year', '100-year']
    color_levels : list[str]
        Colors for start and end bars
    width : float
        Width of bars
    fontsize : float
        Font size for labels
    axes_color : str
        Color for axes
    skip_non_stat : bool
        If True, skip plotting non-stationary return levels
    """
    
    x = arange(len(periods))
    def extract_level_and_ci(entry, period):
        if entry is None or not isinstance(entry, dict) or 'values' not in entry or entry['values'] is None:
            return nan, 0, 0

        lvl = entry['values'].get(period, None)
        if lvl is None:
            return nan, 0, 0
        
        if isinstance(lvl, dict):
            return (
                lvl.get('return_level', nan),
                lvl.get('return_level', nan) - lvl.get('CI_lower', 0),
                lvl.get('CI_upper', 0) - lvl.get('return_level', nan)
            )
        else:
            return lvl, 0, 0

    # ----------------- START -----------------
    start_entry = return_levels.get('nonstationary_start')
    if start_entry is None:
        logger.info("\t WARNING: No non-stationary start return levels")
        levels_start = [nan]*len(periods)
        err_lower_start = [0]*len(periods)
        err_upper_start = [0]*len(periods)
        year_start = "N/A"
    else:
        year_start = start_entry.get('year', "N/A")
        levels_start, err_lower_start, err_upper_start = zip(*[
            extract_level_and_ci(start_entry, p) for p in periods
        ])

    levels_start = array(levels_start)
    yerr_start = array([err_lower_start, err_upper_start])
    
    ax.bar(
        x - width/2, levels_start, width, yerr=yerr_start, capsize=4, alpha=0.7,
        color=color_levels[0], label=f"Start {year_start}"
    )
    
    # ----------------- END -----------------
    if not skip_non_stat:
        end_entry = return_levels.get('nonstationary_end')
        if end_entry is None:
            logger.info("\t WARNING: No non-stationary end return levels")
            levels_end = [nan]*len(periods)
            err_lower_end = [0]*len(periods)
            err_upper_end = [0]*len(periods)
            year_end = "N/A"
        else:
            year_end = end_entry.get('year', "N/A")
            levels_end, err_lower_end, err_upper_end = zip(*[
                extract_level_and_ci(end_entry, p) for p in periods
            ])
        
        levels_end = array(levels_end)
        yerr_end = array([err_lower_end, err_upper_end])
        
        ax.bar(
            x + width/2, levels_end, width, color=color_levels[1], yerr=yerr_end, capsize=4, alpha=0.7,
            label=f"End {year_end}", 
        )
    
    # ----------------- FINAL AXES -----------------
    ax.set_xticks(x)
    ax.set_xticklabels(periods, fontsize=fontsize)
    ax.set_ylabel("Return Level (m)", fontsize=fontsize)
    ax.legend(fontsize=fontsize*0.8, edgecolor=axes_color)
    ax.grid(True, alpha=0.3, color='lightgrey')
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.axhline(y=ax.get_ylim()[0], color=axes_color, linewidth=1.2, zorder=10)
    ax.axvline(x=ax.get_xlim()[0], color=axes_color, linewidth=1.2, zorder=10)
    ax.set_title(
        "Return Levels Non-Stationary Evolution incl Uncertainty" if not skip_non_stat else
        "Return Levels Stationary incl. Uncertainty", fontsize=fontsize
    )


def plot_analysis(
    results: dict,
    model: str,
    lat_lon_tuple: tuple[float, float],
    location_info: str,
    periods_evolution: list[str] = ['10-year', '50-year', '100-year'],
    save_path: str = None,
    width_bar_returns: float = 0.35,
    box_parameters_x:float = 0.05, 
    box_parameters_y: float = 0.95,
    color_markers: str = '#99E3DDFF',
    colors_trends: str = "#1D141BFF",
    bbox_color: str = '#F5F5F5FF',
    colors_models: list[str] = ['#B887ADFF', '#008A80FF'],
    colors_return_levels: list[str] = ['#008A80FF','#CAA5C2FF'],
    linestyle_trends: list = ['dashdot', 'dashed', 'solid'],
    axes_color: str = '#333333',
    leg_comparison_x: float = 0.25,
    leg_comparison_y: float = 0.5,
    fontsize: float = 9,
    figsize: tuple[float, float] = (15, 8),
    linespace: float = 1.5,
    display_results: bool = False,
    ) -> None:
    """Create comprehensive visualization."""

    if model not in results or lat_lon_tuple not in results[model].keys():
        logger.info(f"No results for {model}, {lat_lon_tuple}")
        return
    
    result = results[model][lat_lon_tuple]
    location = result['location info']
    annual_max = result['annual_maxima']
    stat = result['gev_stationary']
    nonstat = result['gev_nonstationary']
    comp = result['model_comparison']

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], figure=fig)

    ax_top_left = fig.add_subplot(gs[0, 0])
    ax_bottom_left = fig.add_subplot(gs[1, 0])
    ax_top_right = fig.add_subplot(gs[0, 1])
    ax_bottom_right = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        f'GEV Analysis: {model} - lat|lon = {str(lat_lon_tuple[0].round(3))}|{str(lat_lon_tuple[1].round(3))} '
        f'closest point {ut.normalize_location_text(location_info)}', 
        fontsize=fontsize*1.25, fontweight='bold'
        )

    # ----------------------------------------------------------------------------
    # Plot TOP-LEFT: Annual maxima with trends
    skip_non_stat = plot_annual_max_with_trends(
        annual_max=annual_max, return_levels=result['return_levels'], nonstat=nonstat, comp=comp,
        ls_periods=periods_evolution, colors_trends=colors_trends, axes_color=axes_color, color_markers=color_markers, 
        linestyle_trends=linestyle_trends, ms=6, fontsize=fontsize, ax=ax_top_left, loc_id=location['site_id'], 
        )

    # ----------------------------------------------------------------------------   
    # Plot TOP-RIGHT: GEV parameters summary
    create_parameter_summary(
        stat=stat, nonstat=nonstat, comp=comp, 
        box_x=box_parameters_x, box_y=box_parameters_y, 
        bbox=dict(boxstyle='round', facecolor='#F5F5F5FF', alpha=0.5), 
        fontsize=fontsize*0.7, linespace=linespace, ax=ax_top_right
        )

    # ----------------------------------------------------------------------------
    # Plot BOTTOM-LEFT: Return levels evolution
    if result['return_levels_nonstationary_start'] and result['return_levels_nonstationary_end']:
        plot_level_evolution(
            return_levels=result['return_levels'], periods=periods_evolution, width=width_bar_returns, 
            color_levels=colors_return_levels, fontsize=fontsize, axes_color=axes_color, skip_non_stat=skip_non_stat,
            ax=ax_bottom_left, 
        )

    # ----------------------------------------------------------------------------
    # Plot BOTTOM-RIGHT: Model comparison
    if stat and nonstat and comp:
        plot_model_comparison(
            comp=result['model_comparison'], stat=result['gev_stationary'], nonstat=result['gev_nonstationary'], 
            models_names=['Stationary', 'Non-Stationary'], colors_models=colors_models, bbox_color=bbox_color, 
            leg_x=leg_comparison_x, leg_y=leg_comparison_y, fontsize=fontsize, ax=ax_bottom_right
    )

    plt.tight_layout()
    fig.canvas.draw()
    
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True) 
        
        time_date = datetime.today().date().isoformat()
        country = location.split(',')[-1].strip()
        lat_str, lon_str = str(round(float(lat_lon_tuple[0]), 3)), str(round(float(lat_lon_tuple[1]),3))
    
        file_name = save_dir / f"/GEVanalysis_{model}_{country}_{lat_str}|{lon_str}_{time_date}.png"
        logger.info(f"\t saving GEV analysis to {save_dir}")

        plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show() if display_results else plt.close(fig)
    

def plot_pooled_analysis(
    result: dict,
    site_id: int,
    periods_evolution: list[str] = ['10-year', '50-year', '100-year'],
    save_path: str = None,
    width_bar_returns: float = 0.35,
    box_parameters_x: float = 0.05, 
    box_parameters_y: float = 0.95,
    color_markers: str = '#99E3DDFF',
    colors_trends: str = "#1D141BFF",
    bbox_color: str = '#F5F5F5FF',
    colors_models: list[str] = ['#B887ADFF', '#008A80FF'],
    colors_return_levels: list[str] = ['#008A80FF','#CAA5C2FF'],
    linestyle_trends: list = ['dashdot', 'dashed', 'solid'],
    axes_color: str = '#333333',
    leg_comparison_x: float = 0.25,
    leg_comparison_y: float = 0.5,
    fontsize: float = 9,
    figsize: tuple[float, float] = (15, 8),
    linespace: float = 1.5,
    display_results: bool = False,
    ) -> Figure:
    """Create comprehensive visualization."""
    location_info = result['location info']
    lat, lon = str(location_info['lat'].round(3)), str(location_info['lon'].round(3))
    annual_max = result['data']
    stat = result['fit results']['gev_stationary']
    nonstat = result['fit results']['gev_nonstationary']
    comp = result['model_comparison']
    
    # ----------------------------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], figure=fig)

    ax_top_left = fig.add_subplot(gs[0, 0])
    ax_bottom_left = fig.add_subplot(gs[1, 0])
    ax_top_right = fig.add_subplot(gs[0, 1])
    ax_bottom_right = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        f'GEV Analysis for location {lat}|{lon} (lat|lon) '
        f'closest point {ut.normalize_location_text(location_info['description'])}', 
        fontsize=fontsize*1.25, fontweight='bold'
        )

    # ----------------------------------------------------------------------------
    # Plot TOP-LEFT: Annual maxima with trends      
    skip_non_stat = plot_annual_max_with_trends(
        annual_max=annual_max, return_levels=result['return_levels'], nonstat=nonstat, comp=comp, fontsize=fontsize, 
        ls_periods=periods_evolution, colors_trends=colors_trends, axes_color=axes_color, color_markers=color_markers, 
        linestyle_trends=linestyle_trends, ms=6, ax=ax_top_left, loc_id=site_id
        )

    # ----------------------------------------------------------------------------   
    # Plot TOP-RIGHT: GEV parameters summary
    create_parameter_summary(
        stat=stat, nonstat=nonstat, comp=comp, box_x=box_parameters_x, box_y=box_parameters_y, fontsize=fontsize*0.7, 
        bbox=dict(boxstyle='round', facecolor='#F5F5F5FF', alpha=0.5), linespace=linespace, ax=ax_top_right
        )

    # ----------------------------------------------------------------------------
    # Plot BOTTOM-LEFT: Return levels evolution
    if result['return_levels']['nonstationary_start'] and result['return_levels']['nonstationary_end']:
        plot_level_evolution(
            return_levels=result['return_levels'], periods=periods_evolution, width=width_bar_returns, 
            color_levels=colors_return_levels, fontsize=fontsize, axes_color=axes_color, skip_non_stat=skip_non_stat,
            ax=ax_bottom_left, 
        )

    # ----------------------------------------------------------------------------
    # Plot BOTTOM-RIGHT: Model comparison
    if stat and nonstat and comp:
        plot_model_comparison(
            comp=result['model_comparison'], stat=result['fit results']['gev_stationary'], 
            nonstat=result['fit results']['gev_nonstationary'], models_names=['Stationary', 'Non-Stationary'], 
            colors_models=colors_models, bbox_color=bbox_color, leg_x=leg_comparison_x, leg_y=leg_comparison_y, 
            fontsize=fontsize, ax=ax_bottom_right
    )

    plt.tight_layout()
    fig.canvas.draw()  
    
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        country = location_info['description'].split(',')[-1].strip()
        country_clean = ut.sanitize_filename(country)
        lat_clean = str(lat).replace('.', '_')
        lon_clean = str(lon).replace('.', '_')

        file_name = f"GEVanalysis_pooled_{site_id}_{country_clean}_{lat_clean}_{lon_clean}.png"
        file_path = save_dir / file_name 

        logger.info(f"\t saving GEV analysis to {save_dir} as {file_name}")
        plt.savefig(file_path, dpi=300, bbox_inches='tight') 
            
    plt.show() if display_results else plt.close(fig)
    
    return fig
