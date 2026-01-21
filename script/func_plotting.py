from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from numpy import any, arange, linspace

rcParams['font.family'] = [
    'Noto Sans',
    'Noto Sans Arabic',
    'Noto Sans Tifinagh',
    'Noto Sans CJK JP',
    'Noto Sans Devanagari'
]
sns.set_style('whitegrid')


def plot_annual_max_with_trends(
    ax, annual_max, result, nonstat, comp, colors_trends, linestyle_trends, 
    axes_color, color_markers='#99E3DDFF', ms=6, fontsize=10
    ):
    ax.plot(
        annual_max['year'], annual_max['annual_max'], 'o', 
        color=color_markers, markersize=ms, label='Annual max'
        )

    if result['return_levels_stationary']:
        for en, period in enumerate(['10-year', '50-year', '100-year']):
            level = result['return_levels_stationary'][period]
            ax.axhline(
                y=level, linestyle=linestyle_trends[en], color=colors_trends, label=f'{period} (stationary)'
                )
    
    if nonstat and comp and comp['p_value'] < 0.05:
        print('non-stationary')
        years_plot = linspace(annual_max['year'].min(), annual_max['year'].max(), 100)
        t_plot = (years_plot - nonstat['years_mean']) / nonstat['years_std']
        mu_plot = nonstat['mu0'] + nonstat['mu1'] * t_plot
        ax.plot(years_plot, mu_plot, 'r-', linewidth=2.5, label='Non-stationary μ(t)', alpha=0.8)
    else:
        print(
            f"\t WARNING! Skipping non-stationary with model comparison {comp['p_value']:.2f} "
            f"(threshold for non-stationary 0.05)"
        )
            
    ax.set_xlabel('Year', fontsize=fontsize*0.9)
    ax.set_ylabel('Storm Surge (m)', fontsize=fontsize*0.9)
    ax.set_title(f'Annual Maximum Storm Surge (n={len(annual_max)} years)', fontsize=fontsize)
    
    leg = ax.legend(loc=0, edgecolor=axes_color, borderpad=.65, fontsize=fontsize*0.75)
    leg.get_frame().set_linewidth(.5)
    
    ax.grid(True, alpha=0.3, color='lightgrey')
    
    for spine in ax.spines.values():
            spine.set_visible(False)

    ax.axhline(y=ax.get_ylim()[0], color=axes_color, linewidth=1.2, zorder=10)
    ax.axvline(x=ax.get_xlim()[0], color=axes_color, linewidth=1.2, zorder=10)

    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)


def plot_model_comparison(
    ax, comp, stat, nonstat, colors_models, models_names=['Stationary', 'Non-Stationary'],
    axes_color: str = '#333333', bbox_color: str = '#F5F5F5FF', leg_x: float = 0.15, leg_y: float = 0.5, 
    fontsize: float = 10
    ):
    aics = [stat['aic'], nonstat['aic']]
    
    bars = ax.bar(models_names, aics, color=colors_models, alpha=0.7, edgecolor=None)
    ax.set_ylabel('AIC (lower is better)', fontsize=fontsize*0.9)
    ax.set_title(f'Model Comparison p = {comp["p_value"]:.4f}', fontsize=fontsize*0.9)
    
    for bar, aic in zip(bars, aics):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{aic:.1f}', ha='center', va='bottom')
    
    ax.text(
        leg_x, leg_y, r"$\bf{>Recommendation:}$" f"\n{comp['recommendation']}",  transform=ax.transAxes, 
        fontsize=fontsize*0.75, verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor=bbox_color, edgecolor=axes_color, linewidth=0.5, alpha=0.3),
        horizontalalignment='left'
        )
    
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.axhline(y=ax.get_ylim()[0], color=axes_color, linewidth=1.2, zorder=10)
    ax.axvline(x=ax.get_xlim()[0], color=axes_color, linewidth=1.2, zorder=10)

    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)

    ax.grid(True, alpha=0.3, color='lightgrey')


def plot_level_evolution(
        ax, result, periods: list, color_levels: list[str], width: float = 0.35, fontsize: float = 9, 
        axes_color: str = '#333333',
        ):
        levels_1960 = [result['return_levels_1960'][p] for p in periods]
        levels_2019 = [result['return_levels_2019'][p] for p in periods]
        
        x = arange(len(periods))
        
        bars_1960 = ax.bar(x - width/2, levels_1960, width, label='1960',  color=color_levels[0])
        bars_2019 = ax.bar(x + width/2, levels_2019, width, label='2019',  color=color_levels[1])
        
        for bar, level in zip(bars_1960, levels_1960):
                height = bar.get_height()
                ax.text(
                        bar.get_x() + bar.get_width()/2., height, f'{level:.2f}m', 
                        ha='center', va='bottom', fontsize=fontsize*0.85,
                        )
        
        for bar, level in zip(bars_2019, levels_2019):
                height = bar.get_height()
                ax.text(
                        bar.get_x() + bar.get_width()/2., height, f'{level:.2f}m',  
                        ha='center', va='bottom', fontsize=fontsize*0.85,
                        )

        ax.set_xlabel('Return Period', fontsize=fontsize)
        ax.set_ylabel('Return Level (m)', fontsize=fontsize)
        ax.set_title('Return Levels: 1960 vs 2019 (in case of non-stationary)', fontsize=fontsize)
        ax.set_xticks(x)
        
        period_labels = []
        for period in periods:
                # 100% / return-period ~ probability
                period_num = int(period.replace('-year', ''))
                probability = 100 / period_num 
                period_labels.append(f'{period} ({probability:.1f}%)')
        ax.set_xticklabels(period_labels, fontsize=fontsize*0.9)
                
        leg = ax.legend(loc=4, edgecolor=axes_color, borderpad=.65, fontsize=fontsize*0.75)
        leg.get_frame().set_linewidth(.5)
        
        ax.grid(True, alpha=0.3, axis='y')
        
        for spine in ax.spines.values():
                spine.set_visible(False)

        ax.axhline(y=ax.get_ylim()[0], color=axes_color, linewidth=1.2, zorder=10)
        ax.axvline(x=ax.get_xlim()[0], color=axes_color, linewidth=1.2, zorder=10)

        ax.tick_params(axis='x', colors=axes_color)
        ax.tick_params(axis='y', colors=axes_color)

        ax.grid(True, alpha=0.3, color='lightgrey')


def create_parameter_summary(
    stat, nonstat, comp, box_x, box_y, ax, fontsize: float = 9, linespace: float = 1.5,
    bbox: Optional[dict]=dict(boxstyle='round', facecolor='#F5F5F5FF', alpha=0.5)
    ):
    ax.axis('off')
    
    info_text = r"$\bf{STATIONARY\ GEV}$" "\n"
    if stat:
        info_text += f"  μ = {stat['location']:.3f}\n"
        info_text += f"  σ = {stat['scale']:.3f}\n"
        info_text += f"  ξ = {stat['shape']:.3f}\n"
        info_text += f"  Type: {stat['dist_type']}\n"
        info_text += f"  AIC = {stat['aic']:.1f}\n"
    
    if nonstat:
        info_text += "\n"r"$\bf{NON-STATIONARY\ GEV}$" "\n"
        info_text += f"  μ(t) = {nonstat['mu0']:.3f} + {nonstat['mu1']:.4f}·t\n"
        info_text += f"  σ = {nonstat['sigma']:.3f}\n"
        info_text += f"  ξ = {nonstat['xi']:.3f}\n"
        info_text += f"  AIC = {nonstat['aic']:.1f}\n"
    
    if comp:
        info_text += "\n" r"$\bf{SIGNIFICANCE\ TEST}$" "\n"
        info_text += f"  p-value = {comp['p_value']:.4f}\n"
        info_text += f"  ΔAIC = {comp['delta_aic']:.1f}\n"
    
    ax.text(box_x, box_y, info_text, transform=ax.transAxes, linespacing=linespace,
            fontsize=fontsize, verticalalignment='top', family='sans serif', bbox=bbox
            )
    

def plot_analysis(
        results: dict,
        model: str,
        lat_lon_tuple: (float, float),
        location_info: str,
        periods_evolution: list[str] = ['10-year', '50-year', '100-year'],
        save_path: str = None,
        width_bar_returns: float = 0.35,
        box_parameters_x=0.05, 
        box_parameters_y=0.95,
        color_markers='#99E3DDFF',
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
    ):
    """Create comprehensive visualization."""

    if model not in results or lat_lon_tuple not in results[model].keys():
            print(f"No results for {model}, {lat_lon_tuple}")
            return
    
    result = results[model][lat_lon_tuple]
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
        f'GEV Analysis: {model} - {normalize_location_text(location_info)}', 
        fontsize=fontsize*1.25, fontweight='bold'
        )
    
    # ----------------------------------------------------------------------------
    # Plot TOP-LEFT: Annual maxima with trends

    plot_annual_max_with_trends(
        annual_max=annual_max, result=result, nonstat=nonstat, comp=comp,
        colors_trends=colors_trends, axes_color=axes_color, color_markers=color_markers, 
        linestyle_trends=linestyle_trends, ms=6, fontsize=fontsize, ax=ax_top_left, 
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
    if result['return_levels_1960'] and result['return_levels_2019']:
        plot_level_evolution(
            result=result, periods=periods_evolution, width=width_bar_returns, 
            color_levels=colors_return_levels, fontsize=fontsize, axes_color=axes_color, 
            ax=ax_bottom_left
        )
        
    # ----------------------------------------------------------------------------
    # Plot BOTTOM-RIGHT: Model comparison
    if stat and nonstat and comp:
        plot_model_comparison(
            comp=result['model_comparison'], stat=result['gev_stationary'], 
            nonstat=result['gev_nonstationary'], models_names=['Stationary', 'Non-Stationary'], 
            colors_models=colors_models, bbox_color=bbox_color, fontsize=fontsize, 
            leg_x=leg_comparison_x, leg_y=leg_comparison_y, ax=ax_bottom_right
    )
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        time_date = datetime.today().date().isoformat()
        location = '-'.join([
            result_display['location info'].split(',')[0].strip(), 
            result_display['location info'].split(',')[-1].strip()
            ])
        file_name = f"/GEVanalysis_{model}_{location}_{time_date}.png"
        print(f"\t saving GEV analysis to {save_path} as {file_name}")

        plt.savefig(save_path+file_name, dpi=300, bbox_inches='tight')
    plt.show()