import logging
import multiprocessing as mp
import os
import pickle
import re
import sys
from datetime import datetime
from glob import glob
from logging import FileHandler, Logger
from pathlib import Path
from typing import Optional, Union

import arabic_reshaper
import func_plotting as dbplt
from bidi.algorithm import get_display
from joblib import Parallel, delayed
from numpy import abs, isnan, median, unique
from pandas import DataFrame, concat, read_parquet
from scipy.stats import norm

logger = logging.getLogger("mp_gev_analysis")
FACTORMTOMM = 1000


def setup_main_logging(dir_log_file:Optional=None, logger_name="mp_gev_analysis"):
    """
    Call this ONCE in the main process.
    Creates queue + listener + file handler.
    """ 
    if dir_log_file is None:
        log_file = f"../logs/LOGS_GEVAnalysis_{datetime.now():%Y%m%d_%H%M%S}.log"
    else:
        log_file = dir_log_file
        
    log_queue = mp.Manager().Queue()

    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        "%(asctime)s - %(processName)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    listener = logging.handlers.QueueListener(log_queue, file_handler)
    listener.start()

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.handlers.QueueHandler(log_queue))
    logger.propagate = False

    return logger, log_queue, listener, log_file


def worker_init(log_queue):
    import logging
    import logging.handlers

    logger = logging.getLogger("mp_gev_analysis")
    logger.setLevel(logging.INFO)

    logger.handlers = []

    qh = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(qh)
    logger.propagate = False


def sanitize_filename(s:str)->str:
    return re.sub(r'[<>:"/\\|?*]', '_', s)


def contains_arabic(text: str) -> bool:
    return any(
        '\u0600' <= ch <= '\u06FF' or
        '\u0750' <= ch <= '\u077F' or
        '\u08A0' <= ch <= '\u08FF'
        for ch in text
    )


def normalize_location_text(text: str) -> str:
    """
    Automatically reshape Arabic text if present.
    Leaves all other scripts untouched.
    """
    if contains_arabic(text):
        return get_display(arabic_reshaper.reshape(text))
    return text


def define_filename_per_parameter(parameter, approach, mark_outlier):    
    if parameter == 'mu1':
        file_name = 'Map_μ₁_inclCI_markedOutlier' if mark_outlier is True else 'Map_μ₁_inclCI_all'
        
    elif parameter == 'mu': 
        file_name = 'Map_μ_inclCI_markedOutlier' if mark_outlier is True else 'Map_μ_inclCI_all'

    elif parameter == 'scale': 
        file_name = 'Map_scale_inclCI_markedOutlier' if mark_outlier is True else 'Map_scale_inclCI_all'

    elif parameter == 'shape': 
        file_name = f'Map_shape_inclCI_markedOutlier' if mark_outlier is True else 'Map_shape_inclCI_all'
    
    else:
        raise ValueError('Could not identify parameter...')
    
    file_name +=f'_{approach}'
    return file_name


def define_title_and_label_per_parameter(parameter, unit, approach):
    if parameter == 'mu1':
        title = f"Map of location parameter $μ_1$ · {approach}"
        label_colormap = f"Location trend $μ_1$, {unit}"
    
    elif parameter == 'mu':
        title = f"Map of location parameter $μ$ · {approach}"
        label_colormap = f"Location $μ$, {unit}"
    
    elif parameter == 'scale':
        title = f"Map of scale parameter $σ$ · {approach}"
        label_colormap = f"Scale $σ$, {unit}"
    
    elif parameter == 'shape':
        title = f"Map of shape parameter $ξ$ · {approach}"
        label_colormap = f"Shape trend $ξ$, {unit}"
    
    return title, label_colormap


def prepare_pooled_data_per_location(
    loc_ex: int, data: DataFrame, hindcast_start:int, hindcast_end: int
    ) -> tuple[dict, str]:
    """Calculate target years and filter to hindcast period."""

    mask = (data['sim_year'] >= hindcast_start) & (data['sim_year'] <= hindcast_end)
    data_hindcast = data[mask].copy()

    # -----------------------------------------------------------------

    message = f"""
            \nData Summary siteID {loc_ex}
            \t  Hindcast period: {hindcast_start}-{hindcast_end}
            \t  Available Models at Location: {len(data_hindcast.model.unique())}
            \t  Observations for analyse: {len(data_hindcast)}
            """.strip()

    return data_hindcast, message


def create_data_overview(dic_data_per_model:dict,ls_files:list[str]) -> list:
    ls_notes = []

    ls_num_samples = []
    dic_sim_years_per_model = dict()
    for model_label in dic_data_per_model.keys():
        years = dic_data_per_model[model_label]['valid data'].sim_year.values
        unique_years = unique(years[~isnan(years)].astype(int))
        dic_sim_years_per_model[model_label] = unique_years
        data_shape = dic_data_per_model[model_label]['valid data'].shape

        ls_num_samples.append(data_shape[0])
        message = f"{model_label} · {data_shape}"
        ls_notes.append(message)
        logger.info(message)

    sites_valid = [
        dic_data_per_model[model_label]['preparation info'][0] 
        for model_label in dic_data_per_model.keys()
        ]
    unique_years_per_model = [
        len(dic_sim_years_per_model[model_label]) for model_label in dic_sim_years_per_model.keys()
        ]
    sim_year_per_model_min = [
        min(dic_sim_years_per_model[model_label]) for model_label in dic_sim_years_per_model.keys()
        ]
    sim_year_per_model_max = [
        max(dic_sim_years_per_model[model_label]) for model_label in dic_sim_years_per_model.keys()
        ]

    message = f"""
        Overall, data is available from
        \t{len(ls_files)} models,
        \t{min(sites_valid)}-{max(sites_valid)} locations (originally {dic_data_per_model[model_label]['preparation info'][1]})
        \t{min(ls_num_samples)}-{max(ls_num_samples)} samples per model
        \t - with {min(unique_years_per_model)}-{max(unique_years_per_model)} unique sim_years
        \t - between {min(sim_year_per_model_min)}-{max(sim_year_per_model_max)}
        """.strip()
    ls_notes.append(message)
    logger.info(message)
    return ls_notes


def save_report_location_results(location_id:int, result_location:dict, base_dir:str) -> str:
    today_ = str(datetime.today().date().isoformat())
    
    loc_dir = os.path.join(base_dir + str(today_), f"location_{location_id}")
    os.makedirs(loc_dir, exist_ok=True)
    
    for k, v in result_location.items():
        if isinstance(v, DataFrame):
            v.to_parquet(os.path.join(loc_dir, f"{k}.parquet"))
        else:
            with open(os.path.join(loc_dir, f"{k}.pkl"), "wb") as f:
                pickle.dump(v, f, protocol=5)
    return loc_dir


def get_all_location_folders(path_import: str) -> list[str]:
    ls_location_folders = [
        os.path.abspath(f) for f in glob(os.path.join(path_import, "**", "location_*"), recursive=True)
        if os.path.isdir(f)
    ]
    return ls_location_folders


def load_location_results(location_id: int, base_dir: str) -> dict:
    loc_dir = os.path.join(base_dir, f"location_{location_id}")
    results = {}
    for f in os.listdir(loc_dir):
        path = os.path.join(loc_dir, f)
        if f.endswith(".parquet"):
            results[f.replace(".parquet","")] = read_parquet(path)
        elif f.endswith(".pkl"):
            with open(path, "rb") as fd:
                results[f.replace(".pkl","")] = pickle.load(fd)
    return results


def select_allowed_locations(dic_data_per_location, start_loc, end_loc):
    dic_data_per_location = {
        loc_id: df
        for loc_id, df in dic_data_per_location.items()
        if start_loc <= loc_id <= end_loc
    }
    return dic_data_per_location


def hex_to_rgba(hex_color:str)->list[int]:
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    a = int(hex_color[6:8], 16) if len(hex_color) == 8 else 255
    return [r, g, b, a]


def store_analysis_notes(dic_notes_analysis: dict, path_export: str) -> None:
    """
    Save the full log dictionary to a timestamped text file.

    Handles nested dicts and lists, including lists of lists.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    file_name = f"{path_export}{timestamp}_analysisNotes.txt"

    def write_item(f, item, indent=0):
        """Recursively write an item to file, handling nested lists."""
        prefix = "  " * indent + "- "
        if isinstance(item, list):
            for subitem in item:
                write_item(f, subitem, indent=indent)
        else:
            f.write(f"{prefix}{item}\n")

    with open(file_name, "w", encoding="utf-8") as f:
        for section, content in dic_notes_analysis.items():
            f.write(f"=== Section: {section} ===\n")
            
            if isinstance(content, dict):
                for site_id, messages in content.items():
                    f.write(f"SiteID {site_id}:\n")
                    write_item(f, messages, indent=1)
                    f.write("\n")
            elif isinstance(content, list):
                write_item(f, content)
                f.write("\n")
            else:
                f.write(f"{content}\n\n")

    logger.info(f"Full log written to {file_name}")


def save_pooled_results(results: dict[int, dict], data, base_dir: str) -> None:
    """
    results: dict[location_id -> result_location dict]
    """

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    parquet_buffers = {}   
    pickle_buffers = {}   

    for loc_id, result in results.items():
        for key, value in result.items():

            if isinstance(value, DataFrame):
                df = value.copy()
                df["location_id"] = loc_id
                parquet_buffers.setdefault(key, []).append(df)

            else:
                pickle_buffers.setdefault(key, {})[loc_id] = value

    for key, dfs in parquet_buffers.items():
        df_all = concat(dfs, ignore_index=True)

        df_all.to_parquet(base_dir / f"{key}.parquet", index=False)
    
    for key, obj in pickle_buffers.items():
        with open(base_dir / f"{key}.pkl", "wb") as f:
            pickle.dump(obj, f, protocol=5)

    if data: 
        df_all_data = concat(data, ignore_index=True)
        df_all_data.to_parquet(base_dir / "data.parquet", index=False)


def deep_merge_dicts(d_old: dict, d_new: dict) -> dict:
    """Recursively merge d_new into d_old."""
    for k, v in d_new.items():
        if k in d_old and isinstance(d_old[k], dict) and isinstance(v, dict):
            d_old[k] = deep_merge_dicts(d_old[k], v)
        else:
            d_old[k] = v
    return d_old


def store_annual_stat_results(results_annual_stat, path_child_folder):
    os.makedirs(path_child_folder, exist_ok=True)

    filename = path_child_folder / 'stationary_per_year.pkl'
    with open(filename, "wb") as f:
        pickle.dump(results_annual_stat, f)

    print(f"Output stored in {filename}")
    

def store_location_regression(fig, loc_id, LatLon, location_info, path_child_folder):
    os.makedirs(path_child_folder/'figures', exist_ok=True)

    lat = str(LatLon[0].round(3))
    lon = str(LatLon[1].round(3))
    country = location_info.split(' ')[-1].strip()
        
    filename = path_child_folder / f'figures/location_{loc_id}_{country}_{lat}_{lon}_TrendAnalysisLocationParameter.png'
    fig.savefig(filename, dpi=150, bbox_inches="tight")

    print(f"Regression analysis for location parameter stored as {filename}")


def extract_scale_and_shape_for_all_sites_ns(results):
    # Note how the parameters are stored: 'location', 'location_trend', 'scale', 'shape'
    
    dic_para = dict()
    for loc_id in results.keys():
        try:
            _, _, scale, shape = results[loc_id]['params_hat']
            _, _, scale_std, shape_std = results[loc_id]['params_std']
            dic_para[loc_id] = (scale, scale_std, shape, shape_std)
        except:
            dic_para[loc_id] = (None, None, None, None)

    df = DataFrame(dic_para, index=['scale_mm', 'scale_std_mm', 'shape', 'shape_std']).T
    df['scale_mm'] = df.scale_mm*FACTORMTOMM
    df['scale_std_mm'] = df.scale_std_mm*FACTORMTOMM
    return df


def extract_mu_for_all_sites_ns(results):
    # Note how the parameters are stored: 'location', 'location_trend', 'scale', 'shape'
    
    dic_mu = dict()
    for loc_id in results.keys():
        try:
            mu0, mu1, _, _ = results[loc_id]['params_hat']
            mu0_std, mu1_std, _, _ = results[loc_id]['params_std']
            dic_mu[loc_id] = (mu0, mu0_std, mu1, mu1_std)
        except:
            dic_mu[loc_id] = (None, None, None, None)
        
    return DataFrame(dic_mu, index=['mu_mm', 'mu_std_mm', 'mu1_mm/yr', 'mu1_std_mm/yr']).T*FACTORMTOMM


def extract_scale_and_shape_for_all_sites_astat(results):
    # Note the parameters and STD are calculated as average of all years analyzed

    dic_para = dict()
    for loc_id in results.keys():
        try:
            dic_para[loc_id] = (
                results[loc_id]['annual_mle']['scale'].mean(), results[loc_id]['annual_mle']['scale'].std(),
                results[loc_id]['annual_mle']['shape'].mean(), results[loc_id]['annual_mle']['shape'].std()
                )
        except:
            dic_para[loc_id] = (None, None, None, None)
    
    df = DataFrame(dic_para, index=['scale_mm', 'scale_std_mm', 'shape', 'shape_std']).T
    df['scale_mm'] = df.scale_mm*FACTORMTOMM
    df['scale_std_mm'] = df.scale_std_mm*FACTORMTOMM
    return df


def extract_mu_for_all_sites_astat(results):
    dic_mu = dict()
    for loc_id in results.keys():
        try:
            dic_mu[loc_id] = (
                results[loc_id]['mu_trend']['mu0'], results[loc_id]['mu_trend']['mu0_se'], 
                results[loc_id]['mu_trend']['mu1'], results[loc_id]['mu_trend']['mu1_se']
                )
        except:
            dic_mu[loc_id] = (None, None, None, None)
        
    return DataFrame(dic_mu, index=['mu_mm', 'mu_std_mm', 'mu1_mm/yr', 'mu1_std_mm/yr']).T*FACTORMTOMM


def extract_scale_and_shape_for_all_sites_stat(results):
    dic_para = dict()
    for loc_id in results.keys():
        try:
            dic_para[loc_id] = (
                results[loc_id]['scale'], results[loc_id]['scale_std'], 
                results[loc_id]['shape'], results[loc_id]['shape_std']
                )
        except:
            dic_para[loc_id] = (None, None, None, None)
    
    df = DataFrame(dic_para, index=['scale_mm', 'scale_std_mm', 'shape', 'shape_std']).T
    df['scale_mm'] = df.scale_mm*FACTORMTOMM
    df['scale_std_mm'] = df.scale_std_mm*FACTORMTOMM
    return df


def extract_mu_for_all_sites_stat(results):
    dic_mu = dict()
    for loc_id in results.keys():
        try:
            dic_mu[loc_id] = (results[loc_id]['location'], results[loc_id]['location_std'])
        except:
            dic_mu[loc_id] = (None, None)
        
    return DataFrame(dic_mu, index=['mu_mm', 'mu_std_mm']).T*FACTORMTOMM


def mark_outliers_zmethod(df_plot, label_col, threshold=3):
    mean = df_plot[label_col].mean()
    std = df_plot[label_col].std()
    df_plot['outliers'] = abs(df_plot[label_col] - mean) > threshold*std  # 3σ

    print(f"Marked {df_plot['outliers'].sum()} outliers using modified Z-score method")
    return df_plot


def get_confidence_intervals_from_parameter(df_para, parameter, confidence_level_pc):
    z_ci = norm.ppf(1 - (1-confidence_level_pc)/2) 
    
    if parameter == 'mu':
        col_para = 'mu_mm'
        col_std = 'mu_std_mm'
        columns_label = ['mu_upper', 'mu_lower']
    
    elif parameter == 'mu1':
        col_para = 'mu1_mm/yr'
        col_std = 'mu1_std_mm/yr'
        columns_label = ['mu1_upper', 'mu1_lower']
    
    elif parameter == 'scale':
        col_para = 'scale_mm'
        col_std = 'scale_std_mm'
        columns_label = ['scale_upper', 'scale_lower']
            
    elif parameter == 'shape':
        col_para = 'shape'
        col_std = 'shape_std'
        columns_label = ['shape_upper', 'shape_lower']
    
    else:
        raise ValueError(f'Could not identify valid parameter {parameter}, skipping...')
    
    return DataFrame([
        df_para[col_para] - z_ci * df_para[col_std],
        df_para[col_para] + z_ci * df_para[col_std]], index=columns_label
    ).T, col_para


def import_info_for_regression(dir_import: str) -> tuple[dict(), dict(), dict(), dict()]:
    
    file_annual_stat = dir_import + '/stationary_per_year.pkl'
    if not os.path.isfile(file_annual_stat):
        print(f'{file_annual_stat} does not exist in folder; skipping data import... ')
        results_annual_stat_all = {}
    else:
        print(f'loading {file_annual_stat}')
        with open(file_annual_stat, "rb") as f:
            results_annual_stat_all = pickle.load(f)
    
    file_nonstat = dir_import + '/nonstationary.pkl'
    if not os.path.isfile(file_nonstat):
        print(f'{file_nonstat} does not exist in folder; skipping data import... ')
        results_nonstat_all = {}
    else:
        print(f'loading {file_nonstat}')
        with open(file_nonstat, "rb") as f:
            results_nonstat_all = pickle.load(f)

    file_loc_geo_info = dir_import + '/LatLon.pkl'
    if not os.path.isfile(file_loc_geo_info):
        print(f'{file_loc_geo_info} does not exist in folder; skipping data import... ')
        location_geo_info = {}
    else:
        print(f'loading {file_loc_geo_info}')
        with open(file_loc_geo_info, "rb") as f:
            location_geo_info = pickle.load(f)

    file_loc_info = dir_import + '/location_info.pkl'
    if not os.path.isfile(file_loc_geo_info):
        print(f'{file_loc_info} does not exist in folder; skipping data import... ')
        location_point_info = {}
    else:
        print(f'loading {file_loc_info}')
        with open(file_loc_info, "rb") as f:
            location_point_info = pickle.load(f)
        
    return results_annual_stat_all, results_nonstat_all, location_geo_info, location_point_info


def import_pickle_data(dir_import, pkl_file):
    file_full_path = dir_import + '/' + pkl_file
    if not os.path.isfile(file_full_path):
        print(f'{file_full_path} does not exist in folder; skipping data import... ')
        results_import_all = {}
    else:
        print(f'loading {file_full_path}')
        with open(file_full_path, "rb") as f:
            results_import_all = pickle.load(f)
    return results_import_all


def remove_nan_sites(df_plot):
    ls_nan_loc = []
    for en, a in enumerate(df_plot.alpha):
        if isnan(a):
            ls_nan_loc.append(en)
    print(f'{len(ls_nan_loc)} missing location information')

    return df_plot.dropna()
