import logging
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
from numpy import isnan, unique
from pandas import DataFrame, concat, read_parquet

logger = logging.getLogger("gev_analysis")


def initialize_logger_v1(log_filename:str, log_dir:str="../logs"):
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    class PrintLogger:
        def write(self, msg):
            if msg.strip(): 
                logger.info(msg.rstrip())
        def flush(self):
            pass

    sys.stdout = PrintLogger()
    sys.stderr = PrintLogger()

    return orig_stdout, orig_stderr, fh, logger, log_path


def initialize_logger_v2(dir_logs:str, log_name:str=None)->tuple[Logger,FileHandler,str]:
    """
    Returns a simple logger that writes to a timestamped file in path_logs.
    """
    if log_name is None:
        log_name = f"LOGS_GEVAnalysis_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_path = os.path.join(dir_logs, log_name)
    logger = logging.getLogger('gev_analysis')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger, fh, log_path


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


def adding_plot_and_text(message:str, ls_messages:list[str], print_msg:bool):
    ls_messages.append(message)
    if print_msg:
        logger.info(message)
    return ls_messages


def prepare_data(data: DataFrame, hindcast_start:int, hindcast_end: int) -> DataFrame:
    """Calculate target years and filter to hindcast period."""
    data['target_year'] = data['sim_year'] + data['lead']

    mask = (data['target_year'] >= hindcast_start) & \
            (data['target_year'] <= hindcast_end)
    data_hindcast = data[mask].copy()

    logger.info(
        f"\nData Summary:"
        f"\n\tHindcast period: {hindcast_start}-{hindcast_end}"
        f"\n\tTotal observations: {len(data_hindcast):,}"
        f"\n\tModels: {data_hindcast['model'].nunique()}"
        f"\n\tLocations: {min(data_hindcast[['lon', 'lat']].nunique().values)}"
        )
    
    return data_hindcast


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


def prepare_pooled_data(dic_data: dict, hindcast_start:int, hindcast_end: int) -> tuple[dict, str]:
        """Calculate target years and filter to hindcast period."""
        dic_data_hindcast = {}
        for loc_id, data in dic_data.items():
            mask = (data['sim_year'] >= hindcast_start) & (data['sim_year'] <= hindcast_end)
            dic_data_hindcast[loc_id] = data[mask].copy()

        # -----------------------------------------------------------------
        ls_nmodels = [len(v.model.unique()) for v in dic_data_hindcast.values()]
        
        message = f"""
                \nData Summary
                \t  Hindcast period: {hindcast_start}-{hindcast_end}
                \t  Available Models per Location: {min(ls_nmodels)}-{max(ls_nmodels)}
                \t  Locations to analyse: {len(dic_data_hindcast.keys())}
                """.strip()
        logger.info(message)
        
        return dic_data_hindcast, message


def get_dataset_overview_for_model_at_location(
    dic_data: dict, model_label:Optional[str] = None, model_nr:Optional[int] = None,
    site_id: Optional[float] = None, lon:Optional[float] = None, lat:Optional[float] = None
    ) -> tuple[DataFrame, float, float]:
    if all(param is not None for param in (model_label, site_id)):
        data_for_model = dic_data[model_label]['valid data']
        data_for_model_at_location = DataFrame(
            data_for_model[:, :, site_id], 
            columns=['member0', 'member1'],
            index=data_for_model[:, :, site_id].sim_year.astype(int)
            )
        lon = data_for_model[:, :, site_id].lon.values
        lat = data_for_model[:, :, site_id].lat.values
    
    elif all(param is not None for param in (model_nr, lon, lat)):
        data_for_model = dic_data.sel(location=dict(lon=lon, lat=lat),)[model_nr]
        data_for_model_at_location = DataFrame(
            data_for_model,
            columns=['member0', 'member1'],
            index=data_for_model.sim_year.astype(int)
            )
        lon = data_for_model.lon.values
        lat = data_for_model.lat.values
    
    else:
        raise ValueError(
            f"Failed to process `data_for_model`. Missing parameters:"
            f"\n\teither provide model label and site_id: {model_label}, {site_id}, "
            f"\n\tor provide model_nr, lon, lat: {model_nr}, {lon}, {lat}"
            )
    
    logger.info(
        f"Model {model_label} ({model_nr}) - location-ID {site_id} "
        f"\nFull dataframe {data_for_model_at_location.shape} vs reduced {data_for_model_at_location.dropna().shape}"
        f"\ncoordinates in original dataset lon|lat: {lon:.5f}|{lat:.5f}")
    return data_for_model_at_location, lon, lat


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
        
    sites_valid = [dic_data_per_model[model_label]['preparation info'][0] for model_label in dic_data_per_model.keys()]
    unique_years_per_model = [len(dic_sim_years_per_model[model_label]) for model_label in dic_sim_years_per_model.keys()]
    sim_year_per_model_min = [min(dic_sim_years_per_model[model_label]) for model_label in dic_sim_years_per_model.keys()]
    sim_year_per_model_max = [max(dic_sim_years_per_model[model_label]) for model_label in dic_sim_years_per_model.keys()]

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


def save_location_results(
    location_id:int, result_location:dict, base_dir:str, plot_period_evolution:list[str], display_results:bool, 
    ) -> str:
    loc_dir = save_report_location_results(
        location_id=location_id, result_location=result_location, base_dir=base_dir
        )
    
    logger.info('export path:', loc_dir)
    _ = dbplt.plot_pooled_analysis(
        result=result_location, 
        site_id=location_id, 
        periods_evolution = plot_period_evolution, 
        save_path=loc_dir,
        box_parameters_x=0.05, box_parameters_y=0.95, width_bar_returns=0.35,
        leg_comparison_x=0.35, leg_comparison_y=0.65, linespace=1.5,
        fontsize=12, figsize=(15, 7.5),
        display_results=display_results
        )
    
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


def import_results_from_files(path_export: str) -> dict:
    results = {}
    ls_location_folders = get_all_location_folders(path_import=path_export)

    for location_files in ls_location_folders:
        location_id = int(location_files.split('location_')[-1])
        base_dir = '/'.join(location_files.split('/')[:-1])
        result_loaded = load_location_results(location_id=location_id, base_dir=base_dir)
        result_loaded['file_path_report'] = location_files
        results[location_id] = result_loaded
    
    return results


def import_results_from_files_mp(path_export: str) -> dict:
    results = {}

    ls_location_folders = get_all_location_folders(path_import=path_export)

    def load_location(location_files):
        location_id = int(location_files.split('location_')[-1])
        base_dir = '/'.join(location_files.split('/')[:-1])
        result_loaded = load_location_results(location_id=location_id, base_dir=base_dir)
        result_loaded['file_path_report'] = location_files
        return location_id, result_loaded

    parallel_results = Parallel(n_jobs=-1)(
        delayed(load_location)(lf) for lf in ls_location_folders
    )

    results = {location_id: result for location_id, result in parallel_results}
    return results


def load_pooled_results(base_dir: Union[str, Path]) -> dict[int, dict[str, any]]:
    """
    Load pooled results stored in artifact-centric format:
      - DataFrames: *.parquet (with location_id column)
      - Python objects: *.pkl (dict keyed by location_id)

    Returns:
        results: Dict[location_id -> dict of artifact_name -> artifact]
    """
    base_dir = Path(base_dir)
    results: dict[int, dict[str, any]] = {}

    # --- Load Parquet artifacts ---
    for parquet_file in base_dir.glob("*.parquet"):
        key = parquet_file.stem
        df = read_parquet(parquet_file)
        if "location_id" in df.columns:
            for loc_id, df_loc in df.groupby("location_id"):
                results.setdefault(int(loc_id), {})[key] = df_loc.drop(columns="location_id")
        else:
            results.setdefault(0, {})[key] = df

    # --- Load Pickle artifacts ---
    for pickle_file in base_dir.glob("*.pkl"):
        key = pickle_file.stem
        with open(pickle_file, "rb") as f:
            obj = pickle.load(f)
            for loc_id, value in obj.items():
                results.setdefault(int(loc_id), {})[key] = value

    return results


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

    df_all_data = concat(data, ignore_index=True)
    df_all_data.to_parquet(base_dir / "data.parquet", index=False)

