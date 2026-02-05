import logging
import os
import pickle
import re
import sys
from datetime import datetime
from glob import glob
from typing import Optional, Tuple

import arabic_reshaper
import func_gev as gev
import func_plotting as dbplt
from bidi.algorithm import get_display
from numpy import isnan, unique
from pandas import DataFrame, read_parquet


def sanitize_filename(s):
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


def initialize_logger(log_filename, log_dir:str="../logs"):
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


def adding_plot_and_text(message, ls_messages, print_msg):
    ls_messages.append(message)
    if print_msg:
        print(message)
    return ls_messages


def prepare_data(data: DataFrame, hindcast_start:int, hindcast_end: int) -> DataFrame:
    """Calculate target years and filter to hindcast period."""
    data['target_year'] = data['sim_year'] + data['lead']

    mask = (data['target_year'] >= hindcast_start) & \
            (data['target_year'] <= hindcast_end)
    data_hindcast = data[mask].copy()

    print(f"\nData Summary:")
    print(f"  Hindcast period: {hindcast_start}-{hindcast_end}")
    print(f"  Total observations: {len(data_hindcast):,}")
    print(f"  Models: {data_hindcast['model'].nunique()}")
    print(f"  Locations: {min(data_hindcast[['lon', 'lat']].nunique().values)}")
    
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


def prepare_pooled_data(dic_data: dict, hindcast_start:int, hindcast_end: int) -> Tuple[dict, str]:
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
        print(message)
        
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
    
    print(
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
        print(message)
        
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
    print(message)
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
    
    print('export path:', loc_dir)
    dbplt.plot_pooled_analysis(
        result=result_location, 
        site_id=location_id, 
        periods_evolution = plot_period_evolution, 
        save_path=loc_dir,
        box_parameters_x=0.05, box_parameters_y=0.95, width_bar_returns=0.35,
        leg_comparison_x=0.35, leg_comparison_y=0.65, linespace=1.5,
        color_markers='#99E3DDFF', colors_trends='#1D141BFF', 
        colors_models=['#B887ADFF', '#008A80FF'],
        colors_return_levels=['#008A80FF','#CAA5C2FF'],
        bbox_color='#F5F5F5FF', axes_color='#333333', 
        linestyle_trends=['dashdot', 'dashed', 'solid'], 
        fontsize=12, figsize=(15, 7.5),
        display_results=display_results
        )
    
    return loc_dir


def get_all_location_folders(path_import: str) -> list[str]:
    ls_location_folders = []

    for parent in glob(os.path.join(path_import, "*")):
        if os.path.isdir(parent):
            for loc in glob(os.path.join(parent, "*")):
                if os.path.isdir(loc):
                    ls_location_folders.append(loc)

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


def hex_to_rgba(hex_color):
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

    print(f"Full log written to {file_name}")
    print(f"Full log written to {file_name}")
    print(f"Full log written to {file_name}")
