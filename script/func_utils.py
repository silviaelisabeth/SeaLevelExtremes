import logging
import os
import sys
from typing import Optional

from numpy import isnan, unique
from pandas import DataFrame, MultiIndex


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


def prepare_pooled_data(dic_data: dict, hindcast_start:int, hindcast_end: int) -> dict:
        """Calculate target years and filter to hindcast period."""
        dic_data_hindcast = {}
        for loc_id, data in dic_data.items():
            mask = (data['sim_year'] >= hindcast_start) & (data['sim_year'] <= hindcast_end)
            dic_data_hindcast[loc_id] = data[mask].copy()

        # -----------------------------------------------------------------
        ls_nmodels = [len(v.model.unique()) for v in dic_data_hindcast.values()]
        
        print(f"\nData Summary")
        print(f"  Hindcast period: {hindcast_start}-{hindcast_end}")
        print(f"  Available Models per Location: {min(ls_nmodels)}-{max(ls_nmodels)}")
        print(f"  Locations to analyse: {len(dic_data_hindcast.keys())}")
        
        return dic_data_hindcast


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


def create_data_overview(dic_data_per_model:dict,ls_files:list[str]) -> None:
    ls_num_samples = []
    dic_sim_years_per_model = dict()
    for model_label in dic_data_per_model.keys():
        years = dic_data_per_model[model_label]['valid data'].sim_year.values
        unique_years = unique(years[~isnan(years)].astype(int))
        dic_sim_years_per_model[model_label] = unique_years
        data_shape = dic_data_per_model[model_label]['valid data'].shape
        
        ls_num_samples.append(data_shape[0])
        print(f"{model_label} · {data_shape}")
        
    sites_valid = [dic_data_per_model[model_label]['preparation info'][0] for model_label in dic_data_per_model.keys()]
    unique_years_per_model = [len(dic_sim_years_per_model[model_label]) for model_label in dic_sim_years_per_model.keys()]
    sim_year_per_model_min = [min(dic_sim_years_per_model[model_label]) for model_label in dic_sim_years_per_model.keys()]
    sim_year_per_model_max = [max(dic_sim_years_per_model[model_label]) for model_label in dic_sim_years_per_model.keys()]

    print(
        "\nOverall, data is available from "
        f"\n\t{len(ls_files)} models, "
        f"\n\t{min(sites_valid)}-{max(sites_valid)} locations "
        f"(originally {dic_data_per_model[model_label]['preparation info'][1]})"
        f"\n\t{min(ls_num_samples)}-{max(ls_num_samples)} samples per model "
        f"\n\t - with {min(unique_years_per_model)}-{max(unique_years_per_model)} unique sim_years"
        f"\n\t - between {min(sim_year_per_model_min)}-{max(sim_year_per_model_max)}"
        )

