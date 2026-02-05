import argparse
import glob
import multiprocessing as mp
import os
import sys
from datetime import datetime

import func_gev as gev
import func_plotting as dbplt
import func_preparation as dbf
import func_utils as ut
import xarray as xr
from joblib import Parallel, delayed
from pandas import DataFrame

# NOTES: for execution, run from your terminal · python analysis.py --input_dir "/path/to/netcdf/files"
# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------

path_export = '../output/gev_analysis/pooled/'
hindcast_start = 1960
hindcast_end = 2026
return_periods = [10, 25, 50, 100, 200]
plot_period_evolution = ['10-year', '50-year', '100-year']

display_results = False
export_report = True
save_regression_summary = True

colors = [
    '#53354DFF','#7D4F73FF','#B887ADFF','#CAA5C2FF','#DBC3D6FF','#F5F5F5FF','#99E3DDFF',
    '#66D4CCFF','#33C6BBFF','#008A80FF','#005C55FF'
    ]


# --------------------------------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------------------------------

def initialize_logger(log_name=None):
    """
    Returns a simple logger that writes to a timestamped file in path_export.
    """
    import logging
    if log_name is None:
        log_name = f"LOGS_GEVAnalysis_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_path = os.path.join(path_export, log_name)
    logger = logging.getLogger('gev_analysis')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger, fh, log_path


def import_all_models(ls_files):
    dic_data_per_model = dict()
    for en, file in enumerate(ls_files):
        model_name = os.path.basename(file).split('.nc')[0].split('_')[-1]
        print(f'Importing data from model {model_name} ({en+1}/{len(ls_files)})...')
        model_name, ds_model = dbf.import_data_from_file(file)
        dic_data_per_model[model_name] = {'raw data': ds_model}
    return dic_data_per_model


def prepare_combined_data(ls_files, dic_data_per_model):
    dic_data_per_model = dbf.data_preparation(ls_files=ls_files, dic_data_per_model=dic_data_per_model)
    notes_overview = ut.create_data_overview(dic_data_per_model, ls_files)
    
    # Combine per-model data into one xarray
    da_list = []
    for model_name, dic_model in dic_data_per_model.items():
        da = dic_model['valid data']
        da_loc = dbf.sites_to_location(da).expand_dims(model=[model_name])
        da_list.append(da_loc)
    combined = xr.concat(da_list, dim="model", join="outer")

    return dic_data_per_model, combined, notes_overview


def extract_location_data(combined):
    results = dbf.data_rearrangement(combined=combined, hindcast_start=hindcast_start, hindcast_end=hindcast_end)
    dic_data_per_location, df_messages = dbf.extract_location_data_and_info(results)
    return dic_data_per_location


def process_location(location_item):
    loc_id, df_prepared = location_item
    messages = []

    lon_loc = df_prepared.lon.unique()[0]
    lat_loc = df_prepared.lat.unique()[0]

    # Location info
    location_closest = dbf.add_location_labels(DataFrame([lon_loc, lat_loc], index=['lon', 'lat']).T)
    location_info = ' '.join(location_closest.values[0][2:])

    # GEV analysis
    result, ls_warnings = gev.analyze_per_location(
        df_prepared, loc_id, lat_loc, lon_loc, location_info, return_periods
    )
    if ls_warnings:
        messages.append({loc_id: ls_warnings})
    
    if result is None:
        messages.append(f"No valid GEV fit for location {loc_id}")
        return loc_id, None, messages

    # Optional report export
    export_path_site = None
    if export_report and path_export:
        export_path_site = ut.save_location_results(
            location_id=loc_id,
            result_location=result,
            base_dir=path_export,
            plot_period_evolution=plot_period_evolution,
            display_results=display_results
        )
    result['file_path_report'] = export_path_site

    return loc_id, result, messages


def run_gev_parallel(dic_data_per_location, n_jobs=None):
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)

    items = list(dic_data_per_location.items())
    out = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
        delayed(process_location)(item) for item in items
    )

    results = {}
    ls_notes = []
    for loc_id, result, messages in out:
        if messages:
            ls_notes.extend(messages)
        if result is not None:
            results[loc_id] = result
    return results, ls_notes


def run_annual_gev(results):
    results_extended, ls_notes_analysis = gev.execute_and_store_stat_gev_per_year(
        results=results, store_results=True, return_periods=return_periods
    )
    # Clean notes
    for key, outer_list in ls_notes_analysis.items():
        ls_notes_analysis[key] = [inner for inner in outer_list if inner]
    return results_extended, ls_notes_analysis


def main(ls_files):
    logger, fh, log_path = initialize_logger()
    dic_notes_analysis = {}

    dic_data_per_model = import_all_models(ls_files)
    print('Importing data done; next pooling and preparing data...')
    
    dic_data_per_model, combined, notes_overview = prepare_combined_data(ls_files, dic_data_per_model)
    dic_notes_analysis['data overview'] = notes_overview
    print('Pooling and preparing data done. Next checking locations with missing data...')
    
    missing_locations = dbf.create_summary_location_w_missing_data(
        dic_data_per_model=dic_data_per_model,
        combined=combined,
        dir_export=os.path.join(path_export, 'exploration')
    )
    dic_notes_analysis['data pooling'] = [f"{len(missing_locations)} locations without any valid data found!"]
    print(f'Found {len(missing_locations)} locations with missing data → check output folder.')

    print('Rearranging data to sort per location...')    
    dic_data_per_location = extract_location_data(combined)

    print('Rearranging done; next run GEV analysis with pooled data...')
    results, ls_notes = run_gev_parallel(dic_data_per_location)
    dic_notes_analysis['GEV pooled analysis'] = ls_notes

    print('Stationary and non-stationary GEV analysis done with pooled data; next, run annual stationary GEV...')
    results_extended, ls_notes_analysis = run_annual_gev(results)
    dic_notes_analysis['annual_statGEV'] = ls_notes_analysis

    print('All analysis done; next store output...')
    ut.store_analysis_notes(dic_notes_analysis, path_export)

    print(f"Analysis completed. Log saved at {log_path}")

    logger.removeHandler(fh)
    fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GEV analysis for multiple locations")
    
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Path to directory containing input NetCDF files (*.nc)"
    )
    
    parser.add_argument(
        "--pattern", type=str, default="*.nc",
        help="Filename pattern to match NetCDF files (default: '*.nc')"
    )
    
    args = parser.parse_args()

    ls_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not ls_files:
        raise FileNotFoundError(f"No files found in {args.input_dir} matching {args.pattern}")

    main(ls_files)
