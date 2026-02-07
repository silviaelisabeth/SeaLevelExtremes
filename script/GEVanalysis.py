import argparse
import glob
import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path

import func_gev as gev
import func_plotting as dbplt
import func_preparation as dbf
import func_utils as ut
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
path_export = '../output/'
path_logs = '../logs/'
hindcast_start = 1960
hindcast_end = 2026
return_periods = [10, 25, 50, 100, 200]
plot_period_evolution = ['10-year', '50-year', '100-year']

display_results = False
export_report = True
save_regression_summary = True

_LOCATION_LABELS = None


ls_default = ['pooled', 'annual-stationary', 'regression', 'map']

#!!!ToDo regression analysis 

# --------------------------------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------------------------------
def set_location_labels(labels):
    global _LOCATION_LABELS
    _LOCATION_LABELS = labels


def process_location(location_item)->dict:
    loc_id, df_prepared = location_item
    messages = []

    lon_loc = df_prepared.lon.unique()[0]
    lat_loc = df_prepared.lat.unique()[0]

    location_info = _LOCATION_LABELS.get((round(lon_loc,6), round(lat_loc,6)), "unknown location")

    result, ls_warnings = gev.analyze_per_location(
        df_prepared, loc_id, lat_loc, lon_loc, location_info, return_periods
    )
    if ls_warnings:
        messages.append({loc_id: ls_warnings})
    
    if result is None:
        messages.append(f"No valid GEV fit for location {loc_id}")
        return loc_id, None, messages

    return {
        "loc_id": loc_id,
        "data": df_prepared,
        "result": result,
        "location_info": {"lat": lat_loc, "lon": lon_loc,"label": location_info},
        "messages": messages,
    }


def run_gev_parallel(dic_data_per_location:dict, location_labels, n_jobs=None)->tuple[dict,list]:
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)

    set_location_labels(location_labels)

    out = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
    delayed(process_location)(item) for item in list(dic_data_per_location.items())
    )
    
    all_data = []
    results = {}
    location_info = {}
    ls_notes = []

    for res in out:
        if res is None:
            continue

        loc_id = res["loc_id"]
        all_data.append(res["data"].assign(location_id=loc_id))
        results[loc_id] = res['result']
        location_info[loc_id] = res["location_info"]

        if res.get("messages"):
            ls_notes.extend(res["messages"])

    return {
        "data": all_data,
        "results": results,
        "location_info": location_info,
    }, ls_notes


def run_annual_gev(results:dict)->tuple[dict,list]:
    results_extended, ls_notes_analysis = gev.execute_and_store_stat_gev_per_year_mp(
        results=results, store_results=False, return_periods=return_periods
    )

    for key, outer_list in ls_notes_analysis.items():
        ls_notes_analysis[key] = [inner for inner in outer_list if inner]
    return results_extended, ls_notes_analysis


def run_weighted_least_square_regression(results: dict)->dict:
    results_parallel = Parallel(n_jobs=-1)(
    delayed(gev.weighted_least_square_regression_for_site_mp)(site_id, results[site_id])
        for site_id in tqdm(list(results.keys()))
    )

    for site_id, analysis_dict in results_parallel:
        results.setdefault(site_id, {}).update(analysis_dict)
        
    return results


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main(args):
    logger, fh, log_path = ut.initialize_logger_v2(dir_logs=path_logs)
    
    # potential jobs to execute 'pooled', 'annual-stationary', 'regression', 'map'
    if args.jobs is not None:
        ls_jobs = list([job.strip() for job in args.jobs.split(',')])
    else:
        ls_jobs = ls_default
    
    save_plots = args.save_plots
    
    # ---------------------------------------------------------------------------------------

    dic_notes_analysis = {}  
    if 'pooled' in ls_jobs or 'map' in ls_jobs:
        logger.info('Computing stationary and non-stationary GEV analysis with all (pooled) data...')

        ls_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
        if not ls_files:
            raise FileNotFoundError(f"No files found in {args.input_dir} matching {args.pattern}")

        dic_notes_analysis = {}
        logger.info('Importing data...')
        dic_data_per_model = dbf.import_all_models(ls_files)
        
        logger.info('Pooling and preparing data...')
        dic_data_per_model, combined, notes_overview = dbf.prepare_combined_data(ls_files, dic_data_per_model)
        dic_notes_analysis['data overview'] = notes_overview
        
        if 'map' in ls_jobs:
            logger.info(
                f'Creating map of locations with missing data with saving selected as {save_plots} '
                '(if preferred otherwise, update save_plots)...'
                )
            missing_locations = dbf.create_summary_location_w_missing_data(
                dic_data_per_model=dic_data_per_model,
                combined=combined,
                dir_export=os.path.join(path_export, 'exploration') if save_plots is True else None
            )
            dic_notes_analysis['data pooling'] = [f"{len(missing_locations)} locations without any valid data found!"]

        logger.info('Rearranging data to sort per location...')    
        dic_data_per_location = dbf.extract_location_data(combined, hindcast_start, hindcast_end)
        
        if args.start_loc is not None or args.end_loc is not None: 
            start_loc = args.start_loc if args.start_loc is not None else min(dic_data_per_location.keys()) 
            end_loc = args.end_loc if args.end_loc is not None else max(dic_data_per_location.keys())
            dic_data_per_location = ut.select_allowed_locations(
                dic_data_per_location=dic_data_per_location, 
                start_loc=start_loc,
                end_loc=end_loc
                )
            logger.info(f"Processing locations {start_loc} to {end_loc} ({len(dic_data_per_location)} total)")

        else:
            logger.info(f"Processing all {len(dic_data_per_location)} locations")
        
        logger.info('Precomputing location labels...')
        location_labels = dbf.precompute_location_labels(dic_data_per_location)

        logger.info('Run GEV analysis with pooled data...')
        output, ls_notes = run_gev_parallel(dic_data_per_location, location_labels)  
        results = output['results']  
        dic_notes_analysis['GEV pooled analysis'] = ls_notes
        
        logger.info('Saving data per artifacts...')   
        today_ = str(datetime.today().date().isoformat())   
        path_child_folder = Path(path_export) / "gev_analysis" / "pooled" / f"{today_}"
        ut.save_pooled_results(results=results, data=output['data'], base_dir=path_child_folder)
        
        fig_dir = Path(path_child_folder) / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        if save_plots is True:
            logger.info(f'Preparing to save figures per location...')
            for loc_id, result in results.items():
                fig = dbplt.plot_pooled_analysis(
                    result=result, 
                    site_id=loc_id, 
                    periods_evolution=plot_period_evolution, 
                    leg_comparison_x=0.35, leg_comparison_y=0.65, 
                    linestyle_trends=['dashdot', 'dashed', 'solid'], 
                    fontsize=12, figsize=(15, 7.5),
                    )
                
                lat = str(result['location info']['lat'].round(3))
                lon = str(result['location info']['lon'].round(3))
                country = result['location info']['description'].split(',')[-1].strip()
        
                fig.savefig(
                    fig_dir / f"location_{loc_id}_{country}_{lat}_{lon}_pooledGEVanalysis.png", 
                    dpi=150, bbox_inches="tight"
                    )
                plt.close(fig)
        logger.info('Data (and Figures) successfully saved per artifacts.')        

    if 'annual-stationary' in ls_jobs:
        logger.info('Computing stationary GEV analysis per year...')

        try:
            results.keys()
            logger.info('✓ continue with available dictionary')
            
        except NameError:
            path_import = os.path.join(args.input_dir)
            logger.info(f'Import data from folder {path_import}...')
            results = ut.load_pooled_results(path_import)
            logger.info(f'Imported {len(results)} locations')

        if args.start_loc is not None or args.end_loc is not None: 
            start_loc = args.start_loc if args.start_loc is not None else min(dic_data_per_location.keys()) 
            end_loc = args.end_loc if args.end_loc is not None else max(dic_data_per_location.keys())
            results = ut.select_allowed_locations(
                dic_data_per_location=results, 
                start_loc=start_loc,
                end_loc=end_loc
                )
            logger.info(f"Processing locations {start_loc} to {end_loc} ({len(results)} total)")
        
        else:
            logger.info(f"Processing all {len(results)} locations")
        
        logger.info(f'Run annual stationary GEV analysis...')
        results_extended, ls_notes_analysis = run_annual_gev(results)
        dic_notes_analysis['annual_statGEV'] = ls_notes_analysis
        
        ut.save_annual_stationary_results(
            results={
                site_id: {"fit results": site_data["fit results"]}
                for site_id, site_data in results_extended.items()
                if "fit results" in site_data
                },
            base_dir=path_import
        )
        logger.info(f'Results successfully added/stored in fit results.pkl {path_import}.')

    if 'regression' in ls_jobs:
        logger.info(
            'Compute regression for location parameter using non-stationary and annual stationary GEV approach...'
            )
        
        try:
            results.keys()
            logger.info('✓ continue with available dictionary')
            
        except NameError:
            path_import = os.path.join(args.input_dir)
            logger.info(
                f'Import fit results for annual stationary and non-stationary GEV analysis from folder {path_import}...'
                )
            results = ut.load_fit_results(path_import)
    
        if args.start_loc is not None or args.end_loc is not None: 
            start_loc = args.start_loc if args.start_loc is not None else min(dic_data_per_location.keys()) 
            end_loc = args.end_loc if args.end_loc is not None else max(dic_data_per_location.keys())
            results = ut.select_allowed_locations(
                dic_data_per_location=results, 
                start_loc=start_loc,
                end_loc=end_loc
                )
            logger.info(f"Processing locations {start_loc} to {end_loc} ({len(results)} total)")

        else:
            logger.info(f"Processing all {len(results)} locations")
        
        results = run_weighted_least_square_regression(results)
        
        logger.info("WLS Regression for done; now plotting regression (and saving)...")
        ut.plot_and_save_regression_analysis(results=results, path_export=path_import, save_output=save_plots)


    logger.info('All analyses done; next store output...')
    ut.store_analysis_notes(dic_notes_analysis, path_export + 'gev_analysis/pooled/')

    # ---------------------------------------------------------------------------------------
    logger.info(f"Analysis completed. Log saved at {log_path}")

    logger.removeHandler(fh)
    fh.close()


if __name__ == "__main__":
    # NOTES: for execution, run from your terminal · python3 GEVanalysis.py --input_dir "/path/to/netcdf/files"
    parser = argparse.ArgumentParser(description="Run GEV analysis for multiple locations")
    
    parser.add_argument(
        "-input_dir", type=str, required=True,
        help="Path to directory containing input files."
    )
    
    parser.add_argument(
        "--pattern", type=str, default="*.nc",
        help="Filename pattern to match NetCDF files (default: '*.nc')"
    )
    
    parser.add_argument(
        "--jobs", type=str, default=None,
        help="List of jobs to execute; available jobs are: pooled, annual-stationary, regression, map"
    )
    
    parser.add_argument(
        "-start_loc", type=int, default=None,
        help="Start location ID (inclusive) to process. Default is None."
    )
    
    parser.add_argument(
        "-end_loc", type=int, default=None,
        help="End location ID (inclusive) to process. Default is None."
    )
    
    parser.add_argument(
        "-save_plots", type=bool, default=True,
        help="Boolean whether to save figures/panels in output folder or not (time-consuming!). Default is True"
    )    
    
    args = parser.parse_args()

    main(args)
    main(args)
    main(args)
    main(args)
    main(args)
