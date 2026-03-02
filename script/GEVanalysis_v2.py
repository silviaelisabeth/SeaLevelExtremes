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
PATH_EXPORT = '../output/'
PATH_LOGS = '../logs/'
HINDCAST_START = 1960
HINDCAST_END = 2026
RETURN_PERIODS = [10, 25, 50, 100, 200]
PLOT_PERIOD_EVOLUTION = ['10-year', '50-year', '100-year']

DISPLAY_RESULTS = False
EXPORT_REPORT = True
SAVE_REGRESSION_SUMMARY = True
CHUNK_SIZE = 500
LS_T_EVAL = 2026, 2050
UNCERTAINTY = 'delta'
B = 300 
SEED = None

_LOCATION_LABELS = None


ls_default = ['pooled', 'annual-stationary', 'regression'] # additional 'map'

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
        df_prepared, loc_id, lat_loc, lon_loc, location_info, RETURN_PERIODS
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

    with Parallel(n_jobs=-1) as parallel:
        out = parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
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
        results=results, store_results=False, return_periods=RETURN_PERIODS
    )

    for key, outer_list in ls_notes_analysis.items():
        ls_notes_analysis[key] = [inner for inner in outer_list if inner]
    return results_extended, ls_notes_analysis


def run_weighted_least_square_regression(results: dict)->dict:
    with Parallel(n_jobs=-1) as parallel:
        results_parallel = parallel(n_jobs=-1)(
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

    logger, fh, log_path = ut.initialize_logger_v2(dir_logs=PATH_LOGS)

    # potential jobs to execute 'map', 'pooled', 'annual-stationary', 'regression'
    if args.jobs is not None:
        ls_jobs = list([job.strip() for job in args.jobs.split(',')])
    else:
        ls_jobs = ls_default
    logger.info('Processing the following jobs %s', ls_jobs)

    save_plots = args.save_plots
    save_results = args.save_results

    # ---------------------------------------------------------------------------------------
    dic_notes_analysis = {}

    if 'map' in ls_jobs:
        logger.info(
            'Computing stationary and non-stationary GEV analysis with all (pooled) data...'
            )

        ls_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
        if not ls_files:
            raise FileNotFoundError(f"No files found in {args.input_dir} matching {args.pattern}")

        dic_notes_analysis = {}
        logger.info('Importing Data...')
        dic_data_per_model = dbf.import_all_models(ls_files)

        logger.info('Pooling and Preparing Data...')
        dic_data_per_model, combined, notes_overview = dbf.prepare_combined_data(ls_files, dic_data_per_model)
        dic_notes_analysis['data overview'] = notes_overview
        
        logger.info('Rearranging Data – sorting per location...')
        dic_data_per_location = dbf.extract_location_data(combined, HINDCAST_START, HINDCAST_END)
        
        logger.info(
            'Creating map of locations with missing data with saving selected as %s '
            '(if preferred otherwise, update save_plots)...',
            save_plots
        )
        n_obs_per_location = dbf.get_number_of_observations_per_site(dic_data_per_location)

        missing_locations = dbf.create_summary_location_w_missing_data(
            dic_data_per_model=dic_data_per_model,
            combined=combined, n_obs_per_location=n_obs_per_location, 
            dir_export=os.path.join(PATH_EXPORT, 'exploration') if save_plots is True else None
        )
        dic_notes_analysis['data pooling'] = [f"{len(missing_locations)} locations without any valid data found!"]
    
    if 'pooled' in ls_jobs:
        logger.info('Computing stationary and non-stationary GEV analysis with all (pooled) data...')

        try:
            dic_data_per_model.keys()
            logger.info('✓ continue with available dictionary')      
        except NameError:
            ls_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
            if not ls_files:
                raise FileNotFoundError(f"No files found in {args.input_dir} matching {args.pattern}")

            dic_notes_analysis = {}
            logger.info('Importing data from folder %s...', args.input_dir)
            dic_data_per_model = dbf.import_all_models(ls_files)
        
            logger.info('Pooling and Preparing Data...')
            dic_data_per_model, combined, notes_overview = dbf.prepare_combined_data(ls_files, dic_data_per_model)
            dic_notes_analysis['data overview'] = notes_overview
        
        logger.info('Rearranging Data – sorting per location...')    
        dic_data_per_location = dbf.extract_location_data(combined, HINDCAST_START, HINDCAST_END)
        
        if args.start_loc is not None or args.end_loc is not None: 
            start_loc = args.start_loc if args.start_loc is not None else min(dic_data_per_location.keys()) 
            end_loc = args.end_loc if args.end_loc is not None else max(dic_data_per_location.keys())
            dic_data_per_location = ut.select_allowed_locations(
                dic_data_per_location=dic_data_per_location, 
                start_loc=start_loc,
                end_loc=end_loc
                )
            logger.info('Processing locations %s to %s (%s total)', start_loc, end_loc, len(dic_data_per_location))
        else:
            logger.info('Processing all %s locations', len(dic_data_per_location))
        
        logger.info(
            'Getting closest point available as location label for orientation. '
            '\nNote this is not the exact location...'
            )
        location_labels = dbf.precompute_location_labels(dic_data_per_location)
        
        logger.info('Run GEV analysis with pooled data...')
        
        location_ids = list(dic_data_per_location.keys())
        parallel_output = Parallel(n_jobs=-1,backend="loky")(
            delayed(gev._pooled_gev_per_single_location)(
                loc_id=loc_id,
                dic_data_per_location=dic_data_per_location,
                return_periods=RETURN_PERIODS,
                ls_t_eval=LS_T_EVAL,
                location_labels=location_labels
            )
            for loc_id in location_ids
        )
        results_all = dict(parallel_output)

        logger.info('Saving data per artifacts...')   
        today_ = str(datetime.today().date().isoformat())   
        path_child_folder = Path(PATH_EXPORT) / "gev_analysis" / f"{today_}"
        if save_results:
            logger.info('Saving fit results per location...')
            ut.save_pooled_results(results=results_all, data=dic_data_per_location, base_dir=path_child_folder)
        else:
            logger.info('Skip saving fit results...')
            
        fig_dir = Path(path_child_folder) / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        if save_plots is True:
            logger.info('Preparing to save figures per location...')
            
            for loc_id, result in results_all.items():
                fig = dbplt.plot_pooled_analysis_v2(
                    result=result, site_id=loc_id, return_periods=RETURN_PERIODS, 
                    plot_t_eval=[LS_T_EVAL[0]], plot_evolution=[int(i.split('-')[0]) for i in PLOT_PERIOD_EVOLUTION],
                    leg_comparison_x=0.075, leg_comparison_y=0.35, box_parameters_x=0.45,  box_parameters_y= 0.95,
                    linestyle_trends = ['-', '--', '-.', ':', (0, (1, 1)), (0, (5, 10))], 
                    fontsize=12, figsize=(15, 7.5), display_results=DISPLAY_RESULTS
                    )

                lat = str(result['LatLon'][0].round(3))
                lon = str(result['LatLon'][1].round(3))
                country = result['location_info'].split(' ')[-1].strip()
        
                fig.savefig(
                    fig_dir / f"location_{loc_id}_{country}_{lat}_{lon}_pooledGEVanalysis.png", 
                    dpi=150, bbox_inches="tight"
                    )
                plt.close(fig)
        logger.info('Data (and Figures) successfully saved per artifacts.')        

    if 'annual-stationary' in ls_jobs:
        logger.info('Computing stationary GEV analysis per year...')

        try:
            dic_data_per_location.keys()
            logger.info('✓ continue with available dictionary')  
        except NameError:
            ls_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
            if not ls_files:
                raise FileNotFoundError(f"No files found in {args.input_dir} matching {args.pattern}")

            logger.info('Import data from folder %s...', os.path.join(args.input_dir))
            dic_data_per_model = dbf.import_all_models(ls_files)

            logger.info('Pooling and Preparing Data...')
            dic_data_per_model, combined, notes_overview = dbf.prepare_combined_data(ls_files, dic_data_per_model)
            dic_notes_analysis['data overview'] = notes_overview
            
            logger.info('Rearranging Data – sorting per location...')
            dic_data_per_location = dbf.extract_location_data(combined, HINDCAST_START, HINDCAST_END)
            
            if args.start_loc is not None or args.end_loc is not None: 
                start_loc = args.start_loc if args.start_loc is not None else min(dic_data_per_location.keys()) 
                end_loc = args.end_loc if args.end_loc is not None else max(dic_data_per_location.keys())
                dic_data_per_location = ut.select_allowed_locations(
                    dic_data_per_location=dic_data_per_location, 
                    start_loc=start_loc,
                    end_loc=end_loc
                    )
                logger.info('Processing locations %s to %s (%s total)', start_loc, end_loc, len(dic_data_per_location))
            else:
                logger.info('Processing all %s locations', len(dic_data_per_location))
        
        logger.info('Run annual stationary GEV analysis...')
        results_extended = gev.fit_all_locations(dic_data_per_location, n_jobs=-1)
        
        if save_results:
            try:
                dir_export = path_child_folder
            except:
                today_ = str(datetime.today().date().isoformat())   
                path_child_folder = Path(PATH_EXPORT) / "gev_analysis" / f"{today_}"
            ut.store_annual_stat_results(results_extended, path_child_folder)
            
            logger.info('Results successfully added/stored in fit results.pkl %s.', path_child_folder)
        else:
            logger.info('Skipping to save fit results...')
            
    if 'regression' in ls_jobs:
        logger.info(
            'Compute regression for location parameter using non-stationary and annual stationary GEV approach...'
            )
        
        logger.info('Import relevant data from folder %s', args.input_dir)
        [
            results_annual_stat_all, results_nonstat_all, location_geo_info, location_point_info
        ] = ut.import_info_for_regression(args.input_dir)
    
    
        for loc_id in results_annual_stat_all.keys():
            years_, dic_trend = gev.prepare_for_regression(
                annual_stationary=results_annual_stat_all[loc_id], 
                nonstationary=results_nonstat_all[loc_id],
                years_mean=results_nonstat_all[loc_id]['years_mean'], 
                years_std=results_nonstat_all[loc_id]['years_std'], 
                hindcast_start=HINDCAST_START, hindcast_end=HINDCAST_END,
                z_percentile=1.96, factor_m_to_mm=1000
                ) 

            fig_reg = dbplt.plot_location_regression(
                loc_id, years_, dic_trend, results_annual_stat_all[loc_id], fontsize=12,
                axes_color='#333333', markers_color="#99E3DDFF", colors_reg=['#CAA5C2FF',  '#005C55FF'], 
                )
            
            try:
                dir_export = path_child_folder
            except:
                today_ = str(datetime.today().date().isoformat())   
                dir_export = Path(PATH_EXPORT) / 'gev_analysis'/ f"{today_}"

            ut.store_location_regression(
                fig_reg,loc_id, location_geo_info[loc_id], location_point_info[loc_id], dir_export
                )

    logger.info('All analyses done; next store output...')
    ut.store_analysis_notes(dic_notes_analysis, PATH_LOGS)

    # ---------------------------------------------------------------------------------------
    logger.info('Analysis completed. Log saved at %s', log_path)

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
        "-save_results", action='store_true', help="Save fit results (default False)"
    )
    parser.add_argument(
        "-no-save_results", dest='save_results', action='store_false', help="Do not save fit results"
    )
    parser.set_defaults(save_results=True)  
        
    parser.add_argument(
        "-save_plots", action='store_true', help="Save figures (default False)"
    )
    parser.add_argument(
        "-no-save_plots", dest='save_plots', action='store_false', help="Do not save figures"
    )
    parser.set_defaults(save_plots=True)
    
    args = parser.parse_args()

    main(args)
