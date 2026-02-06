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


# --------------------------------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------------------------------
def set_location_labels(labels):
    global _LOCATION_LABELS
    _LOCATION_LABELS = labels


def process_location(location_item):
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

    #export_path_site = None
    #if export_report and path_export:
    #    export_path_site = ut.save_location_results(
    #        location_id=loc_id,
    #        result_location=result,
    #        base_dir=path_export + '/gev_analysis/pooled/',
    #        plot_period_evolution=plot_period_evolution,
    #        display_results=display_results
    #    )
    #result['file_path_report'] = export_path_site
    
    return {
        "loc_id": loc_id,
        "data": df_prepared,
        "result": result,
        "location_info": {"lat": lat_loc, "lon": lon_loc,"label": location_info},
        "messages": messages,
    }


def run_gev_parallel(dic_data_per_location, location_labels, n_jobs=None):
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)

    set_location_labels(location_labels)

    items = list(dic_data_per_location.items())
    out = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
        delayed(process_location)(item) for item in items
    )

    #results = {}
    #ls_notes = []
    #for loc_id, result, messages in out:
    #    if messages:
    #        ls_notes.extend(messages)
    #    if result is not None:
    #        results[loc_id] = result
            
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


def run_annual_gev(results):
    results_extended, ls_notes_analysis = gev.execute_and_store_stat_gev_per_year_mp(
        results=results, store_results=False, return_periods=return_periods
    )

    for key, outer_list in ls_notes_analysis.items():
        ls_notes_analysis[key] = [inner for inner in outer_list if inner]
    return results_extended, ls_notes_analysis


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main(ls_files, args):
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
        dic_notes_analysis = {}
        dic_data_per_model = dbf.import_all_models(ls_files)
        print('\nImporting data done; next pooling and preparing data...')
        
        dic_data_per_model, combined, notes_overview = dbf.prepare_combined_data(ls_files, dic_data_per_model)
        dic_notes_analysis['data overview'] = notes_overview
        print('\nPooling and preparing data done.')
        
        if 'map' in ls_jobs:
            print('\nCreating map of locations with missing data...')
            missing_locations = dbf.create_summary_location_w_missing_data(
                dic_data_per_model=dic_data_per_model,
                combined=combined,
                dir_export=os.path.join(path_export, 'exploration') if save_plots is True else None
            )
            dic_notes_analysis['data pooling'] = [f"{len(missing_locations)} locations without any valid data found!"]

        print('\nRearranging data to sort per location...')    
        dic_data_per_location = dbf.extract_location_data(combined, hindcast_start, hindcast_end)
        
        if args.start_loc is not None or args.end_loc is not None:
            start = args.start_loc if args.start_loc is not None else min(dic_data_per_location.keys())
            end = args.end_loc if args.end_loc is not None else max(dic_data_per_location.keys())

            dic_data_per_location = {
                loc_id: df
                for loc_id, df in dic_data_per_location.items()
                if start <= loc_id <= end
            }
            print(f"Processing locations {start} to {end} ({len(dic_data_per_location)} total)")

        print('\nPrecomputing location labels...')
        location_labels = dbf.precompute_location_labels(dic_data_per_location)

        print('\nRearranging done; next run GEV analysis with pooled data...')
        output, ls_notes = run_gev_parallel(dic_data_per_location, location_labels)  
        results = output['results']  
        dic_notes_analysis['GEV pooled analysis'] = ls_notes
        
        print('\nSaving data per artifacts...')   
        today_ = str(datetime.today().date().isoformat())   
        path_child_folder = Path(path_export) / "gev_analysis" / "pooled" / f"{today_}"
        ut.save_pooled_results(results=results, data=output['data'], base_dir=path_child_folder)
        
        fig_dir = Path(path_child_folder) / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        if save_plots is True:
            for loc_id, result in results.items():
                fig = dbplt.plot_pooled_analysis(
                    result=result, 
                    site_id=loc_id, 
                    periods_evolution=plot_period_evolution, 
                    leg_comparison_x=0.35, leg_comparison_y=0.65, 
                    linestyle_trends=['dashdot', 'dashed', 'solid'], 
                    fontsize=12, figsize=(15, 7.5),
                    )
        
                fig.savefig(fig_dir / f"location_{loc_id}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
        print('\nData (and Figures) successfully saved per artifacts.')        

    if 'annual-stationary' in ls_jobs:
        try:
            results.keys()
            print('✓ continue with available dictionary')
            
        except NameError:
            path_import = os.path.join(path_export, 'gev_analysis','pooled/')
            print(f'\nImport data from folder {path_import}...')
            # results = ut.import_results_from_files_mp(path_import)
            results = ut.load_pooled_results(path_import)
    
        print('\nRun annual stationary GEV analysis...')
        results_extended, ls_notes_analysis = run_annual_gev(results)
        dic_notes_analysis['annual_statGEV'] = ls_notes_analysis
        
        today_ = str(datetime.today().date().isoformat())   
        path_child_folder = Path(path_export) / "gev_analysis" / "pooled" / f"{today_}"
        ut.save_pooled_results(
            results={
                site_id: {"fit results": site_data["fit results"]}
                for site_id, site_data in results_extended.items()
                if "fit results" in site_data
                },
            data=[], 
            base_dir=path_child_folder
        )

    if 'regression' in ls_jobs:
        print('\nTO be continued... Import data from files if not available')
        pass
        results_extended_list = Parallel(n_jobs=-1, backend='threading')(
            delayed(process_location)(site_id, dic_location, display_results, save_regression_summary)
            for site_id, dic_location in results_extended.items()
        )
        results_extended = {site_id: dic_location for site_id, dic_location in results_extended_list}

    print('\nAll analyses done; next store output...')
    ut.store_analysis_notes(dic_notes_analysis, path_export + '/gev_analysis/pooled/')

    # ---------------------------------------------------------------------------------------
    print(f"\nAnalysis completed. Log saved at {log_path}")

    logger.removeHandler(fh)
    fh.close()


if __name__ == "__main__":
    # NOTES: for execution, run from your terminal · python3 GEVanalysis.py --input_dir "/path/to/netcdf/files"
    parser = argparse.ArgumentParser(description="Run GEV analysis for multiple locations")
    
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Path to directory containing input NetCDF files (*.nc)"
    )
    
    parser.add_argument(
        "--pattern", type=str, default="*.nc",
        help="Filename pattern to match NetCDF files (default: '*.nc')"
    )
    
    parser.add_argument(
        "--start_loc", type=int, default=None,
        help="Start location ID (inclusive) to process"
    )
    
    parser.add_argument(
        "--end_loc", type=int, default=None,
        help="End location ID (inclusive) to process"
    )
    
    parser.add_argument(
        "--jobs", type=str, default=None,
        help="Started list of jobs to execute; available jobs are: pooled, annual-stationary, regression, map"
    )
    
    parser.add_argument(
        "--save_plots", type=bool, default=False,
        help="Boolean whether to save figures/panels in output folder or not (time-consuming!)"
    )    
    
    args = parser.parse_args()

    ls_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not ls_files:
        raise FileNotFoundError(f"No files found in {args.input_dir} matching {args.pattern}")

    main(ls_files, args)
