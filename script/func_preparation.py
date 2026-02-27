import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import func_plotting as dbplt
import func_utils as ut
import reverse_geocoder as rg
import xarray as xr
from joblib import Parallel, delayed
from numpy import allclose, ndarray
from pandas import DataFrame, MultiIndex, concat
from xarray import DataArray, Dataset

logger = logging.getLogger("gev_analysis")

#!!!ToDo: adding typing, remove unused functions...


def import_all_models(ls_files:list) -> dict[str,dict]:
    dic_data_per_model = dict()
    for en, file in enumerate(ls_files):
        model_name = os.path.basename(file).split('.nc')[0].split('_')[-1]
        logger.info(f'Importing data from model {model_name} ({en+1}/{len(ls_files)})...')
        
        model_name, ds_model = import_data_from_file(file)
        dic_data_per_model[model_name] = {'raw data': ds_model}
    return dic_data_per_model


def import_data_from_file(file:str)->tuple[str,Dataset]:
    model_name = file.split('/')[-1].split('Annual_max_')[-1].split('.')[0]
    ds = xr.open_dataset(file, engine="netcdf4")
    return model_name, ds


def bias_correction(ds: Dataset) ->Dataset:
    return xr.apply_ufunc(
        lambda v, i: v[i], ds["annualMax"], ds["bc_flag"].astype(int) - 1,
        input_core_dims=[["bc"], []], output_core_dims=[[]],
        vectorize=True, dask="allowed", output_dtypes=[ds["annualMax"].dtype],
    )


def verify_bias_correction(
    ds_model:Dataset, 
    annualMax_pref:Dataset,
    ) -> None:
    site = random.randint(0, ds_model.sites.shape[0]-1)
    sample = random.randint(0, ds_model.sample.shape[0]-1)

    expected = ds_model.annualMax[sample, :, ds_model.annualMax.bc_flag[site].astype(int)-1, site].values 
    actual = annualMax_pref[sample, :, site].values

    logger.info(
        f" - Validating bias correction:\n"
        f"\tAsserting whether actual ({actual}) selection matches expected selection ({expected})... "
    )
    assert allclose(actual, expected, equal_nan=True)


def select_valid_data(
    ds_model:Dataset, 
    annualMax_pref:Dataset,
    )->tuple[Dataset,int, int,float]:
    data_valid = annualMax_pref.where(ds_model.model_valid == True, drop=True)

    sites_valid = data_valid.shape[-1]
    sites_total = annualMax_pref.shape[-1]
    rate_invalid = (1 - sites_valid / sites_total) * 100

    return data_valid, sites_valid, sites_total, rate_invalid


def prepare_combined_data(ls_files:list[str], dic_data_per_model:dict)->tuple[dict,Dataset,list]:
    dic_data_per_model = data_preparation(ls_files=ls_files, dic_data_per_model=dic_data_per_model)
    notes_overview = ut.create_data_overview(dic_data_per_model, ls_files)
    
    da_list = []
    for model_name, dic_model in dic_data_per_model.items():
        da = dic_model['valid data']
        da_loc = sites_to_location(da).expand_dims(model=[model_name])
        da_list.append(da_loc)
    combined = xr.concat(da_list, dim="model", join="outer", coords="different")

    return dic_data_per_model, combined, notes_overview


def extract_location_data(combined:Dataset, hindcast_start:int, hindcast_end:int)->dict:
    results = data_rearrangement(combined=combined, hindcast_start=hindcast_start, hindcast_end=hindcast_end)
    dic_data_per_location, _ = extract_location_data_and_info(results)
    return dic_data_per_location


def precompute_location_labels(dic_data_per_location:dict[str,DataFrame]) ->dict:
    coords = []
    for _, df in dic_data_per_location.items():
        lon = df.lon.unique()[0]
        lat = df.lat.unique()[0]
        coords.append((round(lon, 6), round(lat, 6)))

    df_coords = DataFrame(coords, columns=["lon", "lat"]).drop_duplicates()

    df_labels = add_location_labels(df_coords)
    return {
        (row.lon, row.lat): " ".join(row.values[2:])
        for _, row in df_labels.iterrows()
    }


def add_location_labels(locations: DataFrame) -> DataFrame:
    """
    Add city/admin/country labels to a DataFrame of lat/lon points using reverse_geocoder (offline).

    Args:
        locations (DataFrame): Must have 'lat' and 'lon' columns.

    Returns:
        DataFrame: Original DataFrame with additional columns: 'city', 'admin1', 'country'
    """
    if not {'lat', 'lon'}.issubset(locations.columns):
        raise ValueError("Input DataFrame must have 'lat' and 'lon' columns.")

    coords = list(zip(locations['lat'], locations['lon']))
    
    results = rg.search(coords) 
    df_labels = DataFrame(results)
    
    df_labels = df_labels.rename(columns={
        'name': 'city',
        'admin1': 'admin1',
        'cc': 'country'
    })
        
    return concat([locations.reset_index(drop=True), df_labels[['city','admin1','country']]], axis=1)


def prepare_data(data: DataFrame, hindcast_start:int, hindcast_end: int) -> DataFrame:
        """Calculate target years and filter to hindcast period."""
        data['target_year'] = data['sim_year'] + data['lead']

        mask = (data['target_year'] >= hindcast_start) & \
                (data['target_year'] <= hindcast_end)
        data_hindcast = data[mask].copy()

        logger.info(
            f"\nData Summary:"
            f"\n\tHindcast period: {hindcast_start}-{hindcast_end}"
            f"\n\tTotal observations: {len(data_hindcast)}"
            f"\n\tModels: {data_hindcast['model'].nunique()}"
            f"\n\tLocations: {min(data_hindcast[['lon', 'lat']].nunique().values)}"
            )
        
        return data_hindcast
    

def sites_to_location(da:DataArray)->DataArray:
    """
    Replace arbitrary 'sites' index with a geographic location index (lon, lat).

    Parameters
    ----------
    da : xr.DataArray
        dims: sample x member x sites
        coords: lon(sites), lat(sites)
    round_coords : int
        Decimal places for lon/lat to stabilize floating point comparisons
    """

    lon = da.lon.values
    lat = da.lat.values

    location_index = MultiIndex.from_arrays([lon, lat], names=("lon", "lat"))

    # Replace 'sites' with 'location'
    da = da.assign_coords(location=("sites", location_index))
    da = da.swap_dims({"sites": "location"})
    da = da.drop_vars("sites")

    return da


def process_model(file:str)->tuple[str,Dataset,tuple[int,int,float]]:
    model_name, ds_model = import_data_from_file(file) 
    ds_model_corrected = bias_correction(ds_model)
    data_valid, sites_valid, sites_total, rate_invalid = select_valid_data(
        ds_model, ds_model_corrected
    )
    ds_model.close()
    return model_name, data_valid, (sites_valid, sites_total, rate_invalid)


def data_preparation(ls_files:list[str], dic_data_per_model:dict) -> dict:
    results = Parallel(n_jobs=4)(
        delayed(process_model)(file) for file in ls_files
    )

    for model_name, data_valid, prep_info in results:
        dic_data_per_model[model_name]['valid data'] = data_valid
        dic_data_per_model[model_name]['preparation info'] = prep_info
    return dic_data_per_model


def process_location(loc_ex:int, combined:Dataset, hindcast_start:int, hindcast_end:int)->tuple[int,dict,str]:
    data_at_location_ = combined[:, :, :, loc_ex].to_dataframe().dropna().reset_index()
    data_at_location_ = data_at_location_.rename(columns={'annualMax':'storm_surge'})
    
    data_at_location, message = ut.prepare_pooled_data_per_location(
        loc_ex=loc_ex,
        data=data_at_location_,
        hindcast_start=hindcast_start,
        hindcast_end=hindcast_end
    )
    return loc_ex, data_at_location, message


def data_rearrangement(combined:Dataset, hindcast_start:int, hindcast_end:int)->dict:
    results = Parallel(n_jobs=-1, backend='loky', verbose=10)(
        delayed(process_location)(
            loc_ex=loc_ex, combined=combined, hindcast_start=hindcast_start, hindcast_end=hindcast_end
        ) for loc_ex in range(combined.shape[-1])
    )
    
    return results


def extract_location_data_and_info(results:dict)->tuple[dict,DataFrame]:
    dic_data_per_location = {}
    messages = []
    for loc_ex, data_at_location, message in results:
        dic_data_per_location[loc_ex] = data_at_location
        messages.append({'location': loc_ex, 'message': message})

    df_messages = DataFrame(messages).set_index('location')

    dic_location_info = dict(map(lambda site: 
        (int(site), DataFrame([m.strip() for m in df_messages.loc[site, 'message'].split('\t')])), df_messages.index
        ))
    data_overview_preped = concat(dic_location_info, axis=1).T
    data_overview_preped.index = data_overview_preped.index.levels[0]
    data_overview_preped.columns=['siteID', 'hindcast period', 'available models at location', 'observations for analysis']

    column = 'observations for analysis'
    data_overview_preped[column] = [int(o.split('analyse:')[1].strip()) for o in data_overview_preped[column].values]

    column = 'siteID'
    data_overview_preped[column] = [int(o.split('siteID')[1].strip()) for o in data_overview_preped[column].values]

    column = 'hindcast period'
    data_overview_preped[column] = [o.split('period:')[1].strip() for o in data_overview_preped[column].values]

    column = 'available models at location'
    data_overview_preped[column] = [int(o.split('Location:')[1].strip()) for o in data_overview_preped[column].values]
    
    return dic_data_per_location, data_overview_preped


def get_number_of_observations_per_site(dic_data_per_location:dict) -> DataFrame:
    df_all = concat(dic_data_per_location, names=['site_id'])
    return df_all.groupby('site_id').agg(lat=('lat', 'first'), lon=('lon', 'first'), n_obs=('lat', 'size')).reset_index()


def get_location_w_missing_data(
    model_label:str, 
    dic_data_per_model: dict[str, dict[str, DataFrame]],
    n_obs_per_location:DataFrame,
    combined: ndarray  
) -> tuple[DataFrame, DataFrame]:
    df_geocoordinates_raw = DataFrame([
        dic_data_per_model[model_label]['raw data'].lat.values,
        dic_data_per_model[model_label]['raw data'].lon.values
        ], index=['lat', 'lon']).T


    df_geocoordinates_combined = DataFrame(
        [combined[0,0,0, :].lat.values, combined[0,0,0, :].lon.values], 
        index=['lat', 'lon']).T
    
    df_geocoordinates_combined = df_geocoordinates_combined.merge(
        n_obs_per_location[['lat', 'lon', 'n_obs']], on=['lat', 'lon'], how='left'
        )
    df_geocoordinates_combined['info'] = "Observations: " + df_geocoordinates_combined['n_obs'].astype(str)
    
    missing_locations = DataFrame(
        (
            set(zip(df_geocoordinates_raw['lat'], df_geocoordinates_raw['lon'])) 
            - set(zip(df_geocoordinates_combined['lat'], df_geocoordinates_combined['lon']))), 
        columns=['lat', 'lon']
        )
    
    
    
    return missing_locations, df_geocoordinates_combined


def create_summary_location_w_missing_data(
    dic_data_per_model:dict, combined, n_obs_per_location:DataFrame, dir_export:Optional[str]=None
    )->DataFrame:
    
    model_label = list(dic_data_per_model.keys())[0]
    missing_locations, df_valid = get_location_w_missing_data(
        model_label=model_label, dic_data_per_model=dic_data_per_model, combined=combined, 
        n_obs_per_location=n_obs_per_location
        )
    logger.info(f'{len(missing_locations)} locations without any valid data found!')
    
    df_missing_location = add_location_labels(missing_locations)
    df_missing_location = df_missing_location.sort_values('country')[['country', 'city', 'admin1', 'lat', 'lon']]
    
    fig = dbplt.create_map_location_missing_valid_data(
        missing_locations=missing_locations, df_valid=df_valid,
        dir_export=dir_export, store_map=True if dir_export else False, 
        )
    
    if dir_export:
        save_dir = Path(dir_export)
        save_dir.mkdir(parents=True, exist_ok=True) 

        time_date = datetime.today().date().isoformat()
        file_name = save_dir / 'missing_locations_summary_{time_date}.txt'
        df_missing_location.to_csv(file_name, sep='\t', index=False)
        logger.info(f"Overview of location with missing data saved as {file_name}.")


    return df_missing_location, df_valid, fig