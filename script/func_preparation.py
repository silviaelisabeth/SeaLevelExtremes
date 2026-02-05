import pickle
import random
from pathlib import Path
from time import sleep
from typing import Optional, Tuple

import func_plotting as dbplt
import func_utils as ut
import reverse_geocoder as rg
import xarray as xr
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
from joblib import Parallel, delayed
from numpy import allclose, ndarray
from pandas import DataFrame, MultiIndex, concat
from tqdm import tqdm

#!!!ToDo: adding typing, remove unused functions...


def import_data_from_file(file):
    model_name = file.split('/')[-1].split('Annual_max_')[-1].split('.')[0]
    ds = xr.open_dataset(file, engine="netcdf4")
    
    return model_name, ds


def bias_correction(ds):
    return xr.apply_ufunc(
        lambda v, i: v[i], ds["annualMax"], ds["bc_flag"].astype(int) - 1,
        input_core_dims=[["bc"], []], output_core_dims=[[]],
        vectorize=True, dask="allowed", output_dtypes=[ds["annualMax"].dtype],
    )


def verify_bias_correction(
    ds_model, 
    annualMax_pref,
    ) -> None:
    site = random.randint(0, ds_model.sites.shape[0]-1)
    sample = random.randint(0, ds_model.sample.shape[0]-1)

    expected = ds_model.annualMax[sample, :, ds_model.annualMax.bc_flag[site].astype(int)-1, site].values 
    actual = annualMax_pref[sample, :, site].values

    print(
        f" - Validating bias correction:\n"
        f"\tAsserting whether actual ({actual}) selection matches expected selection ({expected})... "
    )
    assert allclose(actual, expected, equal_nan=True)


def select_valid_data(
    ds_model, 
    annualMax_pref
    ):
    data_valid = annualMax_pref.where(ds_model.model_valid == True, drop=True)

    sites_valid = data_valid.shape[-1]
    sites_total = annualMax_pref.shape[-1]
    rate_invalid = (1 - sites_valid / sites_total) * 100

    return data_valid, sites_valid, sites_total, rate_invalid


def locations_label_lookup_with_cache(
    locations: DataFrame, cache_file: str = "geocoding_cache.pkl", max_retries: int = 3, delay: float = 1.5,
    user_agent: str = "storm_surge_analysis"
    ):
    """
    Geocode locations with caching to resume after interruptions.
    
    If the script fails or is interrupted, you can restart and it will
    continue from where it left off.
    
    Parameters:
    -----------
    locations : DataFrame
        Must have 'lat' and 'lon' columns
    cache_file : str
        File to save progress (default: "geocoding_cache.pkl")
    max_retries : int
        Maximum retry attempts per location
    delay : float
        Delay between requests in seconds
    user_agent : str
        Custom user agent for the geocoder
        
    Returns:
    --------
    list : Location objects (or None for failures)
    """
    geolocator = Nominatim(user_agent=user_agent, timeout=10)
    
    cache_path = Path(cache_file)
    if cache_path.exists():
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = {}
    
    locations_label = [None] * len(locations)
    failed_indices = []
        
    try:
        for ix in tqdm(locations.index, desc="\t\tGeocoding"):
            if ix in cache:
                locations_label[ix] = cache[ix]
                continue
            
            lat = locations.loc[ix, 'lat']
            lon = locations.loc[ix, 'lon']
            
            location = None
            
            for attempt in range(max_retries):
                try:
                    location = geolocator.reverse((lat, lon), exactly_one=True, timeout=10)
                    break  
                    
                except GeocoderTimedOut:
                    if attempt < max_retries - 1:
                        sleep(2 ** attempt) 
                        continue
                    else:
                        failed_indices.append(ix)
                        
                except GeocoderServiceError:
                    failed_indices.append(ix)
                    break
                    
                except Exception as e:
                    print(f"\nError at index {ix}: {e}")
                    failed_indices.append(ix)
                    break
            
            locations_label[ix] = location
            cache[ix] = location 
            
            if (ix + 1) % 50 == 0:
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache, f)
            
            sleep(delay)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        print(f"✓ Progress saved to {cache_file}")
        print(f"  Run again to continue from index {ix}")
        raise
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    
    success_count = sum(1 for loc in locations_label if loc is not None)
    print(f"\n✓ Geocoding complete:")
    print(f"  Success: {success_count}/{len(locations)}")
    print(f"  Failed: {len(failed_indices)}")
    
    return locations_label


def locations_label_lookup_batched(
    locations: DataFrame,
    batch_size: int = 100,
    cache_file: str = "geocoding_cache.pkl",
    delay: float = 1.5
    ):
    """
    Geocode locations in batches with caching by (lat, lon).

    Parameters
    ----------
    locations : pd.DataFrame
        Must have 'lat' and 'lon' columns.
    batch_size : int
        Number of locations to process per batch.
    cache_file : str
        Path to cache file for saving progress.
    delay : float
        Delay in seconds between geocoding requests.

    Returns
    -------
    List
        List of geopy Location objects or None for failures, in the same order as `locations`.
    """
    
    cache_path = Path(cache_file)
    if cache_path.exists():
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        print(f"\t✓ Loaded {len(cache)} cached results")
    else:
        cache = {}

    geolocator = Nominatim(user_agent="storm_surge_analysis", timeout=10)
    
    total_batches = (len(locations) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(locations))
        print(f"\tProcessing batch {batch_num + 1}/{total_batches} (locations {start_idx}-{end_idx})")
        
        for ix in tqdm(range(start_idx, end_idx), desc=f"\tBatch {batch_num + 1}"):
            lat = locations.loc[ix, 'lat']
            lon = locations.loc[ix, 'lon']
            key = (lat, lon)
            
            if key in cache:
                continue 

            try:
                location = geolocator.reverse((lat, lon), exactly_one=True)
                cache[key] = location
            except Exception as e:
                print(f"\nError at {key}: {e}")
                cache[key] = None
            
            sleep(delay)
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        print(f"\t✓ Batch {batch_num + 1} saved to cache")
    
    locations_label = [cache.get((lat, lon)) for lat, lon in zip(locations['lat'], locations['lon'])]
    
    success_count = sum(1 for loc in locations_label if loc is not None)
    print(f"\t✓ All batches complete: {success_count}/{len(locations)} successful")
    
    return locations_label


def locations_label_lookup(locations: DataFrame):
    geolocator = Nominatim(user_agent="storm_surge_analysis", timeout=10)
    
    locations_label = []
    for ix in locations.index:
        lat = locations.loc[ix, 'lat']
        lon = locations.loc[ix, 'lon']
        
        try:
            location = geolocator.reverse((lat, lon), exactly_one=True)
        except GeocoderTimedOut:
            print(f"Timeout at index {ix}, retrying...")
            
            sleep(1)
            location = geolocator.reverse((lat, lon), exactly_one=True)
        
        locations_label.append(location)
        sleep(1) 
    return locations_label


def locations_label_lookup_simple(lat: float, lon: float):
    geolocator = Nominatim(user_agent="storm_surge_analysis", timeout=10)
    
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True)
    except GeocoderTimedOut:
        print(f"Timeout, retrying...")
        sleep(1)
        location = geolocator.reverse((lat, lon), exactly_one=True)
        
    return location


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
    
    locations_labeled = concat([locations.reset_index(drop=True), df_labels[['city','admin1','country']]], axis=1)
    
    return locations_labeled


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
    

def sites_to_location(da):
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


def process_model(file):
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


def process_location(loc_ex, combined, hindcast_start, hindcast_end):
    data_at_location_ = combined[:, :, :, loc_ex].to_dataframe().dropna().reset_index().rename(columns={'annualMax':'storm_surge'})
    data_at_location, message = ut.prepare_pooled_data_per_location(
        loc_ex=loc_ex,
        data=data_at_location_,
        hindcast_start=hindcast_start,
        hindcast_end=hindcast_end
    )
    return loc_ex, data_at_location, message


def data_rearrangement(combined, hindcast_start, hindcast_end):
    results = Parallel(n_jobs=-1, backend='loky', verbose=10)(
        delayed(process_location)(
            loc_ex=loc_ex, combined=combined, hindcast_start=hindcast_start, hindcast_end=hindcast_end
        ) for loc_ex in range(combined.shape[-1])
    )
    
    return results


def extract_location_data_and_info(results):
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


def get_location_w_missing_data(
    model_label:str, 
    dic_data_per_model: dict[str, dict[str, DataFrame]],
    combined: ndarray  
) -> Tuple[DataFrame, DataFrame]:
    df_geocoordinates_raw = DataFrame([
        dic_data_per_model[model_label]['raw data'].lat.values,
        dic_data_per_model[model_label]['raw data'].lon.values
        ], index=['lat', 'lon']).T


    df_geocoordinates_combined = DataFrame(
        [combined[0,0,0, :].lat.values, combined[0,0,0, :].lon.values], 
        index=['lat', 'lon']).T
    
    missing_locations = DataFrame(
        (
            set(zip(df_geocoordinates_raw['lat'], df_geocoordinates_raw['lon'])) 
            - set(zip(df_geocoordinates_combined['lat'], df_geocoordinates_combined['lon']))), 
        columns=['lat', 'lon']
        )
    
    return missing_locations, df_geocoordinates_combined


def create_summary_location_w_missing_data(dic_data_per_model:dict, combined, dir_export:Optional[str]):
    model_label = list(dic_data_per_model.keys())[0]

    missing_locations, df_valid = get_location_w_missing_data(model_label, dic_data_per_model, combined)
    print(f'{len(missing_locations)} locations without any valid data found!')
    
    df_missing_location = add_location_labels(missing_locations)
    df_missing_location = df_missing_location.sort_values('country')[['country', 'city', 'admin1', 'lat', 'lon']]
    
    if dir_export:
        file_name = dir_export + '/missing_locations_summary.txt'
        df_missing_location.to_csv(file_name, sep='\t', index=False)
        print(f"Overview of location with missing data saved as {file_name}.")
        
        dbplt.create_map_location_missing_valid_data(
            missing_locations=missing_locations, df_valid=df_valid, 
            dir_export=dir_export, store_map=True, display_map=False
            )
    return df_missing_location