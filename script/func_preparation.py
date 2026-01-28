import pickle
import random
from pathlib import Path
from time import sleep

import xarray as xr
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
from geopy.location import Location
from joblib import Parallel, delayed
from numpy import allclose
from pandas import DataFrame, MultiIndex
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

