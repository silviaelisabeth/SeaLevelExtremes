import random

import xarray as xr
from geopy.geocoders import Nominatim
from geopy.location import Location
from numpy import allclose


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


def verify_bias_correction(ds_model, annualMax_pref):
    site = random.randint(0, ds_model.sites.shape[0]-1)
    sample = random.randint(0, ds_model.sample.shape[0]-1)

    expected = ds_model.annualMax[sample, :, ds_model.annualMax.bc_flag[site].astype(int)-1, site].values 
    actual = annualMax_pref[sample, :, site].values

    print(
        f" - Validating bias correction:\n"
        f"\tAsserting whether actual ({actual}) selection matches expected selection ({expected})... "
    )
    assert allclose(actual, expected, equal_nan=True)


def select_valid_data(ds_model, annualMax_pref):
    data_valid = annualMax_pref.where(ds_model.model_valid == True, drop=True)

    sites_valid = data_valid.shape[-1]
    sites_total = annualMax_pref.shape[-1]
    rate_invalid = (1 - sites_valid / sites_total) * 100

    return data_valid, sites_valid, sites_total, rate_invalid