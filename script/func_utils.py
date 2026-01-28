from typing import Optional

from pandas import DataFrame, MultiIndex


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
