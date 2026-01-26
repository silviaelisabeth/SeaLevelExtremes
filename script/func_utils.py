from pandas import DataFrame


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
