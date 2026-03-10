from datetime import datetime, timezone


mnf_outer = {
    "start_time": '2026-03-01T00:00:00',
    "end_time": datetime.now(timezone.utc).isoformat(),
    "west_lon": 111.0,
    "east_lon": 114.0,
    "south_lat": -25.0,
    "north_lat": -20.0
}

mnf_inner = {
    "west_lon": 113.0,
    "east_lon": 114.0,
    "south_lat": -23.0,
    "north_lat": -21.0
}

def get_proj():
    return mnf_outer, mnf_inner


s3_dict = {
    'layer_keys': ['sentinel3a_sst', 'sentinel3a_chl', 'sentinel3b_sst', 'sentinel3b_chl']
}

def get_s3_layers():
    return s3_dict['layer_keys']


swot_dict = {
    'short_name': 'SWOT_L2_LR_PreCalSSH_D',
    'data_type': {'Expert':['ssha_karin', 'swh_karin', 'sig0_karin']}    
}

def get_swot_info():
    return swot_dict