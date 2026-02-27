from datetime import datetime, timezone

from mnf26_proj import get_proj, get_s3_layers, get_swot_info
from pysatview.himawari.himawari_processor import HimawariWorkflow
from pysatview.sentinel3.sentinel3_processor import Sentinel3Workflow
from pysatview.swot.swot_processor import SwotWorkflow

mnf_outer, mnf_inner = get_proj()


def init_himawari_workflow():
    """Initialise project for Himawari data processor"""
    
    timelims = (mnf_outer['start_time'], mnf_outer['end_time'])
    lonlims = (mnf_outer['west_lon'], mnf_outer['east_lon'])
    latlims = (mnf_outer['south_lat'], mnf_outer['north_lat'])
    tstep = 3600  # 1 hour interval
    
    workflow = HimawariWorkflow()
    
    workflow.run_complete_workflow(
        timelims=timelims,
        lonlims=lonlims,
        latlims=latlims,
        tstep=tstep
    )
    


def init_sentinel3_workflow():
    """Initialise project for Sentinel-3 data processor"""
    
    timelims = (mnf_outer['start_time'], mnf_outer['end_time'])
    region = (mnf_outer['west_lon'], mnf_outer['south_lat'], mnf_outer['east_lon'], mnf_outer['north_lat'])
    
    layer_keys = get_s3_layers()
    
    workflow = Sentinel3Workflow()
    
    workflow.run_complete_workflow(
        layer_keys=layer_keys, 
        region=region, 
        time_range=timelims
    )
    
    
    
def init_swot_workflow():
    """Initialise project for SWOT data processor"""
    
    swot_dict = get_swot_info()
    
    timelims = (mnf_outer['start_time'], mnf_outer['end_time'])
    lonlims = (mnf_outer['west_lon'], mnf_outer['east_lon'])
    latlims = (mnf_outer['south_lat'], mnf_outer['north_lat'])
        
    workflow = SwotWorkflow()

    workflow.run_complete_workflow(
        short_name=swot_dict['short_name'], 
        data_type=swot_dict['data_type'], 
        timelims=timelims, 
        lonlims=lonlims, 
        latlims=latlims, 
        only_last=False
    )
    


def main():
    try:
        init_himawari_workflow()
        print("✅ Himawari modules initialized successfully")
    except Exception as e:
        print(f"⚠️ Failed to initialize Himawari modules: {e}")
    
    try:
        init_sentinel3_workflow()
        print("✅ Sentinel-3 modules initialized successfully")
    except Exception as e:
        print(f"⚠️ Failed to initialize Sentinel-3 modules: {e}")
    
    try:
        init_swot_workflow()
        print("✅ SWOT modules initialized successfully")
    except Exception as e:
        print(f"⚠️ Failed to initialize SWOT modules: {e}")
    
    
if __name__ == "__main__":
    main()