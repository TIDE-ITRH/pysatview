from datetime import datetime, timezone, timedelta
from pathlib import Path
import shutil
from PIL import Image
import numpy as np


from mnf26_proj import get_proj, get_s3_layers, get_swot_info
from pysatview.himawari.himawari_processor import HimawariWorkflow
from pysatview.sentinel3.sentinel3_processor import Sentinel3Workflow
from pysatview.swot.swot_processor import SwotWorkflow


pkg_path = Path(__file__).parent.parent.parent

mnf_outer, mnf_inner = get_proj()

proj_init_file = Path('mnf26_started.txt')

base_dir = pkg_path / 'mnf_test'

if proj_init_file.exists():
    proj_update = True
else:
    proj_update = False
    proj_init_file.touch() 


def init_himawari_workflow():
    """Initialise project for Himawari data processor"""
    
    if proj_update:
        t_start = np.datetime64(mnf_outer['end_time']) - np.timedelta64(1, 'D')
        t_start = t_start.astype('datetime64[h]')
        timelims = (t_start, mnf_outer['end_time'])
    else:
        timelims = (mnf_outer['start_time'], mnf_outer['end_time'])
    lonlims = (mnf_outer['west_lon'], mnf_outer['east_lon'])
    latlims = (mnf_outer['south_lat'], mnf_outer['north_lat'])
    tstep = 3600  # 1 hour interval
    
    workflow = HimawariWorkflow(base_dir=base_dir / 'himawari')
    
    workflow.run_complete_workflow(
        timelims=timelims,
        lonlims=lonlims,
        latlims=latlims,
        tstep=tstep,
        smallbox=([mnf_inner['west_lon'], mnf_inner['east_lon']],
                [mnf_inner['south_lat'], mnf_inner['north_lat']]),
    )
    return workflow.processor.png_dir
    


def init_sentinel3_workflow():
    """Initialise project for Sentinel-3 data processor"""
    
    if proj_update:
        t_start = datetime.fromisoformat(mnf_outer['end_time']) - timedelta(days=2)
        timelims = (t_start.isoformat(), mnf_outer['end_time'])
    else:
        timelims = (mnf_outer['start_time'], mnf_outer['end_time'])

    region = (mnf_outer['west_lon'], mnf_outer['south_lat'], mnf_outer['east_lon'], mnf_outer['north_lat'])
    
    layer_keys = get_s3_layers()
    
    workflow = Sentinel3Workflow(base_dir=base_dir)
    
    workflow.run_complete_workflow(
        layer_keys=layer_keys, 
        region=region, 
        time_range=timelims,
        get_all_available = not proj_update,
        smallbox=([mnf_inner['west_lon'], mnf_inner['east_lon']],
                [mnf_inner['south_lat'], mnf_inner['north_lat']]),
    )
    png_dirs = []
    for layer in layer_keys:
        satellite, data_type = workflow.processor._parse_layer_key(layer)
        png_dirs.append(workflow.processor.get_png_path(satellite, data_type, '_').parent)
    return png_dirs
    
    
    
def init_swot_workflow():
    """Initialise project for SWOT data processor"""
    
    swot_dict = get_swot_info()
    
    if proj_update:
        t_start = np.datetime64(mnf_outer['end_time']) - np.timedelta64(4, 'D')
        timelims = (str(t_start), mnf_outer['end_time'])
    else:
        timelims = (mnf_outer['start_time'], mnf_outer['end_time'])

    lonlims = (mnf_outer['west_lon'], mnf_outer['east_lon'])
    latlims = (mnf_outer['south_lat'], mnf_outer['north_lat'])
        
    workflow = SwotWorkflow(base_dir=base_dir / 'swot')

    workflow.run_complete_workflow(
        short_name=swot_dict['short_name'], 
        data_type=swot_dict['data_type'], 
        timelims=timelims, 
        lonlims=lonlims, 
        latlims=latlims, 
        only_last=proj_update,
        smallbox=([mnf_inner['west_lon'], mnf_inner['east_lon']],
                [mnf_inner['south_lat'], mnf_inner['north_lat']]),
    )
    return workflow.processor.png_dir



def latest_pdf(png_dirs):
    '''Move files and create the latest PDF'''
    # Make the latest dir
    latest_dir = Path(base_dir / '_latest')
    latest_dir.mkdir(parents=True, exist_ok=True)
    
    for old_latest in latest_dir.iterdir():
        if old_latest.is_file():
            old_latest.unlink()
    
    # Add lestest PNGs to a combined PDF
    print("📄 Compiling latest PNGs into a PDF report...")
    pdf_files = []
    for png_dir in png_dirs:
        png_path = Path(png_dir)
        if png_path.exists() and png_path.is_dir():
            outer = [p for p in png_path.glob('*.png') if 'inner' not in p.name]
            if outer:
                pdf_files.append(max(outer, key=lambda p: p.stat().st_mtime))

            inner = [p for p in png_path.glob('*.png') if 'inner' in p.name]
            if inner:
                pdf_files.append(max(inner, key=lambda p: p.stat().st_mtime))
                
    # Copy all latest pngs to latest folder
    for png in pdf_files:
        shutil.copy2(png, latest_dir / png.name)
        
    # Create PDF
    imgs = [Image.open(p) for p in pdf_files]
    imgs = [im.convert("RGB") for im in imgs]

    # first image .save() with the rest as an append list
    out_path = latest_dir / (datetime.now(timezone.utc).strftime("%Y%m%d_%H") + "h_latest.pdf")
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:])
    print(f"Created {out_path} with {len(imgs)} pages")    

    


def main():
    png_dir_all = []
    # try:
    #     him_png = init_himawari_workflow()
    #     png_dir_all += [him_png]
    #     print("✅ Himawari modules initialized successfully\n")
    # except Exception as e:
    #     print(f"⚠️ Failed to initialize Himawari modules: {e}\n")
    
    try:
        sent_png = init_sentinel3_workflow()
        png_dir_all += sent_png
        print("✅ Sentinel-3 modules initialized successfully\n")
    except Exception as e:
        print(f"⚠️ Failed to initialize Sentinel-3 modules: {e}\n")
    
    # try:
    #     swot_png = init_swot_workflow()
    #     png_dir_all += [swot_png]
    #     print("✅ SWOT modules initialized successfully\n")
    # except Exception as e:
    #     print(f"⚠️ Failed to initialize SWOT modules: {e}\n")
        
    try:
        latest_pdf(png_dir_all)
        print("✅ PDF report compiled successfully\n")
    except Exception as e:
        print(f"⚠️ Failed to compile PDF report: {e}\n")
    
    
if __name__ == "__main__":
    main()