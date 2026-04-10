import os
import re
import json
from pathlib import Path
from typing import Tuple, Optional, List

import cartopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import earthaccess

from inversion_sst_gp import plots
from inversion_sst_gp import gp_regression as gpr
from inversion_sst_gp.download import himawari


pkg_path = Path(__file__).parent.parent.parent.parent


class HimSSCDataProcessor:
    """Main class for processing Himawari satellite data."""
    
    # Constants
    COLLECTION_SHORT_NAME = "H09-AHI-L3C-ACSPO-v2.90"
    FILENAME_TIME_RE = re.compile(r"(\d{14})-STAR-L3C_")
    LONG_NAME = "STAR-L3C_GHRSST-SSTsubskin-AHI_H09-ACSPO_V2.90-v02.0-fv01.0"
    
    def __init__(self, base_dir: Path=pkg_path / "test_data" / "himawari_ssc"):
        """Initialize the processor with unified directory structure."""
        self.base_dir = base_dir
        
        # Ensure self.base_dir is a Path object
        self.base_dir = Path(self.base_dir)
        
        self.nc_dir = self.base_dir / "nc"  
        self.png_dir = self.base_dir / "png"

        # Create directories
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary directories for both unified and legacy structures."""
        # Create unified directories
        for directory in [self.base_dir, self.nc_dir, self.png_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
    def get_processed_nc_files(self):
        """Get list of already processed nc files in the unified directory."""
        if not self.nc_dir.exists():
            return []
        return [f for f in self.nc_dir.glob("*.nc") if f.is_file()]
    
    def ensure_earthdata_login(self, netrc_path: Path=pkg_path / "cred" / ".netrc"):
        """Login to Earthdata using .netrc file."""
        if not netrc_path.exists():
            raise FileNotFoundError(
                f".netrc not found at {netrc_path}\n"
                "Create a .netrc file with:\n"
                "machine urs.earthdata.nasa.gov\n  login YOUR_USERNAME\n  password YOUR_PASSWORD\n"
            )
        
        os.environ["NETRC"] = str(netrc_path.resolve())
        
        # Remove proxy settings that might interfere
        for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
            os.environ.pop(key, None)
            
        auth = earthaccess.login(strategy="netrc", persist=True)
        if not auth.authenticated:
            raise RuntimeError("Earthdata login failed. Check .netrc credentials")
        
        print("✓ Successfully authenticated with Earthdata")

    def process_time_series(
        self,
        timelims: Tuple[str, str],
        lonlims: Tuple[float, float],
        latlims: Tuple[float, float],
        tstep: int = 3600,
        netrc_path: Path = Path(__file__).parent / ".netrc",
        smallbox: Optional[Tuple[List, List]] = None
    ):
        self.ensure_earthdata_login(netrc_path)

        # Generate time range
        dtlims = (np.datetime64(timelims[0]), np.datetime64(timelims[1]))
        dtrange = np.arange(dtlims[0], dtlims[1], np.timedelta64(tstep, 's'))

        # Account for processing delay
        now_utc = np.datetime64(pd.Timestamp.utcnow().to_pydatetime())
        safe_latest = now_utc - np.timedelta64(3, 'h')
        dtrange = dtrange[dtrange <= safe_latest]

        print(f"Processing {len(dtrange)} time steps")
        print(f"Time range: {timelims[0]} to {timelims[-1]} UTC")

        for dt in dtrange:
            # Format back to string for download
            dt_str = pd.Timestamp(dt).strftime("%Y-%m-%dT%H:%M:%S")
            self._process_single_timestamp(
                dt_str,
                lonlims,
                latlims,
                smallbox=smallbox
            )
            

    def _process_single_timestamp(
        self,
        dt: np.datetime64,
        lonlims: Tuple[float, float],
        latlims: Tuple[float, float],
        smallbox: Optional[Tuple[List, List]] = None
    ):
        dt_pd = pd.Timestamp(dt)
        plt_str = dt_pd.strftime("%Y%m%d%H%M%S")
        
        ll_box = (lonlims, latlims)
        
        try:
            print(f"\nProcessing timestamp: {dt} UTC")
            # Download the data
            himawari.get_sst_scene_nasa(dt, self.nc_dir)
            
            # Crop and overwrite the full files (optional)
            previous_time, current_time, next_time = himawari.get_str_timesteps(dt)
            time_comb = [previous_time, current_time, next_time]
            [himawari.crop_sst_scene_nasa(self.nc_dir, time_list, ll_box,
                                        file_app='', overwrite=True) for time_list in time_comb]
            
            # Load the data
            ds = himawari.process_sst_scene(self.nc_dir, dt, ll_box, sst_reduce=2)
        except Exception as e:
            print(f"Failed to process timestamp {dt}: {e}")
            return
        
        # Check if dataset is empty after processing
        if ds is None or ds.sizes.get('time', 0) == 0:
            print(f"No valid data for timestamp {dt} after processing. Skipping.")
            return
        
        # Plot the spatial gradient data
        fig, ax = plots.plot_gradients(ds)
        

        png_name = f"{plt_str}_SST_gradients_outer.png"
        png_path = Path.joinpath(self.png_dir, png_name)
        for x in ax:
            x.add_feature(cartopy.feature.LAND, facecolor='w', zorder=2, edgecolor='grey', linewidths=1, alpha=1)
            x.add_feature(cartopy.feature.LAND, facecolor='olive', alpha=0.5, zorder=3, edgecolor=None, linewidths=0)

        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        if smallbox:
            small_name = f"{plt_str}_SST_gradients_inner.png"
            for x in ax:
                x.set_xlim(smallbox[0])
                x.set_ylim(smallbox[1])
            small_path = Path.joinpath(self.png_dir, small_name)
            fig.savefig(small_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved gradients PNG: {png_path}")  

        # Set model hyperparameters
        prop_sat = gpr.get_default_params()

        # Run model
        print("Running GP optimization")

        try:
            print("Fitting hyperparameters...")
            ds_results = gpr.fit_scene(ds, prop_sat, callback='off', coverage=0.4)
            if ds_results is None:
                return
        except Exception as e:
            print(f"Failed to fit hyperparameters: {e}")
            return

        try:
            print("Calculating current predictions")
            ds_prediction, Kpp_posterior = gpr.predict_scene(ds.isel(time=0), ds_results,
                                                            return_prior=True, return_cov=True,
                                                            coverage=0.4)

            fig, ax = plots.plot_pred_ellipses(ds_prediction, Kpp_posterior, scale=7, color='grey',
                                               alpha=0.5, an=False)
            
            png_name = f"{plt_str}_SST_predictions_outer.png"
            png_path = Path.joinpath(self.png_dir, png_name)
            for x in ax:
                x.add_feature(cartopy.feature.LAND, facecolor='w', zorder=2, edgecolor='grey', linewidths=1, alpha=1)
                x.add_feature(cartopy.feature.LAND, facecolor='olive', alpha=0.5, zorder=3, edgecolor=None, linewidths=0)
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            if smallbox:
                small_name = f"{plt_str}_SST_predictions_inner.png"
                for x in ax:
                    x.set_xlim(smallbox[0])
                    x.set_ylim(smallbox[1])
                small_path = Path.joinpath(self.png_dir, small_name)
                fig.savefig(small_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved predictions PNG: {png_path}")  
        
        except Exception as e:
            print(f"Failed to calculate current predictions: {e}")
            return


# Add a convenient utility class to handle the complete workflow
class SSCWorkflow:
    """Complete Himawari SSC workflow"""
    
    def __init__(self, base_dir: str = str(pkg_path) + "/test_data" + "/himawari_ssc"):
        self.processor = HimSSCDataProcessor(Path(base_dir))
    
    def run_complete_workflow(
        self,
        timelims: Tuple[str, str],
        lonlims: Tuple[float, float],
        latlims: Tuple[float, float],
        tstep: int = 3600,
        netrc_path: Path = pkg_path / "cred" / ".netrc",
        new_only=True,
        smallbox: Tuple[List, List] = None
    ):
        """
        Run the complete data processing workflow: query -> download -> process -> merge -> analyze
        
        Args:
            timelims: Time range
            lonlims: Longitude range  
            latlims: Latitude range
            tstep: Time step (seconds)
        """
        print("=== Himawari SSC Workflow ===")

        # Step 1: Update timelims to skip already processed files if new_only is True    
        if new_only:    
            print("\nChecking for already processed files...")        
            try: 
                existing_files = self.processor.get_processed_nc_files()
                if existing_files:
                    last_file_str = existing_files[-2].name.strip(".nc")
                
                    # Update timelims to only include new data
                    last_file_np = np.datetime64(last_file_str[0])
                    time_start_np = np.datetime64(timelims[0])
                    if last_file_np > time_start_np:
                        timelims = (str(last_file_np + np.timedelta64(1, 's')), timelims[1])
                        print(f"Updated time limits to only include new data: {timelims[0]} to {timelims[1]}")
                    
            except Exception as e:
                print(f"Failed to check existing files and update time limits: {e}")
                
        # Step 2: Process time series data
        print("\nProcessing data...")
        try:
            self.processor.process_time_series(
                timelims=timelims,
                lonlims=lonlims,
                latlims=latlims,
                tstep=tstep,
                netrc_path=netrc_path,
                smallbox=smallbox
            )
        except Exception as e:
            print(f"Failed to process data: {e}")
            
        # Save JSON manifests of inner and non-inner PNGs
        self.save_png_manifests()
        
        print(f"\n=== Workflow Complete ===")
        print(f"Results saved to: {self.processor.base_dir}")
        print(f"- Processed files: {self.processor.nc_dir}")
        print(f"- Visualizations: {self.processor.png_dir}")
        
        
    def save_png_manifests(self, output_dir: Optional[Path] = None):
        """Save manifest JSON files for inner and non-inner PNGs."""
        output_dir = output_dir or self.processor.png_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        png_files = sorted([p.name for p in output_dir.glob("*.png") if p.is_file()])
        inner_files = [f for f in png_files if "Ningaloo" in f]
        outer_files = [f for f in png_files if "Ningaloo" not in f]

        # Innner and outer JSON files
        inner_path = output_dir / "inner_png_files.json"
        outer_path = output_dir / "outer_png_files.json"

        inner_path.write_text(json.dumps(inner_files, indent=2))
        outer_path.write_text(json.dumps(outer_files, indent=2))

        print(f"Saved Ningaloo PNG manifest to: {inner_path} ({len(inner_files)} files)")
        print(f"Saved Gascoyne PNG manifest to: {outer_path} ({len(outer_files)} files)")


