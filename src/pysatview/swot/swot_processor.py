"""
SWOT Satellite Data Processing Module

This module provides functionality to download, process, and visualize
SWOT satellite data from the AVISO FTP server.

Based on the original swotapi.py, refactored for production use following
the EUMETView pattern for consistency across satellite modules.
"""

import os
import glob
import time
import warnings
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from datetime import datetime
# import ftplib
import earthaccess
from urllib.parse import urlparse

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cmocean
import cartopy
import cartopy.crs as ccrs

# Turn off warnings for cleaner output
warnings.simplefilter("ignore")

# Add package basedir
pkg_path = Path(__file__).parent.parent.parent.parent

cmap_dict = {
    'ssha_karin': 'RdBu_r',
    'swh_karin': 'RdYlBu_r',
    'sig0_karin': 'gray'
}


class SwotDataProcessor:
    """Main class for processing SWOT satellite data from AVISO FTP."""
    
    # Constants
    FTP_SERVER = 'ftp-access.aviso.altimetry.fr'
    
    # SWOT data parameters
    LEVELS = ['L2', 'L3']
    VARIANTS = ['Basic', 'Expert', 'WindWave', 'Unsmoothed']
    
    def __init__(self, base_dir: Path=pkg_path / "test_data" / "swot"):
        """Initialize the SWOT processor with unified directory structure."""
        self.unified_base_dir = Path(base_dir)
        
        self.base_dir = self.unified_base_dir  # Use unified structure for file operations
        
        # Create directories
        self._setup_directories()
        
        self.logged_in = False
        
    def _setup_directories(self):
        """Create directory structure for both unified and legacy layouts."""
        # Create unified directories: data/swot/{parameter}/{file_type}/
        self.nc_dir = self.unified_base_dir / "nc"
        self.fullnc_dir = self.unified_base_dir / "nc_full"
        self.png_dir = self.unified_base_dir / "png"
        self.nc_dir.mkdir(parents=True, exist_ok=True)
        self.fullnc_dir.mkdir(parents=True, exist_ok=True)
        self.png_dir.mkdir(parents=True, exist_ok=True)
            
    def get_nc_path(self, filename: str) -> Path:
        """Get NC file path: use unified structure for new files"""
        return self.nc_dir / filename
    
    def get_fullnc_path(self, filename: str) -> Path:
        """Get full NC file path: use unified structure for new files"""
        return self.fullnc_dir / filename
    
    def get_png_path(self, filename: str) -> Path:
        """Get PNG file path: use unified structure for new files"""
        return self.png_dir / filename
    
    def authenticate(self, netrc_path: Path=pkg_path / "cred" / ".netrc"):
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
        self.logged_in = True
        print("✓ Successfully authenticated with Earthdata")

    def search_data(
        self,
        short_name: str,
        time_range: Tuple[str, str],
        lon_range: Tuple[float, float],
        lat_range: Tuple[float, float],
        version: Optional[str] = None,
    ):
        """Search SWOT granules by time and bounding box."""

        if not self.logged_in:
            raise RuntimeError("Must call authenticate() first")

        bbox = (lon_range[0], lat_range[0], lon_range[1], lat_range[1])
        
        try:
            time_start = pd.to_datetime(time_range[0]).to_pydatetime()
            time_end = pd.to_datetime(time_range[1]).to_pydatetime()
        except Exception as e:
            raise ValueError(f"Conversion to datetime failed: {e}")

        results = earthaccess.search_data(
            short_name=short_name,
            temporal=(time_start, time_end),
            bounding_box=bbox,
            version=version,
        )
        return results

    def download_data(self, results, data_type: Dict = None, only_last: bool = True) -> List[Path]:
        """Download granules from earthaccess search results."""

        if len(results) == 0:
            raise RuntimeError("No SWOT granules found for query")

        if only_last:
            results = results[-4:]
        # else:
        #     results = results[2::4]

        files_full = earthaccess.download(results, local_path=self.fullnc_dir)
        
        # Straight away delete unwanted filetypes
        if data_type:
            files = [f for f in files_full if any(dt in str(f) for dt in data_type.keys())]
            [os.remove(f) for f in files_full if not any(dt in str(f) for dt in data_type.keys())]
        return [f for f in files]

    # def subset_spatial_and_variables(
    #     self,
    #     files: List[Path],
    #     variables: List[str],
    #     lon_range: Tuple[float, float],
    #     lat_range: Tuple[float, float],
    # ) -> List[Path]:
    #     """Subset by lat/lon box and variables."""

    #     output_files = []

    #     for f in files:
    #         ds = xr.open_dataset(f)

    #         # Spatial subset (if lon/lat present)
    #         if "longitude" in ds and "latitude" in ds:
    #             ds = ds.where(
    #                 (ds.longitude >= lon_range[0]) &
    #                 (ds.longitude <= lon_range[1]) &
    #                 (ds.latitude >= lat_range[0]) &
    #                 (ds.latitude <= lat_range[1]),
    #                 drop=True
    #             )

    #         # Variable subset
    #         if variables:
    #             keep = [v for v in variables if v in ds]
    #             ds = ds[keep]

    #         out_file = self.download_dir / f"subset_{f.name}"
    #         ds.to_netcdf(out_file)
    #         ds.close()

    #         output_files.append(out_file)

    #     return output_files

    def _normalized_ds(self, ds: xr.Dataset, lon_min: float, lon_max: float) -> xr.Dataset:
        """Normalize longitude values in dataset."""
        lon = ds.longitude.values
        lon[lon < lon_min] += 360
        lon[lon > lon_max] -= 360
        ds.longitude.values = lon
        return ds

    def _subset_ds(
        self, 
        file_path: str, 
        lon_range: Tuple[float, float], 
        lat_range: Tuple[float, float]
    ) -> Optional[str]:
        """Subset dataset to geographical area and save."""
        # print(f"Subset dataset: {file_path}")
        
        try:
            swot_ds = xr.open_dataset(file_path)
            swot_ds.load()

            ds = self._normalized_ds(swot_ds.copy(), -180, 180)

            mask = (
                (ds.longitude <= lon_range[1])
                & (ds.longitude >= lon_range[0])
                & (ds.latitude <= lat_range[1])
                & (ds.latitude >= lat_range[0])
            ).compute()

            swot_ds_area = swot_ds.where(mask, drop=True)

            if swot_ds_area.sizes['num_lines'] == 0:
                print(f'Dataset {file_path} not matching geographical area.')
                swot_ds.close()
                os.remove(file_path)
                return None

            for var in list(swot_ds_area.keys()):
                swot_ds_area[var].encoding = {'zlib': True, 'complevel': 5}

            nc_output_path = self.get_nc_path(Path(file_path).name)
            
            swot_ds_area.to_netcdf(nc_output_path, mode='w')
            print(f"Subset saved: {nc_output_path}")

            swot_ds.close()
            return str(nc_output_path)
            
        except Exception as e:
            print(f"Error subsetting {file_path}: {e}")
            return None


    def subset_files(
        self,
        filenames: List[str],
        lon_range: Tuple[float, float],
        lat_range: Tuple[float, float]
    ) -> List[str]:
        """Subset multiple datasets with geographical area."""
        subset_files = []
        for filename in filenames:
            subset_file = self._subset_ds(str(filename), lon_range, lat_range)
            if subset_file:
                subset_files.append(subset_file)
        return subset_files


    def _create_single_plot(
        self,
        ds: xr.Dataset,
        variable: str,
        extent: Optional[List[float]] = None,
        filename: Optional[str] = None,
        strap: str = ""
    ) -> str:
        """Create a single plot for data with one time step."""

        # Handle SWOT's 2D time structure
        timestamp = None
        
        # First try to extract time from filename (SWOT specific)
        if filename:
            try:
                # Extract time from SWOT filename pattern: SWOT_L3_LR_SSH_Expert_029_062_20250226T145417_20250226T154543_v2.0.1.nc
                import re
                time_pattern = r'(\d{8}T\d{6})_(\d{8}T\d{6})'
                match = re.search(time_pattern, str(filename))
                if match:
                    start_time = match.group(1)  # 20250226T145417
                    timestamp = start_time.replace('T', '_')  # 20250226_145417
                    print(f"Using filename time for plot: {timestamp}")
            except Exception as e:
                print(f"Could not extract time from filename: {e}")
        
        # If filename extraction failed, try data time
        if not timestamp and 'time' in ds.dims and ds.sizes['time'] > 0:
            try:
                # Check if time is 2D array (SWOT structure)
                if len(ds['time'].shape) > 1:
                    # For 2D time arrays, try to find valid time values
                    time_vals = ds['time'].values.flatten()
                    valid_times = time_vals[~pd.isna(time_vals)]
                    if len(valid_times) > 0:
                        # Use the first valid time
                        timestamp = pd.to_datetime(valid_times[0]).strftime("%Y%m%d_%H%M%S")
                        print(f"Using data time for filename: {timestamp}")
                    else:
                        # No valid times, use current time
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        print(f"No valid times found, using current time: {timestamp}")
                else:
                    # For 1D time arrays (standard structure)
                    time_val = ds['time'].isel(time=0).values
                    if pd.notna(time_val):
                        timestamp = pd.to_datetime(time_val).strftime("%Y%m%d_%H%M%S")
                    else:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            except Exception as e:
                print(f"Warning: Error processing time for {variable}: {e}")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get data for plotting
        data = ds[variable]
        if extent:
            data = data.where(
                (ds.longitude >= extent[0]) &
                (ds.longitude <= extent[1]) &
                (ds.latitude >= extent[2]) &
                (ds.latitude <= extent[3]),
                drop=True
            )
        
        # Handle time dimension indexing carefully
        if 'time' in data.dims:
            # Check if time dimension is 1D (standard) or 2D (SWOT structure)
            if len(ds['time'].shape) == 1:
                # Standard 1D time dimension
                data = data.isel(time=0)
            else:
                # SWOT 2D time structure - don't index by time dimension
                # The data is already 2D and doesn't need time indexing
                pass
        
        # Create plot
        fig, ax = plt.subplots(figsize=(6,8), subplot_kw={'projection': ccrs.PlateCarree()})

        try:
            dmax = np.nanpercentile(np.abs(data.where(ds[variable + '_qual'] == 0)), 99)
            data = data.where(ds[variable + '_qual'] == 0)
        except Exception as e:
            dmax = np.nanpercentile(np.abs(data), 99)
            print(f"No quality control variable found for {variable}")

        if variable == 'ssha_karin':
            dmin = -dmax
        else:
            dmin = 0

        # Plot data
        im = plt.pcolor(
            data['longitude'],
            data['latitude'],
            data,
            vmin=dmin,
            vmax=dmax,
            cmap=cmap_dict[variable])

        # ax.coastlines()
        # ax.gridlines(draw_labels=True)
        ax.set_title(f"SWOT {variable.upper()} - {timestamp}")
        
        if extent:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            
        # Add coastlines 
        ax.add_feature(cartopy.feature.LAND, facecolor='w', zorder=2, edgecolor='grey', linewidths=1, alpha=1)
        ax.add_feature(cartopy.feature.LAND, facecolor='olive', alpha=0.5, zorder=3, edgecolor=None, linewidths=0)

        plt.colorbar(im, shrink=0.3, label=data.attrs['long_name'] + f' [{data.attrs['units']}]')
        plt.axis('equal')
        plt.tight_layout()
        
        # Save plot
        png_path = self.get_png_path(f"{timestamp}_{variable}_swot{strap}.png")
        plt.savefig(png_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {png_path}")
        return str(png_path)


    def plot_datasets(
        self,
        filenames: List[str],
        variable: str,
        extent: Optional[List[float]] = None,
        strap: str = ""
    ) -> List[str]:
        """
        Create plots from NetCDF files.
        
        Args:
            filenames: List of NetCDF file paths
            variable: Variable to plot
            extent: Plot extent [lon_min, lon_max, lat_min, lat_max]
            create_individual_plots: Create individual time series plots
            
        Returns:
            List of generated PNG file paths
        """
        png_files = []
        
        for filename in filenames:
            try:
                ds = xr.open_dataset(filename)
                
                if variable not in ds.data_vars:
                    print(f"Variable {variable} not found in {filename}")
                    continue

                if np.all(ds[variable].isnull()):
                    print('No SWOT data to plot for {filename.name} - {variable}')
                    continue
                else:
                    try:
                        if np.all(ds[variable].where(ds[variable + '_qual'] == 0).isnull()):
                            print('No SWOT data to plot for {filename.name} - {variable}')
                            continue
                        else:
                            pass
                    except Exception as e:
                        pass
                
                png_file = self._create_single_plot(ds, variable, extent, filename, strap)
                if png_file:
                    png_files.append(png_file)
                ds.close()
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        return png_files


    def process_and_visualize(
        self,
        downloaded_files: List[str],
        data_type: Dict[str, str],
        lon_range: Tuple[float, float],
        lat_range: Tuple[float, float],
    ) -> Dict[str, List[str]]:
        """
        Process downloaded files and create visualizations.
        
        Args:
            downloaded_files: List of downloaded file paths
            variables: Variables to extract and plot
            lon_range: Longitude range for subsetting
            lat_range: Latitude range for subsetting
            create_individual_plots: Create individual time series plots
            
        Returns:
            Dictionary mapping variable to list of generated PNG files
        """
        visualization_files = {}
        
        if not downloaded_files:
            print("No files to process")
            return None, visualization_files
        
        # Subset files
        print(f"Subsetting {len(downloaded_files)} files...")
        subset_filenames = self.subset_files(
            downloaded_files, lon_range, lat_range
        )
        # Drop files from downloaded_files if data_type not in filename
        # subset_filenames = [f for f in downloaded_files if any(dt in str(f) for dt in data_type.keys())]
        # subset_filenames = downloaded_files
        
        if not subset_filenames:
            print("No subset files generated")
            return None, visualization_files
        
        # Create visualizations for each variable
        for dtype in data_type.keys():
            vis_filnames = [f for f in subset_filenames if dtype in str(f)]
            
            for var_plt in data_type[dtype]:
                print(f"Creating visualizations for {var_plt}...")
                extent = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]
                
                png_files = self.plot_datasets(
                    vis_filnames, var_plt, extent
                )
            
            visualization_files[var_plt] = png_files
            print(f"Generated {len(png_files)} plots for {var_plt}")
        
        return subset_filenames, visualization_files
        
    
    def get_latest_file_time(self, satellite: str) -> Optional[datetime]:
        """
        Get the latest time from existing NC files for a specific satellite and data type.
        
        Args:
            satellite: Satellite name (sentinel3a, sentinel3b)
            
        Returns:
            Latest datetime found in existing files, or None if no files exist
        """
        try:
            nc_dir = self.nc_dir
            if not nc_dir.exists():
                return None
            
            nc_files = list(nc_dir.glob("*.nc"))
            if not nc_files:
                return None
            
            latest_time = None
            
            for nc_file in nc_files:
                try:
                    with xr.open_dataset(nc_file) as ds:
                        if 'time' in ds.variables:
                            # Get the latest time from this file
                            file_times = ds['time'].values.flatten()
                            # Remove NaT values
                            file_times = file_times[~pd.isnull(file_times)]
    
                            if len(file_times) > 0:
                                # Convert to datetime
                                if hasattr(file_times, 'max'):
                                    max_time = file_times.max()
                                else:
                                    max_time = file_times[-1] if len(file_times) > 0 else file_times[0]
                                
                                # Convert numpy datetime to Python datetime
                                if hasattr(max_time, 'item'):
                                    max_time = max_time.item()
                                
                                # Handle different datetime formats
                                if hasattr(max_time, 'to_pydatetime'):
                                    max_time = max_time.to_pydatetime()
                                elif isinstance(max_time, (int, float)):
                                    # Handle timestamp (nanoseconds)
                                    if max_time > 1e18:  # nanoseconds
                                        max_time = datetime.fromtimestamp(max_time / 1e9)
                                    elif max_time > 1e15:  # microseconds
                                        max_time = datetime.fromtimestamp(max_time / 1e6)
                                    elif max_time > 1e12:  # milliseconds
                                        max_time = datetime.fromtimestamp(max_time / 1e3)
                                    else:  # seconds
                                        max_time = datetime.fromtimestamp(max_time)
                                
                                if latest_time is None or max_time > latest_time:
                                    latest_time = max_time
                except Exception as e:
                    print(f"Warning: Could not read time from {nc_file}: {e}")
                    continue
            
            return latest_time
            
        except Exception as e:
            print(f"Error getting latest file time for {satellite}: {e}")
            return None



class SwotWorkflow:
    """Complete SWOT data processing workflow"""
    
    def __init__(self, base_dir: str = str(pkg_path) + "/test_data" + "/swot"):
        self.processor = SwotDataProcessor(base_dir)
    
    def run_complete_workflow(
        self,
        short_name: str,
        data_type: List[str],
        timelims: Tuple[str, str],
        lonlims: Tuple[float, float],
        latlims: Tuple[float, float],
        version: Optional[str] = None,
        only_last: bool = True,
        smallbox: Optional[Tuple[List, List]] = None
    ) -> Dict[str, any]:
        """
        Run the complete SWOT data processing workflow.
        
        Args:
            short_name: Short name of the dataset to search for
            variables: Variables to extract and plot
            lon_range: Longitude range for subsetting
            lat_range: Latitude range for subsetting
            version: Data version to search for
            only_last: Download only latest version
            
        Returns:
            Dictionary with processing results
        """
        print("=== SWOT Data Processing Workflow ===")
        
        # Step 1: Authenticate
        print("\n1. Setting up authentication...")
        self.processor.authenticate()
        
        # Adjust time if only looking for new data
        try:
            if only_last:
                latest_time = self.processor.get_latest_file_time(short_name)
                print(latest_time)
                if latest_time:
                    latest_time += pd.Timedelta(minutes=30)
                    timelims = (latest_time.strftime("%Y-%m-%dT%H:%M:%S"), timelims[1])
        except Exception as e:
            print(f"Error adjusting time limits: {e}")
            print("Proceeding with original time limits")
        
        print(f"🔍 Searching SWOT data... {str(timelims[0])} to {str(timelims[1])}")
        try:
            results = self.processor.search_data(
                short_name=short_name,
                time_range=timelims,
                lon_range=lonlims,
                lat_range=latlims,
                version=version,
            )
        except Exception as e:
            print(f"Search failed: {e}")

        print(f"Found {len(results)} granules")

        print("📥 Downloading...")        
        try:
            raw_files = self.processor.download_data(results, data_type=data_type, only_last=only_last)
            print(f"Downloaded {len(raw_files)} files")
                
        except Exception as e:
            print(f"Data download failed: {e}")
            return {"success": False, "error": f"Download failed: {str(e)}"}
        
        # Step 3: Process and visualize
        print("✂ Processing and creating visualizations...")
        try:
            # only process new files
            # raw_files = list(Path(self.processor.fullnc_dir).glob("*.nc"))
            existing_files = list(Path(self.processor.nc_dir).glob("*.nc"))
            existing_file_names = {f.name for f in existing_files}
            new_files = [f for f in raw_files if f.name not in existing_file_names]
            subset_filenames, visualization_files = self.processor.process_and_visualize(
                                                        downloaded_files=new_files,
                                                        data_type=data_type,
                                                        lon_range=lonlims,
                                                        lat_range=latlims,
                                                    )
            
            if (subset_filenames is not None) and (smallbox is not None):
                print("\nCreating small visualizations...")
                smallbox_extent = [smallbox[0][0], smallbox[0][1], smallbox[1][0], smallbox[1][1]]
                for var in list(data_type.values())[0]:
                    png_files = self.processor.plot_datasets(
                        filenames=subset_filenames,
                        variable=var,
                        extent=smallbox_extent,
                        strap='_inner'
                    )
                    # visualization_files[var].extend(png_files)
            
            # Print summary
            total_plots = sum(len(visualization_files[ky]) for ky in visualization_files.keys())
            print(f"\n=== Workflow Complete ===")
            print(f"Downloaded {len(raw_files)} datasets")
            print(f"Generated {total_plots} visualization plots")
            print(f"Results saved to: {self.processor.base_dir}")
            
            return {
                "success": True,
                "downloaded_files": len(raw_files),
                "visualization_files": visualization_files,
                "total_plots": total_plots
            }
            
        except Exception as e:
            print(f"Processing failed: {e}")
            return {"success": False, "error": f"Processing failed: {str(e)}"}



# class SwotFileMonitor:
#     """SWOT file integrity monitoring and repair service"""
    
#     def __init__(self, processor: SwotDataProcessor):
#         """
#         Initialize file monitor
        
#         Args:
#             processor: SwotDataProcessor instance
#         """
#         self.processor = processor
#         self.base_dir = processor.base_dir
#         self.update_threshold_hours = 2  # Files older than 2 hours need updating
    
#     def check_file_status(self, nc_file_path: str) -> dict:
#         """
#         Check NC file status and determine if PNG regeneration is needed
        
#         Args:
#             nc_file_path: Path to the NC file to check
            
#         Returns:
#             Dictionary containing file status information
#         """
#         print("=== Checking SWOT File Status ===")
        
#         nc_path = Path(nc_file_path)
#         if not nc_path.exists():
#             print(f"✗ NC file not found: {nc_file_path}")
#             return {
#                 'nc_exists': False,
#                 'nc_modified_time': None,
#                 'png_count': 0,
#                 'needs_regeneration': True,
#                 'message': f"NC file not found: {nc_file_path}"
#             }
        
#         # Get NC file modification time
#         nc_modified_time = nc_path.stat().st_mtime
#         nc_modified_datetime = datetime.fromtimestamp(nc_modified_time)
        
#         print(f"✓ NC file found: {nc_file_path}")
#         print(f"✓ NC file modified: {nc_modified_datetime}")
        
#         # Check PNG files in the SWOT structure
#         png_dir = self.processor.get_png_path()
#         png_count = 0
#         oldest_png_time = float('inf')
        
#         if png_dir.exists():
#             png_files = list(png_dir.glob("*.png"))
#             png_count = len(png_files)
            
#             if png_files:
#                 # Find the oldest PNG file
#                 for png_file in png_files:
#                     png_time = png_file.stat().st_mtime
#                     if png_time < oldest_png_time:
#                         oldest_png_time = png_time
                
#                 oldest_png_datetime = datetime.fromtimestamp(oldest_png_time)
#                 print(f"✓ Found {png_count} PNG files")
#                 print(f"✓ Oldest PNG modified: {oldest_png_datetime}")
                
#                 # Check if NC file is newer than the oldest PNG
#                 needs_regeneration = nc_modified_time > oldest_png_time
#             else:
#                 needs_regeneration = True
#                 print("⚠ No PNG files found")
#         else:
#             needs_regeneration = True
#             print("⚠ PNG directory not found")
        
#         result = {
#             'nc_exists': True,
#             'nc_modified_time': nc_modified_datetime.isoformat(),
#             'png_count': png_count,
#             'needs_regeneration': needs_regeneration,
#             'message': f"NC file: {nc_modified_datetime}, PNG files: {png_count}, Regeneration needed: {needs_regeneration}"
#         }
        
#         print(f"Status: {result['message']}")
#         return result

#     def regenerate_all_pngs(self, nc_file_path: str, variable: str = 'ssha_filtered') -> dict:
#         """
#         Regenerate all PNG files from a single NC file
        
#         Args:
#             nc_file_path: Path to the NC file
#             variable: Variable to plot
            
#         Returns:
#             Dictionary with regeneration results
#         """
#         print("\n=== Starting SWOT PNG Regeneration ===")
        
#         nc_path = Path(nc_file_path)
#         if not nc_path.exists():
#             return {
#                 'success': False,
#                 'message': f"NC file not found: {nc_file_path}",
#                 'png_generated': 0
#             }
        
#         try:
#             # Create plots using processor
#             png_files = self.processor.plot_datasets([str(nc_path)], variable)
#             png_generated = len(png_files)
            
#             return {
#                 'success': True,
#                 'message': f"Successfully generated {png_generated} PNG files from {nc_file_path}",
#                 'png_generated': png_generated
#             }
                
#         except Exception as e:
#             return {
#                 'success': False,
#                 'message': f"Failed to regenerate PNGs: {str(e)}",
#                 'png_generated': 0
#             }


# def create_swot_processor(base_dir: str = "data") -> SwotDataProcessor:
#     """
#     Convenience function to create a SWOT processor
    
#     Args:
#         base_dir: Data directory
    
#     Returns:
#         SwotDataProcessor instance
#     """
#     return SwotDataProcessor(base_dir)


# def create_swot_workflow(base_dir: str = "data") -> SwotWorkflow:
#     """
#     Convenience function to create a SWOT workflow
    
#     Args:
#         base_dir: Data directory
    
#     Returns:
#         SwotWorkflow instance
#     """
#     return SwotWorkflow(base_dir)


# class SwotFileMonitor:
#     """File monitor for SWOT data files"""
    
#     def __init__(self, processor: SwotDataProcessor):
#         self.processor = processor
#         self.base_dir = processor.base_dir
    
#     def check_file_completeness(self, timelims: tuple, tstep: int = 3600) -> dict:
#         """
#         Check file completeness for SWOT data
        
#         Args:
#             timelims: (start_time, end_time) tuple
#             tstep: Time step in seconds
            
#         Returns:
#             Dictionary with file completeness results
#         """
#         from datetime import datetime, timedelta
        
#         start_time = datetime.fromisoformat(timelims[0].replace('Z', '+00:00'))
#         end_time = datetime.fromisoformat(timelims[1].replace('Z', '+00:00'))
        
#         print(f"=== Checking SWOT File Completeness ===")
#         print(f"Time range: {start_time} to {end_time}")
#         print(f"Time step: {tstep} seconds")
        
#         # Generate expected time steps
#         expected_times = []
#         current_time = start_time
#         while current_time <= end_time:
#             expected_times.append(current_time)
#             current_time += timedelta(seconds=tstep)
        
#         print(f"Expected time steps: {len(expected_times)}")
        
#         # Check NC files
#         nc_dir = self.base_dir / "swot" / "ssha" / "nc"
#         nc_existing = []
#         nc_missing = []
#         nc_corrupted = []
        
#         if nc_dir.exists():
#             existing_files = list(nc_dir.glob("*.nc"))
#             existing_times = set()
            
#             for file_path in existing_files:
#                 try:
#                     # Extract time from filename (format: YYYYMMDD_HHMMSS.nc)
#                     filename = file_path.stem
#                     if len(filename) >= 15:  # YYYYMMDD_HHMMSS
#                         time_str = filename[:15]  # YYYYMMDD_HHMMSS
#                         file_time = datetime.strptime(time_str, "%Y%m%d_%H%M%S")
#                         existing_times.add(file_time)
#                         nc_existing.append(filename + ".nc")
#                 except Exception as e:
#                     print(f"Error parsing file {file_path.name}: {e}")
#                     nc_corrupted.append(file_path.name)
#         else:
#             print("NC directory does not exist")
        
#         # Check PNG files
#         png_dir = self.base_dir / "swot" / "ssha" / "png"
#         png_existing = []
#         png_missing = []
        
#         if png_dir.exists():
#             existing_png_files = list(png_dir.glob("*.png"))
#             for file_path in existing_png_files:
#                 png_existing.append(file_path.name)
#         else:
#             print("PNG directory does not exist")
        
#         # Find missing files
#         for expected_time in expected_times:
#             time_str = expected_time.strftime("%Y%m%d_%H%M%S")
#             nc_filename = f"{time_str}.nc"
#             png_filename = f"{time_str}.png"
            
#             if expected_time not in existing_times:
#                 nc_missing.append(nc_filename)
            
#             if png_filename not in png_existing:
#                 png_missing.append(png_filename)
        
#         # Print results
#         print(f"\nChecking NC files...")
#         for missing_file in nc_missing:
#             print(f"✗ {missing_file} - Missing")
        
#         print(f"\nChecking PNG files...")
#         for missing_file in png_missing:
#             print(f"✗ {missing_file} - Missing")
        
#         # Summary
#         total_expected = len(expected_times)
#         nc_completion_rate = (len(nc_existing) / total_expected) * 100 if total_expected > 0 else 0
#         png_completion_rate = (len(png_existing) / total_expected) * 100 if total_expected > 0 else 0
        
#         print(f"\n=== File Completeness Summary ===")
#         print(f"Total expected files: {total_expected}")
#         print(f"NC files - Existing: {len(nc_existing)}, Missing: {len(nc_missing)}, Corrupted: {len(nc_corrupted)}")
#         print(f"NC completion rate: {nc_completion_rate:.1f}%")
#         print(f"PNG files - Existing: {len(png_existing)}, Missing: {len(png_missing)}")
#         print(f"PNG completion rate: {png_completion_rate:.1f}%")
        
#         return {
#             'nc_files': {
#                 'existing': nc_existing,
#                 'missing': nc_missing,
#                 'corrupted': nc_corrupted
#             },
#             'png_files': {
#                 'existing': png_existing,
#                 'missing': png_missing
#             },
#             'summary': {
#                 'total_expected': total_expected,
#                 'nc_completion_rate': nc_completion_rate,
#                 'png_completion_rate': png_completion_rate
#             }
#         }
    
#     def check_file_status(self, nc_file_path: str) -> dict:
#         """Check status of a specific NC file"""
#         return self.processor.check_file_status(nc_file_path)
    
#     def regenerate_all_pngs(self, nc_file_path: str, variable: str = "ssha_filtered") -> dict:
#         """Regenerate all PNG files from an NC file"""
#         return self.processor.regenerate_all_pngs(nc_file_path, variable)


# class SwotFileMonitor:
#     """File monitor for SWOT data files"""
    
#     def __init__(self, processor: SwotDataProcessor):
#         self.processor = processor
#         self.base_dir = processor.base_dir
    
#     def check_file_completeness(self, timelims: tuple, tstep: int = 3600) -> Dict[str, Any]:
#         """
#         Check file completeness for SWOT data.
#         SWOT files are not generated on a regular time schedule like Himawari.
#         Instead, we check for actual SWOT files and their corresponding PNG files.
        
#         Args:
#             timelims: (start_time, end_time) tuple (not used for SWOT)
#             tstep: Time step in seconds (not used for SWOT)
            
#         Returns:
#             Dictionary with file completeness results
#         """
#         print(f"=== Checking SWOT File Completeness ===")
#         print(f"SWOT files are orbit-based, not time-based like other satellites")
        
#         # Check NC files
#         nc_dir = self.base_dir / "swot" / "ssha" / "nc"
#         nc_existing = []
#         nc_missing = []
#         nc_corrupted = []
        
#         if nc_dir.exists():
#             existing_nc_files = list(nc_dir.glob("*.nc"))
#             for file_path in existing_nc_files:
#                 try:
#                     # Check if file is valid by trying to open it
#                     import xarray as xr
#                     with xr.open_dataset(file_path) as ds:
#                         # If we can open it, it's valid
#                         nc_existing.append(file_path.name)
#                 except Exception as e:
#                     print(f"⚠ {file_path.name} - Corrupted: {e}")
#                     nc_corrupted.append(file_path.name)
#         else:
#             print("NC directory does not exist")
        
#         # Check PNG files
#         png_dir = self.base_dir / "swot" / "ssha" / "png"
#         png_existing = []
#         png_missing = []
        
#         if png_dir.exists():
#             existing_png_files = list(png_dir.glob("*.png"))
#             for file_path in existing_png_files:
#                 png_existing.append(file_path.name)
#         else:
#             print("PNG directory does not exist")
        
#         # For SWOT, we expect each NC file to have a corresponding PNG file
#         # Check for missing PNG files
#         for nc_file in nc_existing:
#             # Extract base name and look for corresponding PNG
#             base_name = nc_file.replace('.nc', '')
#             # SWOT PNG files might have different naming, so we check if any PNG exists
#             # that could correspond to this NC file
#             png_found = False
#             for png_file in png_existing:
#                 if base_name in png_file or png_file.replace('.png', '') in base_name:
#                     png_found = True
#                     break
            
#             if not png_found:
#                 png_missing.append(f"{base_name}.png")
        
#         # Print results
#         print(f"\nChecking NC files...")
#         for existing_file in nc_existing:
#             print(f"✓ {existing_file} - OK")
        
#         for corrupted_file in nc_corrupted:
#             print(f"⚠ {corrupted_file} - Corrupted")
        
#         print(f"\nChecking PNG files...")
#         for existing_file in png_existing:
#             print(f"✓ {existing_file} - OK")
        
#         for missing_file in png_missing:
#             print(f"✗ {missing_file} - Missing")
        
#         # Summary
#         total_nc_files = len(nc_existing) + len(nc_corrupted)
#         total_png_files = len(png_existing) + len(png_missing)
        
#         print(f"\n=== File Completeness Summary ===")
#         print(f"Total NC files: {total_nc_files}")
#         print(f"NC files - Existing: {len(nc_existing)}, Missing: 0, Corrupted: {len(nc_corrupted)}")
#         print(f"Total PNG files: {total_png_files}")
#         print(f"PNG files - Existing: {len(png_existing)}, Missing: {len(png_missing)}")
        
#         return {
#             "nc_files": {
#                 "existing": nc_existing,
#                 "missing": nc_missing,
#                 "corrupted": nc_corrupted
#             },
#             "png_files": {
#                 "existing": png_existing,
#                 "missing": png_missing
#             },
#             "summary": {
#                 "total_expected": total_nc_files,  # Based on actual files, not time steps
#                 "nc_completion_rate": 100.0 if len(nc_corrupted) == 0 else (len(nc_existing) / total_nc_files * 100),
#                 "png_completion_rate": (len(png_existing) / total_png_files * 100) if total_png_files > 0 else 0
#             }
#         }
    
#     def check_file_status(self, nc_file_path: str) -> Dict[str, Any]:
#         """Check status of a specific NC file"""
#         # Implementation for checking individual file status
#         return {
#             "nc_exists": True,
#             "nc_modified_time": "2025-02-26T14:54:17",
#             "png_count": 1,
#             "needs_regeneration": False,
#             "message": "File status OK"
#         }
    
#     def regenerate_all_pngs(self, nc_file_path: str, variable: str = "ssha_filtered") -> Dict[str, Any]:
#         """Regenerate all PNG files from an NC file"""
#         # Implementation for regenerating PNG files
#         return {
#             "success": True,
#             "message": "PNG files regenerated successfully",
#             "png_generated": 1
#         }


# def create_file_monitor(base_dir: str = "data") -> SwotFileMonitor:
#     """
#     Convenience function to create a file monitor
    
#     Args:
#         base_dir: Data directory
    
#     Returns:
#         SwotFileMonitor instance
#     """
#     processor = SwotDataProcessor(base_dir)
#     return SwotFileMonitor(processor)


def run_swot_example():
    """Run an example SWOT data processing workflow"""
    
    # Configuration parameters - Ningaloo region example
    ftp_path = '/swot_products/l3_karin_nadir/l3_lr_ssh/v2_0_1/Expert/'
    level = "L3"
    variant = "Expert"
    cycle_numbers = [29]
    half_orbits = [62]
    variables = ['time', 'ssha_filtered']
    lon_range = (111, 114)  # Ningaloo region
    lat_range = (-25, -20)
    
    # Create workflow instance
    workflow = SwotWorkflow()
    
    # Run complete workflow
    result = workflow.run_complete_workflow(
        ftp_path=ftp_path,
        level=level,
        variant=variant,
        cycle_numbers=cycle_numbers,
        half_orbits=half_orbits,
        variables=variables,
        lon_range=lon_range,
        lat_range=lat_range
    )
    
    print(f"\nWorkflow result: {result}")


if __name__ == "__main__":
    # Run example when script is executed directly
    run_swot_example()
