# src/data_acquisition/downloader.py
import argparse
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import the three clients
from src.data_acquisition.usgs_client import USGSClient
from src.data_acquisition.noaa_client import NOAAClient
from src.data_acquisition.nasa_client import NASAClient

load_dotenv()

def download_usgs(sample: bool = True):
    """Download USGS water level data for selected sites."""
    client = USGSClient()
    
    # Example sites relevant to Jersey City / northern NJ
    sites = ["01388500"]  # Passaic River at Pine Brook (already tested)
    if not sample:
        sites.extend(["01389500", "01378500"])  # Add Rockaway and Hackensack

    print(f"Downloading USGS data for sites: {sites}")
    
    for site in sites:
        client.get_instantaneous_values(
            sites=site,
            parameter_cd="00065",  # Gage height in feet
            start_date="2025-01-01",
            end_date=datetime.now().strftime("%Y-%m-%d"),
            save_to_file=True
        )

def download_noaa(sample: bool = True):
    """Download NOAA GHCN daily climate data."""
    client = NOAAClient()
    
    # Newark Liberty Airport - closest major station to Jersey City
    station = "GHCND:USW00014734"
    
    print(f"Downloading NOAA data for station: {station}")
    
    client.get_daily_data(
        stationid=station,
        datatypeids=["TMAX", "TMIN", "PRCP", "AWND"],
        startdate="2025-01-01",
        enddate=datetime.now().strftime("%Y-%m-%d"),
        save_to_file=True
    )

def download_nasa(sample: bool = True):
    """Download NASA precipitation data (GPM IMERG daily)."""
    client = NASAClient()
    
    count = 2 if sample else 10
    
    print(f"Downloading {count} NASA GPM IMERG granules...")
    
    files = client.search_and_download(
        short_name="GPM_3IMERGDF",
        version="07",
        temporal=("2025-01-01", "2025-01-31"),  # Short range for testing
        count=count
    )
    
    return files

def download_all(sample: bool = True):
    """Main function to download from all sources."""
    raw_dir = os.getenv("DATA_RAW_DIR", "data/raw")
    os.makedirs(raw_dir, exist_ok=True)
    print(f"Starting data download (sample mode: {sample}) to: {raw_dir}")
    
    download_usgs(sample)
    download_noaa(sample)
    nasa_files = download_nasa(sample)
    
    print("\nDownload complete!")
    print(f"NASA files downloaded: {len(nasa_files)}")
    if nasa_files:
        print("Sample NASA file:", nasa_files[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified downloader for USGS, NOAA, NASA data")
    parser.add_argument("--sample", action="store_true", help="Download small sample (faster for testing)")
    parser.add_argument("--only", choices=["usgs", "noaa", "nasa"], help="Download from one source only")
    
    args = parser.parse_args()
    
    if args.only == "usgs":
        download_usgs(args.sample)
    elif args.only == "noaa":
        download_noaa(args.sample)
    elif args.only == "nasa":
        download_nasa(args.sample)
    else:
        download_all(args.sample)