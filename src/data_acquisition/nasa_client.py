# src/data_acquisition/nasa_client.py
import earthaccess
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from typing import List, Tuple

load_dotenv()

class NASAClient:
    def __init__(self):
        # Login persists credentials in ~/.netrc
        earthaccess.login(persist=True)
        print("NASA Earthdata login successful (or already persisted).")

    def search_and_download(
        self,
        short_name: str = "GPM_3IMERGDF",  # Daily precipitation (IMERG Final V07)
        version: str = "07",               # Current version in 2026
        temporal: Tuple[str, str] = None,
        bounding_box: Tuple[float, float, float, float] = (-74.3, 40.6, -73.9, 41.0),  # Jersey City approx bbox (lon_min, lat_min, lon_max, lat_max)
        count: int = 5,                    # Limit results (small for testing)
        download_dir: str = None
    ) -> List[str]:
        """
        Search and download NASA granules (e.g., daily precip NetCDF/HDF5 files).
        - short_name: e.g., "GPM_3IMERGDF" for daily precip, "SPL3SMP_E" for SMAP soil moisture
        - temporal: (start_date, end_date) in 'YYYY-MM-DD'
        - bounding_box: (west, south, east, north) in degrees
        """
        if not temporal:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            temporal = (start_date, end_date)

        print(f"Searching NASA CMR for {short_name} ({temporal[0]} to {temporal[1]})...")

        results = earthaccess.search_data(
            short_name=short_name,
            version=version,
            temporal=temporal,
            bounding_box=bounding_box,
            count=count
        )

        if not results:
            print("No granules found for the query.")
            return []

        download_dir = download_dir or os.path.join(os.getenv("DATA_RAW_DIR", "data/raw"), "nasa", short_name)
        os.makedirs(download_dir, exist_ok=True)

        print(f"Downloading {len(results)} granules to {download_dir}...")
        downloaded_files = earthaccess.download(results, download_dir)

        file_paths = [str(f) for f in downloaded_files if f is not None]
        print(f"Downloaded {len(file_paths)} files successfully.")

        return file_paths