# src/data_acquisition/usgs_client.py
import requests
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Dict, Optional, List, Union
import os
import json
from dotenv import load_dotenv
from diskcache import Cache

# Import shared utilities
from .utils import logger, get_data_path, parse_date

load_dotenv()

class USGSClient:
    """
    Client for USGS NWIS (National Water Information System) API.
    Fetches instantaneous values (e.g., gage height / water level).
    Uses legacy waterservices.usgs.gov endpoint (still active in 2026).
    """
    BASE_URL_IV = "https://waterservices.usgs.gov/nwis/iv/"
    BASE_URL_DV = "https://waterservices.usgs.gov/nwis/dv/"  # Daily values (not used yet)

    def __init__(self):
        # Create session with polite User-Agent
        self.session = requests.Session()
        user_agent_email = os.getenv("USER_AGENT_EMAIL", "your.email@example.com")
        user_agent = f"GPU-Climate-Forecasting-System/1.0 ({user_agent_email})"
        self.session.headers.update({"User-Agent": user_agent})

        # NOTE: Legacy IV endpoint does NOT use token → we ignore it even if present
        self.api_token = os.getenv("USGS_API_TOKEN")  # kept for future/newer API

        # Persistent cache (expires after 24 hours)
        cache_dir = get_data_path("cache/usgs")
        self.cache = Cache(cache_dir)
        logger.info(f"USGS cache initialized at: {cache_dir}")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
    def get_instantaneous_values(
        self,
        sites: Union[str, List[str]],
        parameter_cd: str = "00065",           # Gage height, feet (water level)
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        format: str = "json",
        save_to_file: bool = True,
        use_cache: bool = True
    ) -> Dict:
        """
        Fetch instantaneous water level / gage height data.

        Args:
            sites: Single site number (str) or list of site numbers
            parameter_cd: USGS parameter code (default "00065" = gage height ft)
            start_date, end_date: YYYY-MM-DD (defaults to last 365 days)
            format: "json" (default) or "rdb", "xml"
            save_to_file: Save raw JSON to data/raw/usgs/
            use_cache: Check/use disk cache first

        Returns:
            Parsed JSON response from USGS
        """
        # Normalize sites to comma-separated string
        if isinstance(sites, list):
            sites = ",".join(str(s) for s in sites)
        else:
            sites = str(sites)

        # Default date range: last year
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Create unique cache key
        cache_key = f"iv_{sites}_{parameter_cd}_{start_date}_{end_date}_{format}"

        # Check cache first
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.info(f"USGS cache hit for key: {cache_key}")
                return cached_data

        # Build query parameters (NO TOKEN HERE - legacy endpoint is public)
        params = {
            "format": format,
            "sites": sites,
            "parameterCd": parameter_cd,
            "startDT": start_date,
            "endDT": end_date,
            "siteStatus": "all"
        }

        logger.info(f"Fetching USGS IV data: sites={sites}, param={parameter_cd}, {start_date} to {end_date}")
        logger.debug(f"Request URL: {self.BASE_URL_IV}?{requests.compat.urlencode(params)}")

        try:
            response = self.session.get(self.BASE_URL_IV, params=params, timeout=45)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Log full response body for debugging
            error_text = response.text[:500]  # first 500 chars to avoid flooding log
            logger.error(f"USGS API error {response.status_code}: {error_text}")
            logger.error(f"Full URL: {response.url}")
            raise

        data = response.json()

        # Cache the result
        self.cache.set(cache_key, data, expire=86400)  # 24 hours
        logger.info(f"USGS response cached for key: {cache_key}")

        # Save raw data to disk
        if save_to_file:
            raw_dir = get_data_path("usgs")
            safe_sites = sites.replace(",", "_")
            filename = f"{safe_sites}_{start_date}_to_{end_date}.json"
            file_path = os.path.join(raw_dir, filename)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved raw USGS data to: {file_path}")

        return data

    def get_site_info(self, sites: Union[str, List[str]]) -> Dict:
        """
        Optional helper: Get metadata about sites (name, location, etc.)
        """
        if isinstance(sites, list):
            sites = ",".join(str(s) for s in sites)

        params = {"format": "rdb", "sites": sites}
        response = self.session.get("https://waterservices.usgs.gov/nwis/site/", params=params)
        response.raise_for_status()

        # RDB is tab-delimited; simple parse for demo
        lines = response.text.splitlines()
        data = {}
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) > 2 and parts[0].isdigit():
                site_no = parts[0]
                data[site_no] = {
                    "agency_cd": parts[1],
                    "site_no": parts[2],
                    "station_nm": parts[3] if len(parts) > 3 else "",
                    # Add more fields as needed
                }
        return data


# Quick test / CLI usage (run file directly)
if __name__ == "__main__":
    import sys

    client = USGSClient()

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Example test fetch
        try:
            data = client.get_instantaneous_values(
                sites="01388500",           # Passaic River at Pine Brook, NJ
                parameter_cd="00065",
                start_date="2025-01-01",
                end_date="2025-02-01",
                save_to_file=True,
                use_cache=True
            )
            print("Fetched keys:", list(data.keys()))
            if 'value' in data and data['value'].get('timeSeries'):
                readings = data['value']['timeSeries'][0]['values'][0]['value']
                print("Number of readings:", len(readings))
                if readings:
                    print("Sample reading:", readings[0])
        except Exception as e:
            print("Error during test fetch:", str(e))
    else:
        print("USGSClient ready. Use get_instantaneous_values() method.")