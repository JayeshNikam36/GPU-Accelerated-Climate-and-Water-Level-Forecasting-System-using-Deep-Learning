# src/data_acquisition/noaa_client.py
import requests
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
import os
import json
from dotenv import load_dotenv

load_dotenv()

class NOAAClient:
    BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"

    def __init__(self):
        self.token = os.getenv("NOAA_TOKEN")
        if not self.token:
            raise ValueError("NOAA_TOKEN not found in .env")
        self.headers = {"token": self.token}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=30))
    def get_daily_data(
        self,
        datasetid: str = "GHCND",
        stationid: str = "GHCND:USW00014734",  # Newark Liberty Intl Airport (near Jersey City)
        datatypeids: list = ["TMAX", "TMIN", "PRCP", "AWND"],  # List instead of str
        startdate: str = None,
        enddate: str = None,
        limit: int = 1000,
        units: str = "standard",
        save_to_file: bool = True
    ) -> list:  # Return list of all results (paginated)
        if not startdate:
            startdate = (datetime.now() - timedelta(days=365*5)).strftime("%Y-%m-%d")  # 5 years
        if not enddate:
            enddate = datetime.now().strftime("%Y-%m-%d")

        all_results = []
        offset = 0

        while True:
            params = {
                "datasetid": datasetid,
                "stationid": stationid,
                "startdate": startdate,
                "enddate": enddate,
                "limit": limit,
                "offset": offset,
                "units": units
            }
            # Repeat datatypeid for each
            for dt in datatypeids:
                params["datatypeid"] = dt  # One per request (API requires separate)

            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            all_results.extend(results)

            # Pagination check
            metadata = data.get("metadata", {}).get("resultset", {})
            total = metadata.get("count", 0)
            if offset + limit >= total:
                break
            offset += limit

        if save_to_file:
            raw_dir = os.getenv("DATA_RAW_DIR", "data/raw")
            os.makedirs(f"{raw_dir}/noaa", exist_ok=True)
            file_path = f"{raw_dir}/noaa/{stationid.split(':')[-1]}_{startdate}_to_{enddate}.json"
            with open(file_path, "w") as f:
                json.dump({"results": all_results}, f, indent=2)
            print(f"Saved NOAA data ({len(all_results)} records) to: {file_path}")

        return all_results