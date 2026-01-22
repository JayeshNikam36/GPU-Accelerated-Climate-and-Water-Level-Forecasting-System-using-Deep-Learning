# scripts/data/download_data.py
import argparse
from datetime import datetime
from src.data_acquisition.downloader import download_all
from src.data_acquisition.usgs_client import USGSClient
from src.data_acquisition.noaa_client import NOAAClient
from src.data_acquisition.nasa_client import NASAClient
from src.data_acquisition.utils import logger

def main():
    parser = argparse.ArgumentParser(
        description="Download climate and water level data from USGS, NOAA, NASA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--source", choices=["usgs", "noaa", "nasa", "all"],
                        default="all", help="Data source(s) to download")
    
    parser.add_argument("--sample", action="store_true",
                        help="Download small sample (faster for testing)")
    
    parser.add_argument("--usgs-sites", nargs="+", default=["01388500"],
                        help="USGS site numbers (space-separated)")
    
    parser.add_argument("--start-date", type=str,
                        default="2025-01-01", help="Start date YYYY-MM-DD")
    
    parser.add_argument("--end-date", type=str,
                        default=datetime.now().strftime("%Y-%m-%d"),
                        help="End date YYYY-MM-DD")
    
    parser.add_argument("--noaa-station", default="GHCND:USW00014734",
                        help="NOAA GHCND station ID")
    
    args = parser.parse_args()

    if args.source == "all":
        download_all(sample=args.sample)
    elif args.source == "usgs":
        client = USGSClient()
        for site in args.usgs_sites:
            logger.info(f"Downloading USGS for site {site}...")
            client.get_instantaneous_values(
                sites=site,
                start_date=args.start_date,
                end_date=args.end_date
            )
    elif args.source == "noaa":
        client = NOAAClient()
        logger.info(f"Downloading NOAA for station {args.noaa_station}...")
        client.get_daily_data(
            stationid=args.noaa_station,
            startdate=args.start_date,
            enddate=args.end_date
        )
    elif args.source == "nasa":
        client = NASAClient()
        count = 2 if args.sample else 10
        logger.info(f"Downloading {count} NASA GPM granules...")
        client.search_and_download(
            short_name="GPM_3IMERGDF",
            temporal=(args.start_date, args.end_date),
            count=count
        )

    logger.info("Download finished.")

if __name__ == "__main__":
    main()