# scripts/data/validate_data.py
import argparse
import glob
import os
from src.data_acquisition.validator import DataValidator
from src.data_acquisition.utils import logger

def main():
    parser = argparse.ArgumentParser(description="Validate raw data files")
    parser.add_argument("--source", choices=["usgs", "noaa", "nasa", "all"], default="all",
                        help="Source to validate (or all)")
    parser.add_argument("--file", help="Specific file path to validate")
    
    args = parser.parse_args()
    
    raw_dir = os.getenv("DATA_RAW_DIR", "data/raw")
    
    if args.file:
        files = [args.file]
    elif args.source == "all":
        files = glob.glob(f"{raw_dir}/**/*.json", recursive=True) + glob.glob(f"{raw_dir}/**/*.nc4", recursive=True)
    else:
        files = glob.glob(f"{raw_dir}/{args.source}/**/*", recursive=True)
    
    for file_path in files:
        if not os.path.isfile(file_path):
            continue
        source = "usgs" if "usgs" in file_path else "noaa" if "noaa" in file_path else "nasa"
        validator = DataValidator(source=source)
        try:
            results = validator.validate_file(file_path)
            print(f"\nValidation for {file_path}:")
            print(results)
        except Exception as e:
            logger.error(f"Failed to validate {file_path}: {e}")

if __name__ == "__main__":
    main()