# test_env.py or test_loader.py
from src.preprocessing.loader import DataLoader
from src.data_acquisition.utils import logger
import logging

logging.basicConfig(level=logging.INFO)

loader = DataLoader(use_gpu=True)

usgs_file = "data/raw/usgs/01388500_2025-01-01_to_2026-01-20.json"
noaa_file = "data/raw/noaa/USW00014734_2025-01-01_to_2026-01-20.json"
nasa_file = "data/raw/nasa/GPM_3IMERGDF/3B-DAY.MS.MRG.3IMERG.20250101-S000000-E235959.V07B.nc4"  # your file

print("Starting GPU data loading...")

dfs = loader.load_all(usgs_file, noaa_file, nasa_file)

print("\nLoading successful!")
print("USGS shape:", dfs['usgs'].shape)
print("USGS columns:", list(dfs['usgs'].columns))
print("NOAA shape:", dfs['noaa'].shape)
print("NOAA columns:", list(dfs['noaa'].columns))
if dfs['nasa'] is not None:
    print("NASA shape:", dfs['nasa'].shape)
    print("NASA columns:", list(dfs['nasa'].columns))

print("\nFirst 5 rows of USGS on GPU:")
print(dfs['usgs'].head())