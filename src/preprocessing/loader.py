# src/preprocessing/loader.py
import cudf
import cupy as cp
import json
import os
import xarray as xr
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
from src.data_acquisition.utils import logger, get_data_path

class DataLoader:
    """
    Loads raw data from USGS, NOAA, NASA into GPU-resident cuDF DataFrames.
    Handles JSON and NetCDF formats.
    """

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and cp.cuda.runtime.getDeviceCount() > 0
        if self.use_gpu:
            logger.info(f"GPU acceleration enabled: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
        else:
            logger.warning("No GPU detected or disabled — falling back to CPU (Pandas)")

    def load_usgs(self, file_path: str) -> cudf.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"USGS file not found: {file_path}")

        with open(file_path, "r") as f:
            raw = json.load(f)

        try:
            ts = raw['value']['timeSeries'][0]['values'][0]['value']
            df_pd = pd.DataFrame(ts)
            df_pd['dateTime'] = pd.to_datetime(df_pd['dateTime'], utc=True).dt.tz_localize(None)
            df_pd['value'] = pd.to_numeric(df_pd['value'], errors='coerce')
            df_pd = df_pd[['dateTime', 'value']].rename(columns={'dateTime': 'timestamp', 'value': 'gage_height_ft'})

            if self.use_gpu:
                df = cudf.from_pandas(df_pd)
            else:
                df = df_pd

            logger.info(f"Loaded USGS data: {len(df)} rows from {file_path}")
            return df
        except KeyError as e:
            logger.error(f"Invalid USGS JSON structure: {e}")
            raise

    def load_noaa(self, file_path: str) -> cudf.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"NOAA file not found: {file_path}")

        with open(file_path, "r") as f:
            raw = json.load(f)

        try:
            df_pd = pd.DataFrame(raw.get('results', []))
            df_pd['date'] = pd.to_datetime(df_pd['date'])
            df_pivot = df_pd.pivot_table(
                index='date',
                columns='datatype',
                values='value',
                aggfunc='first'
            ).reset_index()
            df_pivot = df_pivot.rename(columns={'date': 'timestamp'})

            if self.use_gpu:
                df = cudf.from_pandas(df_pivot)
            else:
                df = df_pivot

            logger.info(f"Loaded NOAA data: {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading NOAA data: {e}")
            raise

    def load_nasa_precip(self, file_path: str, lat: float = 40.728, lon: float = -74.078) -> cudf.DataFrame:
        try:
            ds = xr.open_dataset(file_path)
            precip = ds['precipitation']  # Confirmed correct variable for GPM_3IMERGDF V07

            nearest = precip.sel(lat=lat, lon=lon, method='nearest')

            df_pd = nearest.to_dataframe().reset_index()
            df_pd = df_pd[['time', 'precipitation']].rename(columns={'time': 'timestamp', 'precipitation': 'precip_mm'})

            if self.use_gpu:
                df = cudf.from_pandas(df_pd)
            else:
                df = df_pd

            logger.info(f"Loaded NASA precip data: {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading NASA NetCDF: {e}")
            if 'ds' in locals():
                logger.info(f"Available variables: {list(ds.variables)}")
            raise
    def load_nasa_multi(self, folder_path: str, lat: float = 40.728, lon: float = -74.078) -> cudf.DataFrame:
        """Load and concatenate all .nc4 files in a folder."""
        import glob
        files = glob.glob(os.path.join(folder_path, "*.nc4"))
        if not files:
            raise FileNotFoundError(f"No .nc4 files in {folder_path}")

        dfs = []
        for f in files:
            df_day = self.load_nasa_precip(f, lat, lon)
            dfs.append(df_day)

        combined = cudf.concat(dfs).sort_values('timestamp').reset_index(drop=True)
        logger.info(f"Combined {len(combined)} rows from {len(files)} NASA files")
        return combined

    def load_all(self, usgs_file: str, noaa_file: str = None, nasa_file: str = None) -> Dict[str, cudf.DataFrame]:
        """Load all sources and return dict of DataFrames. 
        Skips sources with None path."""
        result = {}
        
        if usgs_file:
            result['usgs'] = self.load_usgs(usgs_file)
        else:
            logger.warning("No USGS file provided — skipping")
        
        if noaa_file:
            result['noaa'] = self.load_noaa(noaa_file)
        else:
            logger.info("No NOAA file provided — skipping")
        
        if nasa_file:
            result['nasa'] = self.load_nasa_precip(nasa_file)
        else:
            logger.info("No NASA file provided — skipping")
        
        return result
    
    def merge_sources(
    self,
    usgs_df: cudf.DataFrame,
    noaa_df: Optional[cudf.DataFrame] = None,
    nasa_df: Optional[cudf.DataFrame] = None
    ) -> cudf.DataFrame:
        """
        Merge NOAA daily climate and NASA precip into high-frequency USGS data.
        Uses forward-fill (ffill) to align timestamps.
        """
        merged = usgs_df.copy()

        if noaa_df is not None and not noaa_df.empty:
            # Reindex NOAA to USGS timestamps (ffill daily values)
            noaa_reindexed = noaa_df.set_index('timestamp').reindex(
                merged['timestamp'], method='ffill'
            ).reset_index()
            merged = merged.merge(noaa_reindexed, on='timestamp', how='left', suffixes=('', '_noaa'))

        if nasa_df is not None and not nasa_df.empty:
            nasa_reindexed = nasa_df.set_index('timestamp').reindex(
                merged['timestamp'], method='ffill'
            ).reset_index()
            merged = merged.merge(nasa_reindexed, on='timestamp', how='left', suffixes=('', '_nasa'))

        # Fill any remaining NaNs (e.g., before first daily value)
        merged = merged.fillna(method='ffill').fillna(0)

        logger.info(f"Merged DataFrame shape: {merged.shape}")
        return merged

    def merge_sources(
            self,
            usgs_df: cudf.DataFrame,
            noaa_df: Optional[cudf.DataFrame] = None,
            nasa_df: Optional[cudf.DataFrame] = None,
            timestamp_col: str = 'timestamp'
        ) -> cudf.DataFrame:
        """
        Merge daily NOAA and NASA data into high-frequency USGS data using forward-fill.
        Uses Pandas for ffill (cuDF limitation) then back to GPU.
        """
        # Convert to Pandas for ffill/reindex (cuDF doesn't support it yet)
        usgs_pd = usgs_df.to_pandas()
        merged_pd = usgs_pd.copy()

        if noaa_df is not None and not noaa_df.empty:
            logger.info("Merging NOAA climate data (forward-fill)...")
            noaa_pd = noaa_df.to_pandas()
            noaa_pd = noaa_pd.set_index(timestamp_col)
            # Reindex and ffill
            noaa_reindexed = noaa_pd.reindex(usgs_pd[timestamp_col], method='ffill').reset_index()
            merged_pd = merged_pd.merge(noaa_reindexed, on=timestamp_col, how='left', suffixes=('', '_noaa'))

        if nasa_df is not None and not nasa_df.empty:
            logger.info("Merging NASA precipitation data (forward-fill)...")
            nasa_pd = nasa_df.to_pandas()
            nasa_pd = nasa_pd.set_index(timestamp_col)
            nasa_reindexed = nasa_pd.reindex(usgs_pd[timestamp_col], method='ffill').reset_index()
            merged_pd = merged_pd.merge(nasa_reindexed, on=timestamp_col, how='left', suffixes=('', '_nasa'))

        # Fill any remaining NaNs
        merged_pd = merged_pd.fillna(method='ffill').fillna(0)

        # Back to cuDF
        merged = cudf.from_pandas(merged_pd)

        logger.info(f"Merged DataFrame shape: {merged.shape}")
        logger.info(f"Merged columns: {list(merged.columns)}")
        return merged