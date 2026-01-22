import cudf
import cupy as cp
import numpy as np
from datetime import datetime
from src.data_acquisition.utils import logger

def add_temporal_features(df: cudf.DataFrame, timestamp_col: str = 'timestamp') -> cudf.DataFrame:
    """
    Add cyclical time features (hour, day, month sin/cos) on GPU.
    """
    df = df.copy()
    df[timestamp_col] = cudf.to_datetime(df[timestamp_col])

    # Extract components
    df['hour'] = df[timestamp_col].dt.hour
    df['dayofweek'] = df[timestamp_col].dt.dayofweek
    df['month'] = df[timestamp_col].dt.month

    # Cyclical encoding
    df['hour_sin'] = cp.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = cp.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = cp.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos'] = cp.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = cp.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = cp.cos(2 * np.pi * df['month'] / 12)

    logger.info("Added cyclical time features")
    return df

def add_lags_and_rolling(
    df: cudf.DataFrame,
    target_col: str = 'gage_height_ft',
    lags: list = [1, 6, 12, 24],
    windows: list = [3, 6, 12, 24],
    min_periods: int = 1
) -> cudf.DataFrame:
    """
    Add lag features and rolling statistics on GPU.
    """
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Lags
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

    # Rolling stats
    for w in windows:
        rolling = df[target_col].rolling(window=w, min_periods=min_periods)
        df[f'{target_col}_rolling_mean_{w}'] = rolling.mean()
        df[f'{target_col}_rolling_std_{w}'] = rolling.std()
        df[f'{target_col}_rolling_max_{w}'] = rolling.max()
        df[f'{target_col}_rolling_min_{w}'] = rolling.min()

    logger.info(f"Added {len(lags)} lags and {len(windows)} rolling windows")
    return df.dropna()  # Drop rows with NaN from shifting/rolling

def merge_external_forcings(
    usgs_df: cudf.DataFrame,
    noaa_df: cudf.DataFrame,
    nasa_df: cudf.DataFrame,
    timestamp_col: str = 'timestamp'
) -> cudf.DataFrame:
    """
    Merge NOAA daily climate + NASA precip into USGS high-frequency data (forward fill).
    """
    # Resample NOAA/NASA to match USGS timestamps (forward fill)
    noaa_df = noaa_df.set_index(timestamp_col).reindex(usgs_df[timestamp_col], method='ffill').reset_index()
    nasa_df = nasa_df.set_index(timestamp_col).reindex(usgs_df[timestamp_col], method='ffill').reset_index()

    # Merge
    df = usgs_df.merge(noaa_df, on=timestamp_col, how='left', suffixes=('', '_noaa'))
    df = df.merge(nasa_df, on=timestamp_col, how='left', suffixes=('', '_nasa'))

    logger.info("Merged external forcings (NOAA + NASA) into USGS data")
    return df