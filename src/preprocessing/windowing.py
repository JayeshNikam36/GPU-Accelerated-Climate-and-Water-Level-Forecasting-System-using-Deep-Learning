# src/preprocessing/windowing.py
import cudf
import cupy as cp
import numpy as np
from typing import Tuple, List, Optional, Dict
from src.data_acquisition.utils import logger

def create_sequences(
    df: cudf.DataFrame,
    target_col: str = 'gage_height_ft',
    timestamp_col: str = 'timestamp',
    lookback: int = 24,
    horizon: int = 12,
    feature_cols: Optional[List[str]] = None,
    stride: int = 1,
    return_torch: bool = False
) -> Tuple[cudf.DataFrame, cudf.DataFrame]:
    """
    Create sliding window sequences for time-series forecasting on GPU.
    """
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in [target_col, timestamp_col]]

    # Prevent duplicate columns
    if target_col in feature_cols:
        logger.warning(f"target_col '{target_col}' was in feature_cols — removing from features")
        feature_cols = [col for col in feature_cols if col != target_col]

    all_cols = feature_cols + [target_col]

    # Sort by timestamp and check monotonicity
    if not df[timestamp_col].is_monotonic_increasing:
        logger.warning("DataFrame not sorted by timestamp — sorting now")
        df = df.sort_values(timestamp_col).reset_index(drop=True)

    n_samples = len(df) - lookback - horizon + 1
    if n_samples <= 0:
        raise ValueError(f"Data too short for lookback={lookback} + horizon={horizon}. Need at least {lookback + horizon} rows.")

    logger.info(f"Creating {n_samples} sequences with lookback={lookback}, horizon={horizon}, stride={stride}")

    X_list = []
    y_list = []

    for i in range(0, n_samples, stride):
        window_start = i
        window_end = i + lookback
        target_start = i + lookback
        target_end = i + lookback + horizon

        x_window = df.iloc[window_start:window_end][all_cols]
        y_future = df.iloc[target_start:target_end][target_col]

        X_list.append(x_window)
        y_list.append(y_future)

    # Concatenate on GPU
    X = cudf.concat(X_list)
    y = cudf.concat(y_list)

    # Reshape X to (n_samples, lookback, n_features)
    n_features = len(all_cols)
    X = X.values.reshape(-1, lookback, n_features)

    # y: mean over horizon (simple multi-step target)
    y = y.values.reshape(-1, horizon)
    y_mean = cp.mean(y, axis=1)  # CuPy handles axis reduction reliably
    y = cudf.Series(y_mean)

    if return_torch:
        import torch
        device = 'cuda' if cp.cuda.runtime.getDeviceCount() > 0 else 'cpu'
        X = torch.as_tensor(X, device=device)
        y = torch.as_tensor(y, device=device)

    logger.info(f"Generated sequences: X shape {X.shape}, y shape {y.shape}")
    return X, y


def create_multi_horizon_sequences(
    df: cudf.DataFrame,
    target_col: str = 'gage_height_ft',
    lookback: int = 24,
    horizons: List[int] = [1, 3, 6, 12],
    feature_cols: Optional[List[str]] = None
) -> Tuple[cudf.DataFrame, Dict[int, cudf.Series]]:
    """
    Create sequences for multiple forecast horizons (useful for multi-step forecasting).
    Returns X and dict of y for each horizon.
    """
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in [target_col, 'timestamp']]

    X, _ = create_sequences(df, target_col, lookback, max(horizons), feature_cols, stride=1)

    y_multi = {}
    for h in horizons:
        _, y_h = create_sequences(df, target_col, lookback, h, feature_cols, stride=1)
        y_multi[h] = y_h

    return X, y_multi