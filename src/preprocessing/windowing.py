import cudf
import cupy as cp
from typing import Tuple, Optional, List
from src.data_acquisition.utils import logger

def create_sequences(
    df: cudf.DataFrame,
    target_col: str = 'gage_height_ft',
    timestamp_col: str = 'timestamp',
    lookback: int = 96,
    horizon: int = 12,
    feature_cols: Optional[List[str]] = None,
    stride: int = 12,
    return_torch: bool = False
) -> Tuple[cp.ndarray, cp.ndarray]:
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in [target_col, timestamp_col]]

    if target_col in feature_cols:
        logger.warning(f"target_col '{target_col}' was in feature_cols — removing")
        feature_cols = [col for col in feature_cols if col != target_col]

    all_cols = feature_cols + [target_col]

    if not df[timestamp_col].is_monotonic_increasing:
        df = df.sort_values(timestamp_col).reset_index(drop=True)

    n = len(df)
    n_samples = (n - lookback - horizon + 1) // stride
    if n_samples <= 0:
        raise ValueError(f"Too few samples: {n_samples}")

    logger.info(f"Creating ~{n_samples} sequences | lookback={lookback}, horizon={horizon}, stride={stride}")

    data_cp = cp.asarray(df[all_cols].values)  # (n_rows, n_features)

    starts = cp.arange(0, n - lookback - horizon + 1, stride)
    X_indices = starts[:, None] + cp.arange(lookback)               # (n_samples, lookback)

    # Indices for the next horizon steps
    future_indices = starts[:, None] + cp.arange(lookback, lookback + horizon)

    X = data_cp[X_indices]                                    # (n_samples, lookback, n_features)
    y_future = data_cp[future_indices]                        # (n_samples, horizon, n_features)
    y_mean = cp.mean(y_future[:, :, all_cols.index(target_col)], axis=1)  # (n_samples,)

    if return_torch:
        import torch
        device = 'cuda' if cp.cuda.runtime.getDeviceCount() > 0 else 'cpu'
        X = torch.as_tensor(X, device=device)
        y_mean = torch.as_tensor(y_mean, device=device)

    logger.info(f"Generated: X {X.shape}, y {y_mean.shape} (mean over {horizon} steps)")

    return X, y_mean