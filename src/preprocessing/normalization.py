# src/preprocessing/normalization.py
import cudf
import cupy as cp
from typing import Tuple
from src.data_acquisition.utils import logger


class GPUNormalizer:
    """
    GPU-accelerated normalization for cuDF DataFrames.
    Supports MinMax and Standard scaling.
    """

    def __init__(self, method: str = "minmax", feature_range: Tuple[float, float] = (0, 1)):
        if method not in ["minmax", "standard"]:
            raise ValueError("method must be 'minmax' or 'standard'")

        self.method = method
        self.feature_range = feature_range

        self._min = None
        self._max = None
        self._mean = None
        self._std = None

    def fit(self, df: cudf.DataFrame, columns) -> "GPUNormalizer":
        if not isinstance(df, cudf.DataFrame):
            raise TypeError("GPUNormalizer expects a cudf.DataFrame")

        numeric_df = df[columns].astype("float32")

        if self.method == "minmax":
            self._min = numeric_df.min()
            self._max = numeric_df.max()
            logger.info(f"Fitted MinMax scaler")

        else:
            self._mean = numeric_df.mean()
            self._std = numeric_df.std()
            logger.info(f"Fitted Standard scaler")

        return self

    def transform(self, df: cudf.DataFrame, columns) -> cudf.DataFrame:
        if not isinstance(df, cudf.DataFrame):
            raise TypeError("GPUNormalizer expects a cudf.DataFrame")

        if self._min is None and self._mean is None:
            raise RuntimeError("Must call fit() before transform()")

        df_out = df.copy()

        for col in columns:
            if self.method == "minmax":
                denom = self._max[col] - self._min[col]
                df_out[col] = cp.where(
                    denom == 0,
                    0.0,
                    (df[col] - self._min[col]) / denom
                    * (self.feature_range[1] - self.feature_range[0])
                    + self.feature_range[0],
                )
            else:
                df_out[col] = (df[col] - self._mean[col]) / self._std[col]

        return df_out

    def fit_transform(self, df: cudf.DataFrame, columns) -> cudf.DataFrame:
        return self.fit(df, columns).transform(df, columns)

    def inverse_transform(self, df: cudf.DataFrame, columns) -> cudf.DataFrame:
        df_out = df.copy()

        for col in columns:
            if self.method == "minmax":
                df_out[col] = (
                    (df[col] - self.feature_range[0])
                    / (self.feature_range[1] - self.feature_range[0])
                    * (self._max[col] - self._min[col])
                    + self._min[col]
                )
            else:
                df_out[col] = df[col] * self._std[col] + self._mean[col]

        return df_out
