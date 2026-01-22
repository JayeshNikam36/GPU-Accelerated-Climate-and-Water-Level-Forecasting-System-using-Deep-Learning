# src/data_acquisition/validator.py
import json
import pandas as pd
from scipy import stats
import os
from typing import Dict, List
from .utils import logger, get_data_path

class DataValidator:
    def __init__(self, source: str = "usgs"):  # e.g., "usgs", "noaa", "nasa"
        self.source = source

    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        if self.source == "usgs":
            ts_values = data['value']['timeSeries'][0]['values'][0]['value']
            df = pd.DataFrame(ts_values)
            # Fix timezone warning + dtype
            df['dateTime'] = pd.to_datetime(df['dateTime'], utc=True).dt.tz_localize(None)
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
        elif self.source == "noaa":
            df = pd.DataFrame(data['results'])
            df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
        else:
            raise NotImplementedError(f"Loader for {self.source} not implemented")
        
        logger.info(f"Loaded {len(df)} records from {file_path}")
        return df

    def check_missing(self, df: pd.DataFrame, threshold: float = 0.2) -> Dict:
        """Check missing values percentage."""
        missing_pct = df.isnull().mean()
        flags = {col: pct for col, pct in missing_pct.items() if pct > threshold}
        if flags:
            logger.warning(f"Missing data flags: {flags}")
        return {"missing_flags": flags, "total_rows": len(df), "missing_rows": df.isnull().any(axis=1).sum()}

    def check_outliers(self, df: pd.DataFrame, method: str = "iqr") -> Dict:
        numeric_cols = df.select_dtypes(include=['number']).columns
        if method == "iqr":
            Q1 = df[numeric_cols].quantile(0.25)
            Q3 = df[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        else:  # z-score fallback
            z_scores = stats.zscore(df[numeric_cols], nan_policy='omit')
            outliers = (abs(z_scores) > 3).any(axis=1)
        
        outlier_count = outliers.sum()
        if outlier_count > 0:
            logger.warning(f"{outlier_count} outliers detected ({method})")
        return {"outlier_count": outlier_count, "outlier_method": method}

    def check_schema(self, df: pd.DataFrame, expected_cols: List[str], expected_dtypes: Dict) -> Dict:
        """Validate columns and dtypes."""
        missing_cols = [col for col in expected_cols if col not in df.columns]
        dtype_errors = {}
        for col, exp_dtype in expected_dtypes.items():
            if col in df.columns and not pd.api.types.is_dtype_equal(df[col].dtype, exp_dtype):
                dtype_errors[col] = f"Expected {exp_dtype}, got {df[col].dtype}"
        
        if missing_cols or dtype_errors:
            logger.warning(f"Schema issues: Missing cols {missing_cols}, Dtype errors {dtype_errors}")
        return {"missing_cols": missing_cols, "dtype_errors": dtype_errors}

    def validate_file(self, file_path: str) -> Dict:
        """Run all checks on a file."""
        df = self.load_raw_data(file_path)
        results = {}
        results.update(self.check_missing(df))
        results.update(self.check_outliers(df))
        
        # Source-specific schema
        if self.source == "usgs":
            expected_cols = ["value", "dateTime"]
            expected_dtypes = {"value": "float64", "dateTime": "datetime64[ns]"}
        elif self.source == "noaa":
            expected_cols = ["value", "date", "datatype"]
            expected_dtypes = {"value": "float64", "date": "datetime64[ns]"}
        results.update(self.check_schema(df, expected_cols, expected_dtypes))
        
        logger.info(f"Validation complete for {file_path}: {results}")
        return results