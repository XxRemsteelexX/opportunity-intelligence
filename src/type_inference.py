"""
type inference service - ported from thompson_env's cast_types adn split_cols
automaticaly detects column types (categorical, numeric, datetime)
"""

import pandas as pd
import numpy as np
from typing import Tuple

NUMERIC_CONVERSION_THRESHOLD = 0.7
DATETIME_CONVERSION_THRESHOLD = 0.5


class TypeInferenceService:
    """infer column types from data"""

    def cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """smart type casting - try datetime first then numeric"""
        for col in df.columns:
            if df[col].dtype == object:
                # try datetime
                try:
                    converted = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                    if converted.notna().mean() >= DATETIME_CONVERSION_THRESHOLD:
                        df[col] = converted
                        continue
                except Exception:
                    pass

                # try numerc
                try:
                    converted = pd.to_numeric(df[col], errors="coerce")
                    if converted.notna().mean() >= NUMERIC_CONVERSION_THRESHOLD:
                        df[col] = converted
                        continue
                except Exception:
                    pass

        return df

    def split_cols(self, df: pd.DataFrame) -> Tuple[list, list, list]:
        """split columns into categorical, numeric, adn date lists"""
        cats, nums, dates = [], [], []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                dates.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                nums.append(col)
            else:
                cats.append(col)
        return cats, nums, dates

    def profile_column(self, series: pd.Series) -> dict:
        """Generate a profile for a single colum"""
        profile = {
            "name": series.name,
            "dtype": str(series.dtype),
            "nullCount": int(series.isna().sum()),
            "nullPercent": round(float(series.isna().mean() * 100), 1),
            "uniqueCount": int(series.nunique()),
        }

        if pd.api.types.is_numeric_dtype(series):
            profile["type"] = "numeric"
            clean = series.dropna()
            if len(clean) > 0:
                profile["min"] = float(clean.min())
                profile["max"] = float(clean.max())
                profile["mean"] = round(float(clean.mean()), 2)
                profile["median"] = round(float(clean.median()), 2)
                profile["std"] = round(float(clean.std()), 2)
        elif pd.api.types.is_datetime64_any_dtype(series):
            profile["type"] = "datetime"
            clean = series.dropna()
            if len(clean) > 0:
                profile["min"] = str(clean.min())
                profile["max"] = str(clean.max())
        else:
            profile["type"] = "categorical"
            top_values = series.value_counts().head(10).to_dict()
            profile["topValues"] = {str(k): int(v) for k, v in top_values.items()}

        return profile
