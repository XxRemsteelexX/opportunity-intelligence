"""
data cleaning service - ported from ml full selection tool's preprocessor
and dalbey_analytics data loading logic
"""



import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from src.type_inference import TypeInferenceService
MISSING_THRESHOLD = 0.5


class DataCleaningService:
    """clean -profile uploaded datasets"""

    def __init__(self):
        self.type_service = TypeInferenceService()

    def load_file(self, filepath: str, filename: str = "") -> pd.DataFrame:
        """load a file in any supported format"""
        path = Path(filepath)
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(filepath)
        elif suffix in (".xlsx", ".xls"):
            return pd.read_excel(filepath)
        elif suffix == ".json":
            return pd.read_json(filepath)
        elif suffix == ".parquet":
            return pd.read_parquet(filepath)
        elif suffix == ".tsv":
            return pd.read_csv(filepath, sep="\t")
        else:
            # try csv as defualt
            return pd.read_csv(filepath)

    def clean_and_profile(self, filepath: str, filename: str = "") -> dict:
        """Full cleaning pipeline - load- profile"""
        actions = []

        # load the file
        df = self.load_file(filepath, filename)
        original_shape = df.shape
        actions.append(f"Loaded {original_shape[0]} rows x {original_shape[1]} columns")

        #remove completly empty rows/columns
        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")
        if df.shape != original_shape:
            actions.append(
                f"Removed empty rows/columns: {original_shape[0] - df.shape[0]} rows, "
                f"{original_shape[1] - df.shape[1]} columns"
            )

        # remove exact duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            df = df.drop_duplicates()
            actions.append(f"Removed {dup_count} duplicate rows")

        # strip whitespace from string colums
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip()

        # replace infinte values
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            df = df.replace([np.inf, -np.inf], np.nan)
            actions.append(f"Replaced {inf_count} infinite values with NaN")

        # smart type casting
        df = self.type_service.cast_types(df)
        actions.append("Applied smart type inference (datetime, numeric detection)")

        #drop columns with too many missing values
        high_missing = [
            col for col in df.columns
            if df[col].isna().mean() > MISSING_THRESHOLD
        ]
        if high_missing:
            df = df.drop(columns=high_missing)
            actions.append(f"Dropped {len(high_missing)} columns with >{MISSING_THRESHOLD*100}% missing: {high_missing}")

        #  impute remaining missing values
        for col in df.columns:
            if df[col].isna().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    actions.append(f"Imputed {col} missing values with median ({median_val:.2f})")
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    pass  # dont impute dates
                else:
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val.iloc[0])
                        actions.append(f"Imputed {col} missing values with mode ({mode_val.iloc[0]})")

        
        #profile the colums
        cats, nums, dates = self.type_service.split_cols(df)
        columns = [self.type_service.profile_column(df[col]) for col in df.columns]

        data_summary = {
            "numericColumns": nums,
            "categoricalColumns": cats,
            "dateColumns": dates,
            "totalRows": len(df),
            "totalColumns": len(df.columns),
            "missingCells": int(df.isna().sum().sum()),
            "memoryUsageMB": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        }


        
        # save cleaned file
        cleaned_path = filepath.rsplit(".", 1)[0] + "_cleaned.csv"
        df.to_csv(cleaned_path, index=False)

        return {
            "status": "READY",
            "rowCount": len(df),
            "columnCount": len(df.columns),
            "columns": columns,
            "cleaningLog": actions,
            "dataSummary": data_summary,
        }
