"""
ml model recommender - ported from ml-model-recommender adn ml full selection tool
profiles datasets adn suggests appropriate analysis types
"""

import pandas as pd
import numpy as np
from typing import Optional

from src.data_cleaning import DataCleaningService
from src.type_inference import TypeInferenceService


class ModelRecommender:
    """analyze datasets adn recommend analysis approaches"""

    def __init__(self):
        self.cleaner = DataCleaningService()
        self.type_service = TypeInferenceService()

    def infer_problem_type(self, df: pd.DataFrame, target: Optional[str] = None) -> Optional[str]:
        """infer whether this is classification, regression, clustering, or time-series"""
        if target and target in df.columns:
            series = df[target]
            if pd.api.types.is_numeric_dtype(series):
                if series.nunique() <= 10:
                    return "classification"
                return "regression"
            return "classification"
        return None

    def detect_target_column(self, df: pd.DataFrame) -> Optional[str]:
        """heuristic target column detecton"""
        target_names = ["target", "label", "y", "class", "outcome", "result", "status"]
        for col in df.columns:
            if col.lower().strip() in target_names:
                return col
        return None

    def assess_data_quality(self, df: pd.DataFrame) -> dict:
        """assess overall data quailty"""
        cats, nums, dates = self.type_service.split_cols(df)

        quality = {
            "rowCount": len(df),
            "columnCount": len(df.columns),
            "missingPercent": round(float(df.isna().mean().mean() * 100), 1),
            "duplicatePercent": round(float(df.duplicated().mean() * 100), 1),
            "numericColumns": len(nums),
            "categoricalColumns": len(cats),
            "dateColumns": len(dates),
            "highCardinalityColumns": [],
            "outlierColumns": [],
        }

        # detect high cardinality
        for col in cats:
            if df[col].nunique() > 50:
                quality["highCardinalityColumns"].append(col)

        # detect outliers using iqr
        for col in nums:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outlier_count = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
            if outlier_count > len(df) * 0.05:
                quality["outlierColumns"].append({"column": col, "count": int(outlier_count)})

        return quality

    def generate_suggestions(self, df: pd.DataFrame, quality: dict, problem_type: Optional[str]) -> list:
        """generate ranked analysis sugestions based on data characteristics"""
        cats, nums, dates = self.type_service.split_cols(df)
        suggestions = []

        # descriptive stats - always aplicable
        suggestions.append({
            "analysis_type": "summary_statistics",
            "category": "descriptive",
            "confidence": 1.0,
            "description": "Basic summary statistics for all columns",
            "requires_target": False,
            "applicable": True,
            "reason": "Applicable to any dataset",
        })

        # correlation - if 2+ numeric columns
        if len(nums) >= 2:
            suggestions.append({
                "analysis_type": "correlation_matrix",
                "category": "descriptive",
                "confidence": 0.95,
                "description": "Correlation analysis between numeric variables",
                "requires_target": False,
                "applicable": True,
                "reason": f"Found {len(nums)} numeric columns",
            })

        # distribution analysis
        if len(nums) >= 1:
            suggestions.append({
                "analysis_type": "distribution_analysis",
                "category": "descriptive",
                "confidence": 0.9,
                "description": "Distribution plots and normality tests for numeric columns",
                "requires_target": False,
                "applicable": True,
                "reason": f"Found {len(nums)} numeric columns to analyze",
            })

        # outlier detection
        if len(nums) >= 1:
            suggestions.append({
                "analysis_type": "zscore_detection",
                "category": "anomaly",
                "confidence": 0.85,
                "description": "Detect statistical anomalies using Z-score method",
                "requires_target": False,
                "applicable": True,
                "reason": f"Found {len(nums)} numeric columns suitable for anomaly detection",
            })

        # time series forecast - if date + numeric columns
        if len(dates) >= 1 and len(nums) >= 1:
            suggestions.append({
                "analysis_type": "time_series_forecast",
                "category": "predictive",
                "confidence": 0.9,
                "description": "Forecast future values using time series methods",
                "requires_target": False,
                "applicable": True,
                "reason": f"Date column '{dates[0]}' with {len(nums)} numeric columns available",
            })

        # clustering - if 2+ numeric columns, no clear target
        if len(nums) >= 2 and not problem_type:
            suggestions.append({
                "analysis_type": "kmeans",
                "category": "clustering",
                "confidence": 0.8,
                "description": "Group similar records using K-Means clustering",
                "requires_target": False,
                "applicable": True,
                "reason": f"{len(nums)} numeric features available for clustering",
            })

        # classification/regression - if target detected
        if problem_type == "classification":
            suggestions.append({
                "analysis_type": "random_forest",
                "category": "predictive",
                "confidence": 0.85,
                "description": "Random Forest classifier for prediction",
                "requires_target": True,
                "applicable": True,
                "reason": "Classification target detected",
            })

        if problem_type == "regression":
            suggestions.append({
                "analysis_type": "linear_regression",
                "category": "predictive",
                "confidence": 0.85,
                "description": "Linear regression for predicting numeric outcomes",
                "requires_target": True,
                "applicable": True,
                "reason": "Regression target detected",
            })

        # group comparison - if categorical + numeric
        if len(cats) >= 1 and len(nums) >= 1:
            suggestions.append({
                "analysis_type": "group_comparison",
                "category": "comparative",
                "confidence": 0.8,
                "description": "Compare numeric metrics across categorical groups",
                "requires_target": False,
                "applicable": True,
                "reason": f"Can compare {nums[0]} across {cats[0]} groups",
            })

        # chi-square - if 2+ categorical
        if len(cats) >= 2:
            suggestions.append({
                "analysis_type": "chi_square_test",
                "category": "comparative",
                "confidence": 0.75,
                "description": "Test independence between categorical variables",
                "requires_target": False,
                "applicable": True,
                "reason": f"Found {len(cats)} categorical columns to test",
            })

        # market basket - if transaction-like data detected
        if len(cats) >= 2 and len(df) > 100:
            has_transaction_like = any(
                df[col].nunique() > len(df) * 0.5
                for col in cats
            )
            if has_transaction_like:
                suggestions.append({
                    "analysis_type": "market_basket_analysis",
                    "category": "association",
                    "confidence": 0.7,
                    "description": "Find product/item associations using Apriori algorithm",
                    "requires_target": False,
                    "applicable": True,
                    "reason": "Transaction-like data pattern detected",
                })

        # pca - if many numeric columns
        if len(nums) >= 5:
            suggestions.append({
                "analysis_type": "pca_analysis",
                "category": "feature_analysis",
                "confidence": 0.75,
                "description": "Reduce dimensionality and identify key components",
                "requires_target": False,
                "applicable": True,
                "reason": f"{len(nums)} numeric features - PCA can reveal key patterns",
            })

        # feature importance - if target detected
        if problem_type and len(nums) >= 2:
            suggestions.append({
                "analysis_type": "feature_importance",
                "category": "feature_analysis",
                "confidence": 0.8,
                "description": "Identify which features most influence the target variable",
                "requires_target": True,
                "applicable": True,
                "reason": f"Target detected with {len(nums)} numeric features",
            })

        # sort by confidence
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        return suggestions

    def analyze_and_recommend(self, filepath: str, target_column: Optional[str] = None) -> dict:
        """Full pipeline: load, profile, adn recommend"""
        df = self.cleaner.load_file(filepath)

        # detect target
        suggested_target = target_column or self.detect_target_column(df)
        problem_type = self.infer_problem_type(df, suggested_target)

        # assess quality
        quality = self.assess_data_quality(df)

        # generate sugestions
        suggestions = self.generate_suggestions(df, quality, problem_type)

        return {
            "problem_type": problem_type,
            "suggested_target": suggested_target,
            "data_quality": quality,
            "suggestions": suggestions,
        }
