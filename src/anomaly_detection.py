"""
anomaly detection service -- ported from geospatial crime analysis
multi-tier zscore system and isolation forest
"""

import pandas as pd
import numpy as np
from typing import Optional


class AnomalyDetector:
    """multi-method anomaly detection"""

    def zscore_detection(self, series: pd.Series, thresholds: dict = None) -> dict:
        """
        multi-tier zscore anomaly detection
        tiers: critical (>3s), warning (>2s), info (>1s), normal
        """
        if thresholds is None:
            thresholds = {"critical": 3.0, "warning": 2.0, "info": 1.0}

        mean = series.mean()
        std = series.std()

        if std == 0:
            return {"alerts": [], "summary": "No variation in data"}

        z_scores = (series - mean) / std

        alerts = []
        for idx, z in z_scores.items():
            abs_z = abs(z)
            if abs_z > thresholds["critical"]:
                severity = "critical"
            elif abs_z > thresholds["warning"]:
                severity = "warning"
            elif abs_z > thresholds["info"]:
                severity = "info"
            else:
                continue

            alerts.append({
                "index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                "value": float(series[idx]),
                "z_score": round(float(z), 3),
                "severity": severity,
            })

        return {
            "alerts": sorted(alerts, key=lambda x: abs(x["z_score"]), reverse=True),
            "summary": {
                "critical": sum(1 for a in alerts if a["severity"] == "critical"),
                "warning": sum(1 for a in alerts if a["severity"] == "warning"),
                "info": sum(1 for a in alerts if a["severity"] == "info"),
                "total": len(alerts),
            },
            "stats": {
                "mean": round(float(mean), 4),
                "std": round(float(std), 4),
            },
        }

    def isolation_forest(self, df: pd.DataFrame, columns: list, contamination: float = 0.05) -> dict:
        """isolation forest anomaly deteciton"""
        from sklearn.ensemble import IsolationForest

        X = df[columns].dropna()
        model = IsolationForest(contamination=contamination, random_state=42)
        predictions = model.fit_predict(X)

        anomaly_indices = X.index[predictions == -1].tolist()
        scores = model.decision_function(X)

        return {
            "anomaly_count": int((predictions == -1).sum()),
            "total_count": len(X),
            "anomaly_indices": anomaly_indices[:100],  # limit for api response
            "anomaly_scores": {
                "mean": round(float(scores.mean()), 4),
                "min": round(float(scores.min()), 4),
                "max": round(float(scores.max()), 4),
            },
        }

    def detect_all(self, df: pd.DataFrame, columns: list, method: str = "auto") -> dict:
        """Run anomaly detection on multiple columns adn aggregate results"""
        results = {}

        for col in columns:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) < 5:
                continue

            # zscore is always fast so run it
            zscore_result = self.zscore_detection(series)
            results[col] = {
                "zscore": zscore_result,
                "total_alerts": zscore_result["summary"]["total"],
                "critical_alerts": zscore_result["summary"]["critical"],
            }

        # isolation forest on all columns together if enough data
        if method in ("auto", "isolation_forest") and len(df) >= 20:
            valid_cols = [c for c in columns if c in df.columns]
            if len(valid_cols) >= 2:
                iso_result = self.isolation_forest(df, valid_cols)
                results["_isolation_forest"] = iso_result

        return results
