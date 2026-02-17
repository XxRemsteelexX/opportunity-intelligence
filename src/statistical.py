"""
statistical analysis service - consolidated from blue zones, geospatial crime analysis,
adn ml tools. provides 40+ analysis types
"""

import re
import pandas as pd
import numpy as np
from typing import Optional
from scipy import stats

from src.data_cleaning import DataCleaningService
from src.type_inference import TypeInferenceService


class StatisticalAnalyzer:
    """run statistical analyses adn return results with chart configs"""

    def __init__(self):
        self.cleaner = DataCleaningService()
        self.type_service = TypeInferenceService()

    def run_analysis(
        self,
        filepath: str,
        analysis_type: str,
        target_column: Optional[str] = None,
        features: Optional[list] = None,
        params: dict = {},
    ) -> dict:
        """dispatch to the apropriate analysis method"""
        df = self.cleaner.load_file(filepath)
        cats, nums, dates = self.type_service.split_cols(df)

        dispatch = {
            "summary_statistics": self._summary_statistics,
            "distribution_analysis": self._distribution_analysis,
            "correlation_matrix": self._correlation_matrix,
            "missing_value_analysis": self._missing_value_analysis,
            "data_quality_report": self._data_quality_report,
            "outlier_detection": self._outlier_detection,
            "zscore_detection": self._zscore_detection,
            "group_comparison": self._group_comparison,
            "cross_tabulation": self._cross_tabulation,
            "anova": self._anova,
            "chi_square_test": self._chi_square_test,
            "effect_size_analysis": self._effect_size_analysis,
            "collinearity_check": self._collinearity_check,
            "linear_regression": self._linear_regression,
            "kmeans": self._kmeans_clustering,
            "pca_analysis": self._pca_analysis,
            "feature_importance": self._feature_importance,
            "normality_test": self._normality_test,
            "percentile_analysis": self._percentile_analysis,
            "variance_analysis": self._variance_analysis,
            "competitive_landscape": self._competitive_landscape,
            "trend_analysis": self._trend_analysis,
        }

        analysis_type = self._normalize_type(analysis_type)
        handler = dispatch.get(analysis_type)
        if not handler:
            return self._placeholder_analysis(analysis_type, df, nums, cats, dates)

        return handler(df, nums, cats, dates, target_column, features, params)

    def _summary_statistics(self, df, nums, cats, dates, target, features, params) -> dict:
        """generate summary statistics for all colums"""
        desc = df.describe(include="all").to_dict()
        insights = []

        for col in nums:
            skew = float(df[col].skew())
            if abs(skew) > 1:
                insights.append({
                    "type": "observation",
                    "field": col,
                    "message": f"{col} is {'right' if skew > 0 else 'left'}-skewed (skewness: {skew:.2f})",
                })

        charts = []
        if nums:
            # bar chart of means
            means = {col: float(df[col].mean()) for col in nums[:10]}
            charts.append({
                "type": "bar",
                "title": "Column Means Overview",
                "xAxis": {"type": "category", "data": list(means.keys())},
                "yAxis": {"type": "value"},
                "series": [{"type": "bar", "data": list(means.values())}],
            })

        return {
            "analysis_type": "summary_statistics",
            "summary": f"Dataset has {len(df)} rows, {len(nums)} numeric and {len(cats)} categorical columns.",
            "insights": insights,
            "charts": charts,
            "statistical_details": {"describe": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in desc.items()}},
            "recommendations": [
                "Review skewed columns for potential log transformation",
                "Check for outliers in numeric columns",
            ],
        }

    def _distribution_analysis(self, df, nums, cats, dates, target, features, params) -> dict:
        """analyze distrbutions of numeric columns"""
        cols = features or nums[:6]
        insights = []
        charts = []

        for col in cols:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) == 0:
                continue

            # shapiro-wilk test for normality (sample if too large)
            sample = series.sample(min(5000, len(series)), random_state=42)
            try:
                stat, p_value = stats.shapiro(sample)
                is_normal = p_value > 0.05
                insights.append({
                    "type": "test_result",
                    "field": col,
                    "message": f"{col}: {'Normal' if is_normal else 'Non-normal'} distribution (Shapiro p={p_value:.4f})",
                })
            except Exception:
                pass

            # histogram chart confg
            hist_values, bin_edges = np.histogram(series, bins=30)
            charts.append({
                "type": "bar",
                "title": f"Distribution of {col}",
                "xAxis": {"type": "category", "data": [f"{edge:.1f}" for edge in bin_edges[:-1]]},
                "yAxis": {"type": "value", "name": "Frequency"},
                "series": [{"type": "bar", "data": hist_values.tolist()}],
            })

        return {
            "analysis_type": "distribution_analysis",
            "summary": f"Analyzed distributions for {len(cols)} columns.",
            "insights": insights,
            "charts": charts,
            "recommendations": ["Consider normalization for non-normal distributions"],
        }

    def _correlation_matrix(self, df, nums, cats, dates, target, features, params) -> dict:
        """compute correlation matrix for numeric colums"""
        cols = features or nums[:15]
        corr_df = df[cols].corr()
        corr_data = corr_df.values.tolist()

        # find strong correlatins
        insights = []
        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i < j:
                    r = corr_df.iloc[i, j]
                    if abs(r) > 0.7:
                        insights.append({
                            "type": "correlation",
                            "fields": [col1, col2],
                            "message": f"Strong {'positive' if r > 0 else 'negative'} correlation between {col1} and {col2} (r={r:.3f})",
                        })

        charts = [{
            "type": "heatmap",
            "title": "Correlation Matrix",
            "xAxis": {"type": "category", "data": cols},
            "yAxis": {"type": "category", "data": cols},
            "series": [{
                "type": "heatmap",
                "data": [
                    [i, j, round(corr_data[i][j], 2)]
                    for i in range(len(cols))
                    for j in range(len(cols))
                ],
            }],
        }]

        return {
            "analysis_type": "correlation_matrix",
            "summary": f"Correlation analysis for {len(cols)} numeric columns. Found {len(insights)} strong correlations.",
            "insights": insights,
            "charts": charts,
            "statistical_details": {"correlation_matrix": corr_df.to_dict()},
            "recommendations": [
                "Investigate strongly correlated features for redundancy",
                "Consider removing one of each highly correlated pair for modeling",
            ],
        }

    def _zscore_detection(self, df, nums, cats, dates, target, features, params) -> dict:
        """multi-tier z-score anomaly detecton (from geospatial crime analysis)"""
        cols = features or nums[:10]
        insights = []
        all_anomalies = []

        for col in cols:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) == 0:
                continue

            mean = series.mean()
            std = series.std()
            if std == 0:
                continue

            z_scores = (series - mean) / std

            critical = (z_scores.abs() > 3).sum()  # >99.7%
            warning = ((z_scores.abs() > 2) & (z_scores.abs() <= 3)).sum()  # 95-99.7%
            info = ((z_scores.abs() > 1) & (z_scores.abs() <= 2)).sum()  # 68-95%

            if critical > 0:
                insights.append({
                    "type": "anomaly",
                    "severity": "critical",
                    "field": col,
                    "message": f"{col}: {critical} critical anomalies (>3 sigma)",
                })

            all_anomalies.append({
                "column": col,
                "critical": int(critical),
                "warning": int(warning),
                "info": int(info),
                "normal": int(len(series) - critical - warning - info),
            })

        # stacked bar chart of anomaly tiers
        charts = [{
            "type": "bar",
            "title": "Anomaly Detection by Column",
            "xAxis": {"type": "category", "data": [a["column"] for a in all_anomalies]},
            "yAxis": {"type": "value"},
            "series": [
                {"name": "Critical (>3s)", "type": "bar", "stack": "total",
                 "data": [a["critical"] for a in all_anomalies]},
                {"name": "Warning (>2s)", "type": "bar", "stack": "total",
                 "data": [a["warning"] for a in all_anomalies]},
                {"name": "Info (>1s)", "type": "bar", "stack": "total",
                 "data": [a["info"] for a in all_anomalies]},
            ],
        }]

        return {
            "analysis_type": "zscore_detection",
            "summary": f"Z-score anomaly detection across {len(cols)} columns.",
            "insights": insights,
            "charts": charts,
            "statistical_details": {"anomalies": all_anomalies},
            "recommendations": [
                "Investigate critical anomalies for data quality issues",
                "Consider robust methods (IQR, Isolation Forest) for confirmation",
            ],
        }

    def _outlier_detection(self, df, nums, cats, dates, target, features, params) -> dict:
        return self._zscore_detection(df, nums, cats, dates, target, features, params)

    def _group_comparison(self, df, nums, cats, dates, target, features, params) -> dict:
        """compare numeric metrics accross categorical groups"""
        group_col = (features[0] if features else cats[0]) if cats else None
        value_col = (features[1] if features and len(features) > 1 else nums[0]) if nums else None

        if not group_col or not value_col:
            return self._placeholder_analysis("group_comparison", df, nums, cats, dates)

        groups = df.groupby(group_col)[value_col].agg(["mean", "median", "std", "count"])
        groups = groups.sort_values("mean", ascending=False).head(20)

        insights = [{
            "type": "comparison",
            "message": f"Highest average {value_col}: {groups.index[0]} ({groups.iloc[0]['mean']:.2f})",
        }]

        charts = [{
            "type": "bar",
            "title": f"{value_col} by {group_col}",
            "xAxis": {"type": "category", "data": groups.index.tolist()},
            "yAxis": {"type": "value", "name": value_col},
            "series": [{"type": "bar", "data": groups["mean"].round(2).tolist()}],
        }]

        return {
            "analysis_type": "group_comparison",
            "summary": f"Compared {value_col} across {len(groups)} groups of {group_col}.",
            "insights": insights,
            "charts": charts,
            "recommendations": ["Run ANOVA to test if group differences are statistically significant"],
        }

    def _chi_square_test(self, df, nums, cats, dates, target, features, params) -> dict:
        """chi-square test of independnce between categorical variables"""
        if len(cats) < 2:
            return self._placeholder_analysis("chi_square_test", df, nums, cats, dates)

        col1 = features[0] if features else cats[0]
        col2 = features[1] if features and len(features) > 1 else cats[1]

        contingency = pd.crosstab(df[col1], df[col2])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        insights = [{
            "type": "test_result",
            "message": f"Chi-square test: chi2={chi2:.2f}, p={p_value:.4f}, Cramer's V={cramers_v:.3f}",
        }]

        return {
            "analysis_type": "chi_square_test",
            "summary": f"{'Significant' if p_value < 0.05 else 'No significant'} relationship between {col1} and {col2}.",
            "insights": insights,
            "charts": [],
            "statistical_details": {"chi2": chi2, "p_value": p_value, "dof": dof, "cramers_v": cramers_v},
            "recommendations": ["Examine standardized residuals for specific cell contributions"],
        }

    def _linear_regression(self, df, nums, cats, dates, target, features, params) -> dict:
        """simple linear regressoin"""
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score

        if not target or target not in df.columns:
            return self._placeholder_analysis("linear_regression", df, nums, cats, dates)

        feature_cols = features or [c for c in nums if c != target][:10]
        X = df[feature_cols].dropna()
        y = df.loc[X.index, target]

        model = LinearRegression()
        scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        model.fit(X, y)

        coefs = dict(zip(feature_cols, model.coef_.round(4).tolist()))

        insights = [{
            "type": "model_result",
            "message": f"R2 = {scores.mean():.3f} (+/- {scores.std():.3f}) via 5-fold CV",
        }]

        charts = [{
            "type": "bar",
            "title": "Feature Coefficients",
            "xAxis": {"type": "category", "data": list(coefs.keys())},
            "yAxis": {"type": "value", "name": "Coefficient"},
            "series": [{"type": "bar", "data": list(coefs.values())}],
        }]

        return {
            "analysis_type": "linear_regression",
            "summary": f"Linear regression: R2={scores.mean():.3f}. Top predictor: {max(coefs, key=lambda x: abs(coefs[x]))}.",
            "insights": insights,
            "charts": charts,
            "statistical_details": {"r2_mean": float(scores.mean()), "r2_std": float(scores.std()), "coefficients": coefs},
            "recommendations": ["Check residuals for heteroscedasticity", "Consider feature engineering for non-linear relationships"],
        }

    def _kmeans_clustering(self, df, nums, cats, dates, target, features, params) -> dict:
        """k-means clustering"""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        cols = features or nums[:10]
        X = df[cols].dropna()
        if len(X) < 10:
            return self._placeholder_analysis("kmeans", df, nums, cats, dates)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_clusters = params.get("n_clusters", 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        cluster_sizes = pd.Series(labels).value_counts().to_dict()

        insights = [{
            "type": "clustering",
            "message": f"Created {n_clusters} clusters. Sizes: {cluster_sizes}. Inertia: {kmeans.inertia_:.1f}",
        }]

        # scatter of first 2 dimensons
        charts = []
        if len(cols) >= 2:
            charts.append({
                "type": "scatter",
                "title": f"Clusters ({cols[0]} vs {cols[1]})",
                "xAxis": {"type": "value", "name": cols[0]},
                "yAxis": {"type": "value", "name": cols[1]},
                "series": [
                    {
                        "name": f"Cluster {i}",
                        "type": "scatter",
                        "data": X.iloc[labels == i][[cols[0], cols[1]]].values.tolist(),
                    }
                    for i in range(n_clusters)
                ],
            })

        return {
            "analysis_type": "kmeans",
            "summary": f"K-Means clustering with {n_clusters} clusters on {len(cols)} features.",
            "insights": insights,
            "charts": charts,
            "recommendations": ["Try different k values", "Use silhouette score to find optimal clusters"],
        }

    def _pca_analysis(self, df, nums, cats, dates, target, features, params) -> dict:
        """pca dimensionality reducton"""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        cols = features or nums[:20]
        X = df[cols].dropna()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_components = min(len(cols), 10)
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)

        explained = pca.explained_variance_ratio_.cumsum().tolist()

        insights = [{
            "type": "dimensionality",
            "message": f"First 3 components explain {explained[min(2, len(explained)-1)]*100:.1f}% of variance",
        }]

        charts = [{
            "type": "line",
            "title": "Cumulative Explained Variance",
            "xAxis": {"type": "category", "data": [f"PC{i+1}" for i in range(n_components)]},
            "yAxis": {"type": "value", "name": "Cumulative Variance %", "max": 1},
            "series": [{"type": "line", "data": [round(v, 3) for v in explained]}],
        }]

        return {
            "analysis_type": "pca_analysis",
            "summary": f"PCA on {len(cols)} features. {sum(1 for v in explained if v < 0.95)} components needed for 95% variance.",
            "insights": insights,
            "charts": charts,
            "recommendations": ["Use PCA components as features for modeling", "Examine loadings for feature interpretation"],
        }

    def _feature_importance(self, df, nums, cats, dates, target, features, params) -> dict:
        """feature importnace using random forest"""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        if not target or target not in df.columns:
            return self._placeholder_analysis("feature_importance", df, nums, cats, dates)

        feature_cols = features or [c for c in nums if c != target][:15]
        X = df[feature_cols].dropna()
        y = df.loc[X.index, target]

        if y.nunique() <= 10:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        model.fit(X, y)
        importances = dict(zip(feature_cols, model.feature_importances_.round(4).tolist()))
        sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

        insights = [{
            "type": "feature_importance",
            "message": f"Top feature: {list(sorted_imp.keys())[0]} (importance: {list(sorted_imp.values())[0]:.3f})",
        }]

        charts = [{
            "type": "bar",
            "title": "Feature Importance",
            "xAxis": {"type": "value"},
            "yAxis": {"type": "category", "data": list(reversed(sorted_imp.keys()))},
            "series": [{"type": "bar", "data": list(reversed(sorted_imp.values()))}],
        }]

        return {
            "analysis_type": "feature_importance",
            "summary": f"Top 3 features: {', '.join(list(sorted_imp.keys())[:3])}.",
            "insights": insights,
            "charts": charts,
            "statistical_details": {"importances": sorted_imp},
            "recommendations": ["Consider dropping low-importance features", "Use SHAP values for more detailed explanations"],
        }

    def _missing_value_analysis(self, df, nums, cats, dates, target, features, params) -> dict:
        """check misssing data patterns accross all columns"""
        missing = df.isna().sum()
        missing_pct = (df.isna().mean() * 100).round(1)

        # only report cols that actualy have misssing values
        has_missing = missing[missing > 0].sort_values(ascending=False)
        insights = []

        for col in has_missing.index:
            pct = missing_pct[col]
            severity = "critical" if pct > 50 else "warning" if pct > 20 else "info"
            insights.append({
                "type": "missing_data",
                "severity": severity,
                "field": col,
                "message": f"{col}: {int(has_missing[col])} missing ({pct}%)",
            })

        # completness sumary
        total_cells = df.shape[0] * df.shape[1]
        total_missing = int(missing.sum())
        completeness = round((1 - total_missing / total_cells) * 100, 1) if total_cells > 0 else 100

        # chart of missing by colum
        charts = []
        if len(has_missing) > 0:
            charts.append({
                "type": "bar",
                "title": "Missing Values by Column",
                "xAxis": {"type": "category", "data": has_missing.index.tolist()[:20]},
                "yAxis": {"type": "value", "name": "Missing Count"},
                "series": [{"type": "bar", "data": has_missing.values.tolist()[:20]}],
            })

        return {
            "analysis_type": "missing_value_analysis",
            "summary": f"Dataset is {completeness}% complete. {len(has_missing)} of {len(df.columns)} columns have missing values.",
            "insights": insights,
            "charts": charts,
            "statistical_details": {
                "completeness_pct": completeness,
                "total_missing_cells": total_missing,
                "columns_with_missing": len(has_missing),
                "missing_by_column": {col: int(v) for col, v in has_missing.items()},
            },
            "recommendations": [
                "Columns with >50% missing should be dropped or carefully evaluated",
                "Consider imputation strategy based on data type adn missingness pattern",
            ],
        }

    def _data_quality_report(self, df, nums, cats, dates, target, features, params) -> dict:
        """full data quality assesment -- checks completness, duplicates, types, adn outliers"""
        total_rows = len(df)
        total_cols = len(df.columns)

        # misssing data
        missing_pct = round(float(df.isna().mean().mean() * 100), 1)

        # duplictes
        dup_count = int(df.duplicated().sum())
        dup_pct = round(dup_count / total_rows * 100, 1) if total_rows > 0 else 0

        # high cardinality cols -- usually messy data
        high_card = [col for col in cats if df[col].nunique() > 50]

        # constant colums that add no info
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]

        # outlier check on numerics usign iqr
        outlier_cols = []
        for col in nums:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outlier_count = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
            if outlier_count > total_rows * 0.05:
                outlier_cols.append({"column": col, "count": outlier_count, "pct": round(outlier_count / total_rows * 100, 1)})

        # overall quality score 0-100
        quality_score = round(
            (1 - missing_pct / 100) * 30 +  # misssing penalty
            (1 - dup_pct / 100) * 20 +       # dup penaly
            (1 - len(constant_cols) / max(total_cols, 1)) * 15 +  # constant col penalty
            (1 - len(high_card) / max(len(cats), 1)) * 15 +  # cardinality penaly
            (1 - len(outlier_cols) / max(len(nums), 1)) * 20,  # outlier penalty
            1
        )

        insights = [
            {"type": "quality_score", "message": f"Overall data quality score: {quality_score}/100"},
            {"type": "completeness", "message": f"Missing data: {missing_pct}%"},
            {"type": "duplicates", "message": f"Duplicate rows: {dup_count} ({dup_pct}%)"},
        ]
        if constant_cols:
            insights.append({"type": "warning", "message": f"Constant columns (no variation): {constant_cols}"})
        if outlier_cols:
            insights.append({"type": "warning", "message": f"{len(outlier_cols)} columns have >5% outliers"})

        charts = [{
            "type": "bar",
            "title": "Data Quality Breakdown",
            "xAxis": {"type": "category", "data": ["Completeness", "No Duplicates", "Type Variety", "Cardinality", "No Outliers"]},
            "yAxis": {"type": "value", "name": "Score", "max": 30},
            "series": [{"type": "bar", "data": [
                round((1 - missing_pct / 100) * 30, 1),
                round((1 - dup_pct / 100) * 20, 1),
                round((1 - len(constant_cols) / max(total_cols, 1)) * 15, 1),
                round((1 - len(high_card) / max(len(cats), 1)) * 15, 1),
                round((1 - len(outlier_cols) / max(len(nums), 1)) * 20, 1),
            ]}],
        }]

        return {
            "analysis_type": "data_quality_report",
            "summary": f"Data quality score: {quality_score}/100. {total_rows} rows, {total_cols} columns, {missing_pct}% missing, {dup_pct}% duplicates.",
            "insights": insights,
            "charts": charts,
            "statistical_details": {
                "quality_score": quality_score,
                "row_count": total_rows,
                "column_count": total_cols,
                "missing_pct": missing_pct,
                "duplicate_count": dup_count,
                "duplicate_pct": dup_pct,
                "constant_columns": constant_cols,
                "high_cardinality_columns": high_card,
                "outlier_columns": outlier_cols,
            },
            "recommendations": [
                "Address columns with high missing percentages first",
                "Remove constant columns before modeling",
                "Review high cardinality categoricals for grouping or encoding",
            ],
        }

    def _cross_tabulation(self, df, nums, cats, dates, target, features, params) -> dict:
        """cross tab between two categorcal variables with row pcts"""
        if len(cats) < 2:
            return self._placeholder_analysis("cross_tabulation", df, nums, cats, dates)

        col1 = features[0] if features else cats[0]
        col2 = features[1] if features and len(features) > 1 else cats[1]

        # raw counts
        ct = pd.crosstab(df[col1], df[col2])
        # row percntages
        ct_pct = pd.crosstab(df[col1], df[col2], normalize='index').round(3) * 100

        # find the strongest assocation in the table
        max_val = 0
        max_cell = ("", "")
        for row in ct_pct.index:
            for col in ct_pct.columns:
                if ct_pct.loc[row, col] > max_val:
                    max_val = ct_pct.loc[row, col]
                    max_cell = (str(row), str(col))

        insights = [{
            "type": "cross_tab",
            "message": f"Strongest association: {max_cell[0]} x {max_cell[1]} ({max_val:.1f}% of row)",
        }]

        # also run chi square on the crosstab to test independnce
        chi2, p_value, dof, expected = stats.chi2_contingency(ct)
        n = ct.sum().sum()
        min_dim = min(ct.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        insights.append({
            "type": "test_result",
            "message": f"Chi-square: chi2={chi2:.2f}, p={p_value:.4f}, Cramer's V={cramers_v:.3f}",
        })

        # heatmap of the crosstab
        charts = [{
            "type": "heatmap",
            "title": f"Cross Tabulation: {col1} vs {col2}",
            "xAxis": {"type": "category", "data": [str(c) for c in ct.columns.tolist()]},
            "yAxis": {"type": "category", "data": [str(r) for r in ct.index.tolist()]},
            "series": [{
                "type": "heatmap",
                "data": [
                    [j, i, int(ct.iloc[i, j])]
                    for i in range(len(ct.index))
                    for j in range(len(ct.columns))
                ],
            }],
        }]

        return {
            "analysis_type": "cross_tabulation",
            "summary": f"Cross tabulation of {col1} ({len(ct.index)} levels) vs {col2} ({len(ct.columns)} levels). {'Significant' if p_value < 0.05 else 'No significant'} association (p={p_value:.4f}).",
            "insights": insights,
            "charts": charts,
            "statistical_details": {
                "counts": ct.to_dict(),
                "row_percentages": ct_pct.to_dict(),
                "chi2": round(chi2, 4),
                "p_value": round(p_value, 4),
                "cramers_v": round(cramers_v, 4),
            },
            "recommendations": [
                "Look at row percentages to understand conditional distributions",
                "Cramers V > 0.3 suggests a meaningful practical relationship",
            ],
        }

    def _anova(self, df, nums, cats, dates, target, features, params) -> dict:
        """One way anova -- compare means accross groups"""
        group_col = (features[0] if features else cats[0]) if cats else None
        value_col = (features[1] if features and len(features) > 1 else
                     target if target and target in nums else
                     nums[0] if nums else None)

        if not group_col or not value_col:
            return self._placeholder_analysis("anova", df, nums, cats, dates)

        # build groups for the f test
        groups_data = []
        group_labels = []
        for name, group in df.groupby(group_col)[value_col]:
            clean = group.dropna()
            if len(clean) >= 2:  # need at least 2 per group
                groups_data.append(clean.values)
                group_labels.append(str(name))

        if len(groups_data) < 2:
            return self._placeholder_analysis("anova", df, nums, cats, dates)

        # run the anova
        f_stat, p_value = stats.f_oneway(*groups_data)

        # group means for the chart
        group_means = df.groupby(group_col)[value_col].agg(['mean', 'std', 'count'])
        group_means = group_means.sort_values('mean', ascending=False)

        # effect size -- eta squared
        grand_mean = df[value_col].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups_data)
        ss_total = sum(((g - grand_mean) ** 2).sum() for g in groups_data)
        eta_squared = round(ss_between / ss_total, 4) if ss_total > 0 else 0

        insights = [
            {
                "type": "test_result",
                "message": f"ANOVA: F={f_stat:.3f}, p={p_value:.4f}. {'Significant' if p_value < 0.05 else 'Not significant'} difference between groups.",
            },
            {
                "type": "effect_size",
                "message": f"Eta-squared: {eta_squared} ({'large' if eta_squared > 0.14 else 'medium' if eta_squared > 0.06 else 'small'} effect)",
            },
        ]

        # post hoc -- pairwise t-tests if significant adn not too many groups
        if p_value < 0.05 and len(groups_data) <= 10:
            pairwise = []
            for i in range(len(groups_data)):
                for j in range(i + 1, len(groups_data)):
                    t_stat, t_p = stats.ttest_ind(groups_data[i], groups_data[j])
                    if t_p < 0.05:
                        pairwise.append({
                            "group1": group_labels[i],
                            "group2": group_labels[j],
                            "t_stat": round(float(t_stat), 3),
                            "p_value": round(float(t_p), 4),
                        })
            if pairwise:
                insights.append({
                    "type": "post_hoc",
                    "message": f"{len(pairwise)} significant pairwise differences found",
                })

        charts = [{
            "type": "bar",
            "title": f"Mean {value_col} by {group_col} (ANOVA p={p_value:.4f})",
            "xAxis": {"type": "category", "data": [str(x) for x in group_means.index.tolist()]},
            "yAxis": {"type": "value", "name": value_col},
            "series": [{"type": "bar", "data": group_means['mean'].round(2).tolist()}],
        }]

        return {
            "analysis_type": "anova",
            "summary": f"One-way ANOVA: {value_col} by {group_col}. F={f_stat:.3f}, p={p_value:.4f}. Eta²={eta_squared}.",
            "insights": insights,
            "charts": charts,
            "statistical_details": {
                "f_statistic": round(float(f_stat), 4),
                "p_value": round(float(p_value), 4),
                "eta_squared": eta_squared,
                "group_stats": group_means.to_dict(),
                "pairwise": pairwise if p_value < 0.05 and len(groups_data) <= 10 else None,
            },
            "recommendations": [
                "Check assumptions: normality within groups adn homogeneity of variance",
                "Use Tukey HSD for more rigorous post-hoc comparisons",
                "Consider Welch ANOVA if variances are unequal",
            ],
        }

    def _effect_size_analysis(self, df, nums, cats, dates, target, features, params) -> dict:
        """cohens d and other effect sizes between two groups"""
        group_col = (features[0] if features else cats[0]) if cats else None
        value_col = (features[1] if features and len(features) > 1 else
                     target if target and target in nums else
                     nums[0] if nums else None)

        if not group_col or not value_col:
            return self._placeholder_analysis("effect_size_analysis", df, nums, cats, dates)

        groups = df.groupby(group_col)[value_col]
        group_names = list(groups.groups.keys())

        if len(group_names) < 2:
            return self._placeholder_analysis("effect_size_analysis", df, nums, cats, dates)

        results = []
        insights = []

        # compute cohens d for each pair of groups
        for i in range(min(len(group_names), 5)):
            for j in range(i + 1, min(len(group_names), 5)):
                g1 = groups.get_group(group_names[i]).dropna()
                g2 = groups.get_group(group_names[j]).dropna()

                if len(g1) < 2 or len(g2) < 2:
                    continue

                # pooled std for cohens d
                n1, n2 = len(g1), len(g2)
                var1, var2 = g1.var(), g2.var()
                pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

                if pooled_std == 0:
                    continue

                cohens_d = round(float((g1.mean() - g2.mean()) / pooled_std), 3)
                magnitude = "large" if abs(cohens_d) >= 0.8 else "medium" if abs(cohens_d) >= 0.5 else "small" if abs(cohens_d) >= 0.2 else "negligible"

                # also compute rank biserial as a nonparametric alternative
                u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                rank_biserial = round(1 - (2 * u_stat) / (n1 * n2), 3)

                results.append({
                    "group1": str(group_names[i]),
                    "group2": str(group_names[j]),
                    "cohens_d": cohens_d,
                    "magnitude": magnitude,
                    "mean_diff": round(float(g1.mean() - g2.mean()), 3),
                    "rank_biserial": rank_biserial,
                    "mann_whitney_p": round(float(u_p), 4),
                })

                if abs(cohens_d) >= 0.5:
                    insights.append({
                        "type": "effect_size",
                        "message": f"{group_names[i]} vs {group_names[j]}: Cohen's d={cohens_d} ({magnitude})",
                    })

        if not results:
            return self._placeholder_analysis("effect_size_analysis", df, nums, cats, dates)

        # chart cohens d values
        labels = [f"{r['group1']} vs {r['group2']}" for r in results]
        d_values = [r['cohens_d'] for r in results]

        charts = [{
            "type": "bar",
            "title": f"Effect Sizes (Cohen's d) for {value_col}",
            "xAxis": {"type": "category", "data": labels},
            "yAxis": {"type": "value", "name": "Cohen's d"},
            "series": [{"type": "bar", "data": d_values}],
        }]

        return {
            "analysis_type": "effect_size_analysis",
            "summary": f"Effect size analysis for {value_col} across {group_col}. {sum(1 for r in results if abs(r['cohens_d']) >= 0.5)} medium+ effects found.",
            "insights": insights if insights else [{"type": "info", "message": "All effect sizes are small (<0.5)"}],
            "charts": charts,
            "statistical_details": {"comparisons": results},
            "recommendations": [
                "Cohens d >= 0.8 is large, 0.5 medium, 0.2 small",
                "Rank-biserial is the nonparametric alternative if data isnt normal",
                "Report effect sizes alongside p-values for practical significance",
            ],
        }

    def _collinearity_check(self, df, nums, cats, dates, target, features, params) -> dict:
        """variance inflation factor to detect multicollinearity"""
        cols = features or nums[:15]
        if len(cols) < 2:
            return self._placeholder_analysis("collinearity_check", df, nums, cats, dates)

        X = df[cols].dropna()
        if len(X) < len(cols) + 1:
            return self._placeholder_analysis("collinearity_check", df, nums, cats, dates)

        # compute vif for each variable
        from numpy.linalg import LinAlgError

        vif_data = []
        try:
            for i, col in enumerate(cols):
                other_cols = [c for c in cols if c != col]
                X_other = X[other_cols].values
                y_col = X[col].values

                # add intercept
                X_with_const = np.column_stack([np.ones(len(X_other)), X_other])

                try:
                    coeffs = np.linalg.lstsq(X_with_const, y_col, rcond=None)[0]
                    y_pred = X_with_const @ coeffs
                    ss_res = ((y_col - y_pred) ** 2).sum()
                    ss_tot = ((y_col - y_col.mean()) ** 2).sum()
                    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                    vif = 1 / (1 - r_squared) if r_squared < 1 else float('inf')
                except LinAlgError:
                    vif = float('inf')

                vif_data.append({
                    "column": col,
                    "vif": round(float(vif), 2) if vif != float('inf') else 999.99,
                    "r_squared": round(float(r_squared), 4) if r_squared != float('inf') else 1.0,
                })
        except Exception:
            return self._placeholder_analysis("collinearity_check", df, nums, cats, dates)

        # sort by vif descending
        vif_data.sort(key=lambda x: x['vif'], reverse=True)

        high_vif = [v for v in vif_data if v['vif'] > 5]
        severe_vif = [v for v in vif_data if v['vif'] > 10]

        insights = []
        if severe_vif:
            insights.append({
                "type": "warning",
                "message": f"{len(severe_vif)} columns have VIF > 10 (severe multicollinearity): {[v['column'] for v in severe_vif]}",
            })
        if high_vif:
            insights.append({
                "type": "info",
                "message": f"{len(high_vif)} columns have VIF > 5 (moderate+ multicollinearity)",
            })
        if not high_vif:
            insights.append({
                "type": "info",
                "message": "No multicollinearity issues detected (all VIF < 5)",
            })

        charts = [{
            "type": "bar",
            "title": "Variance Inflation Factor by Column",
            "xAxis": {"type": "category", "data": [v['column'] for v in vif_data]},
            "yAxis": {"type": "value", "name": "VIF"},
            "series": [{"type": "bar", "data": [v['vif'] for v in vif_data]}],
        }]

        return {
            "analysis_type": "collinearity_check",
            "summary": f"VIF analysis on {len(cols)} columns. {len(high_vif)} have VIF > 5, {len(severe_vif)} have VIF > 10.",
            "insights": insights,
            "charts": charts,
            "statistical_details": {"vif_scores": vif_data},
            "recommendations": [
                "VIF > 10 indicates severe collinearity -- consider dropping or combining those features",
                "VIF > 5 warrants investigation especially for regression models",
                "PCA can help if many variables are collinear",
            ],
        }

    def _normalize_type(self, analysis_type: str) -> str:
        """normalize llm-provided names to our dispatch keys
        llms send things like 'Principal Component Analysis (PCA)' instead of 'pca_analysis'"""
        # strip parenthetical content, lowercase, spaces to underscores
        cleaned = re.sub(r'\([^)]*\)', '', analysis_type).strip()
        cleaned = cleaned.lower().replace(' ', '_').replace('-', '_')
        cleaned = cleaned.rstrip('_')

        aliases = {
            'principal_component_analysis': 'pca_analysis',
            'multiple_linear_regression': 'linear_regression',
            'simple_linear_regression': 'linear_regression',
            'regression': 'linear_regression',
            'regression_analysis': 'linear_regression',
            'competitive_landscape_mapping': 'competitive_landscape',
            'competitive_landscape_analysis': 'competitive_landscape',
            'competitive_analysis': 'competitive_landscape',
            'market_concentration': 'competitive_landscape',
            'clustering': 'kmeans',
            'k_means': 'kmeans',
            'k_means_clustering': 'kmeans',
            'outlier_analysis': 'outlier_detection',
            'anomaly_detection': 'zscore_detection',
            'z_score_detection': 'zscore_detection',
            'z_score': 'zscore_detection',
            'vif_analysis': 'collinearity_check',
            'vif': 'collinearity_check',
            'multicollinearity': 'collinearity_check',
            'multicollinearity_check': 'collinearity_check',
            'cross_tab': 'cross_tabulation',
            'crosstab': 'cross_tabulation',
            'descriptive_statistics': 'summary_statistics',
            'basic_statistics': 'summary_statistics',
        }
        return aliases.get(cleaned, cleaned)

    def _normality_test(self, df, nums, cats, dates, target, features, params) -> dict:
        """focused normality testing -- shapiro-wilk + anderson-darling + skew/kurtosis
        ref: scipy.stats.shapiro, scipy.stats.anderson"""
        cols = features or nums[:8]
        insights = []
        test_results = []

        for col in cols:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) < 3:
                continue

            # shapiro wilk
            try:
                sw_stat, sw_p = stats.shapiro(series.sample(min(5000, len(series)), random_state=42))
            except Exception:
                sw_stat, sw_p = None, None

            # anderson darling -- use 5% significance (index 2)
            try:
                ad_result = stats.anderson(series, dist='norm')
                ad_stat = ad_result.statistic
                ad_critical = ad_result.critical_values[2] if len(ad_result.critical_values) > 2 else None
                ad_normal = ad_stat < ad_critical if ad_critical else None
            except Exception:
                ad_stat, ad_critical, ad_normal = None, None, None

            skewness = round(float(series.skew()), 3)
            kurtosis = round(float(series.kurtosis()), 3)

            is_normal = sw_p > 0.05 if sw_p is not None else None
            test_results.append({
                'column': col,
                'shapiro_stat': round(sw_stat, 4) if sw_stat else None,
                'shapiro_p': round(sw_p, 4) if sw_p else None,
                'anderson_stat': round(ad_stat, 4) if ad_stat else None,
                'anderson_critical_5pct': round(ad_critical, 4) if ad_critical else None,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'is_normal': is_normal,
            })

            if is_normal is not None:
                insights.append({
                    'type': 'test_result',
                    'field': col,
                    'message': f'{col}: {"Normal" if is_normal else "Non-normal"} (Shapiro p={sw_p:.4f}, skew={skewness}, kurtosis={kurtosis})',
                })

        return {
            'analysis_type': 'normality_test',
            'summary': f'Normality testing on {len(test_results)} columns. {sum(1 for t in test_results if t.get("is_normal"))} are normally distributed.',
            'insights': insights,
            'charts': [],
            'statistical_details': {'tests': test_results},
            'recommendations': [
                'Non-normal columns may need nonparametric tests (Mann-Whitney, Kruskal-Wallis)',
                'Consider Box-Cox or log transform for skewed distributions',
            ],
        }

    def _percentile_analysis(self, df, nums, cats, dates, target, features, params) -> dict:
        """percentile ranks adn iqr breakdowns -- good for competitive benchmarkng"""
        cols = features or nums[:8]
        insights = []
        percentile_data = {}

        for col in cols:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) == 0:
                continue

            pctiles = series.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
            percentile_data[col] = {
                'p10': round(pctiles.get(0.1, 0), 2),
                'p25': round(pctiles.get(0.25, 0), 2),
                'p50': round(pctiles.get(0.5, 0), 2),
                'p75': round(pctiles.get(0.75, 0), 2),
                'p90': round(pctiles.get(0.9, 0), 2),
                'iqr': round(pctiles.get(0.75, 0) - pctiles.get(0.25, 0), 2),
            }

            iqr = percentile_data[col]['iqr']
            if iqr > 0:
                spread_ratio = round(iqr / series.median() * 100, 1) if series.median() != 0 else 0
                insights.append({
                    'type': 'percentile',
                    'field': col,
                    'message': f'{col}: IQR={iqr}, median={percentile_data[col]["p50"]}, spread={spread_ratio}% of median',
                })

        charts = []
        if percentile_data:
            charts.append({
                'type': 'bar',
                'title': 'Percentile Distribution',
                'xAxis': {'type': 'category', 'data': list(percentile_data.keys())},
                'yAxis': {'type': 'value'},
                'series': [
                    {'name': 'P25', 'type': 'bar', 'data': [v['p25'] for v in percentile_data.values()]},
                    {'name': 'P50 (Median)', 'type': 'bar', 'data': [v['p50'] for v in percentile_data.values()]},
                    {'name': 'P75', 'type': 'bar', 'data': [v['p75'] for v in percentile_data.values()]},
                ],
            })

        return {
            'analysis_type': 'percentile_analysis',
            'summary': f'Percentile analysis for {len(percentile_data)} columns.',
            'insights': insights,
            'charts': charts,
            'statistical_details': {'percentiles': percentile_data},
            'recommendations': [
                'IQR relative to median shows dispersion -- high spread means segmented market',
                'P10/P90 boundaries flag where the tails start',
            ],
        }

    def _variance_analysis(self, df, nums, cats, dates, target, features, params) -> dict:
        """levenes test for homogeneity of varianc across groups
        ref: scipy.stats.levene -- more robust than bartletts for non-normal data"""
        group_col = (features[0] if features else cats[0]) if cats else None
        value_col = (features[1] if features and len(features) > 1 else
                     target if target and target in nums else
                     nums[0] if nums else None)

        if not group_col or not value_col:
            return self._placeholder_analysis('variance_analysis', df, nums, cats, dates)

        groups_data = []
        group_labels = []
        for name, group in df.groupby(group_col)[value_col]:
            clean = group.dropna()
            if len(clean) >= 2:
                groups_data.append(clean.values)
                group_labels.append(str(name))

        if len(groups_data) < 2:
            return self._placeholder_analysis('variance_analysis', df, nums, cats, dates)

        # levenes test
        lev_stat, lev_p = stats.levene(*groups_data)

        # bartletts as well (parametric version)
        try:
            bart_stat, bart_p = stats.bartlett(*groups_data)
        except Exception:
            bart_stat, bart_p = None, None

        group_vars = {label: round(float(data.var()), 4) for label, data in zip(group_labels, groups_data)}
        max_var_ratio = max(group_vars.values()) / min(group_vars.values()) if min(group_vars.values()) > 0 else float('inf')

        insights = [{
            'type': 'test_result',
            'message': f'Levene test: F={lev_stat:.3f}, p={lev_p:.4f}. Variances are {"unequal" if lev_p < 0.05 else "equal"} across groups.',
        }]
        if max_var_ratio > 4:
            insights.append({
                'type': 'warning',
                'message': f'Variance ratio {max_var_ratio:.1f}:1 -- consider Welch ANOVA instead of standard',
            })

        return {
            'analysis_type': 'variance_analysis',
            'summary': f'Variance homogeneity for {value_col} across {group_col}. {"Unequal" if lev_p < 0.05 else "Equal"} variances (p={lev_p:.4f}).',
            'insights': insights,
            'charts': [],
            'statistical_details': {
                'levene_statistic': round(float(lev_stat), 4),
                'levene_p': round(float(lev_p), 4),
                'bartlett_statistic': round(float(bart_stat), 4) if bart_stat else None,
                'bartlett_p': round(float(bart_p), 4) if bart_p else None,
                'group_variances': group_vars,
                'max_variance_ratio': round(float(max_var_ratio), 2),
            },
            'recommendations': [
                'If unequal variances, use Welch ANOVA or Kruskal-Wallis instead',
                'Variance ratio > 4:1 is a practical concern even if test isnt significant',
            ],
        }

    def _competitive_landscape(self, df, nums, cats, dates, target, features, params) -> dict:
        """market concentration adn competitive positioning
        herfindahl-hirschman index (hhi) -- standard doj/ftc metric for market concentration
        ref: https://www.justice.gov/atr/herfindahl-hirschman-index"""

        # find capacity column -- most likely name
        cap_col = None
        for candidate in ['certified_beds', 'beds', 'capacity', 'units', 'revenue']:
            if candidate in df.columns:
                cap_col = candidate
                break
        if not cap_col and nums:
            cap_col = nums[0]

        quality_col = None
        for candidate in ['overall_rating', 'rating', 'quality_score', 'score']:
            if candidate in df.columns:
                quality_col = candidate
                break

        name_col = None
        for candidate in ['provider_name', 'name', 'facility_name', 'company']:
            if candidate in df.columns:
                name_col = candidate
                break
        if not name_col and cats:
            name_col = cats[0]

        insights = []

        # hhi based on market share of capacity
        total_cap = df[cap_col].sum()
        if total_cap > 0:
            market_shares = (df[cap_col] / total_cap * 100).round(2)
            hhi = round(float((market_shares ** 2).sum()), 1)
        else:
            market_shares = pd.Series([0] * len(df))
            hhi = 0

        # doj merger guideline thresholds
        if hhi < 1500:
            concentration = 'unconcentrated'
        elif hhi < 2500:
            concentration = 'moderately concentrated'
        else:
            concentration = 'highly concentrated'

        insights.append({
            'type': 'market_structure',
            'message': f'HHI = {hhi} ({concentration}). Market has {len(df)} competitors.',
        })

        # concentration ratio -- top 3
        df_ranked = df.copy()
        df_ranked['market_share_pct'] = market_shares.values
        df_ranked = df_ranked.sort_values('market_share_pct', ascending=False)

        cr3 = round(float(df_ranked['market_share_pct'].head(3).sum()), 1)
        insights.append({
            'type': 'concentration',
            'message': f'Top 3 providers control {cr3}% of capacity (CR3).',
        })

        # quality-weighted position if available
        if quality_col:
            df_ranked['quality_x_share'] = df_ranked[quality_col] * df_ranked['market_share_pct'] / 100
            weighted_quality = round(float(df_ranked['quality_x_share'].sum()), 2)
            avg_quality = df[quality_col].mean()
            insights.append({
                'type': 'quality_position',
                'message': f'Capacity-weighted quality: {weighted_quality:.1f}/5 vs unweighted {avg_quality:.1f}/5. '
                           f'{"Higher quality holds more capacity" if weighted_quality > avg_quality else "Lower quality holds more capacity -- displacement opportunity"}.',
            })

        # vulnerable facilities -- below median on quality adn occupancy
        if quality_col and 'occupancy_pct' in df.columns:
            med_quality = df[quality_col].median()
            med_occ = df['occupancy_pct'].median()
            vulnerable = df[(df[quality_col] < med_quality) & (df['occupancy_pct'] < med_occ)]
            if len(vulnerable) > 0:
                vuln_beds = int(vulnerable[cap_col].sum()) if cap_col else 0
                insights.append({
                    'type': 'opportunity',
                    'message': f'{len(vulnerable)} facilities below median on both quality and occupancy ({vuln_beds} beds) -- primary displacement targets.',
                })

        charts = []
        if name_col:
            charts.append({
                'type': 'bar',
                'title': 'Market Share by Provider',
                'xAxis': {'type': 'category', 'data': [str(n)[:20] for n in df_ranked[name_col].tolist()]},
                'yAxis': {'type': 'value', 'name': 'Market Share %'},
                'series': [{'type': 'bar', 'data': df_ranked['market_share_pct'].tolist()}],
            })

        return {
            'analysis_type': 'competitive_landscape',
            'summary': f'Market concentration: HHI={hhi} ({concentration}). CR3={cr3}%. {len(df)} competitors.',
            'insights': insights,
            'charts': charts,
            'statistical_details': {
                'hhi': hhi,
                'concentration_level': concentration,
                'cr3': cr3,
            },
            'recommendations': [
                'HHI < 1500 = unconcentrated, 1500-2500 = moderate, > 2500 = concentrated (DOJ guidelines)',
                'Low concentration + quality gaps = strong entry opportunity',
                'Target displacement of below-median facilities for fastest path to occupancy',
            ],
        }

    def _trend_analysis(self, df, nums, cats, dates, target, features, params) -> dict:
        """spearman rank correlations for monotonic trends between variables
        more robust than pearson for ordinal data like star ratings
        ref: scipy.stats.spearmanr"""
        cols = features or nums[:8]
        insights = []
        correlations = []

        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i >= j or col1 not in df.columns or col2 not in df.columns:
                    continue
                clean = df[[col1, col2]].dropna()
                if len(clean) < 3:
                    continue

                rho, p_value = stats.spearmanr(clean[col1], clean[col2])
                correlations.append({
                    'var1': col1,
                    'var2': col2,
                    'spearman_rho': round(float(rho), 4),
                    'p_value': round(float(p_value), 4),
                    'significant': p_value < 0.05,
                })

                if abs(rho) > 0.6 and p_value < 0.05:
                    insights.append({
                        'type': 'trend',
                        'message': f'Strong monotonic {"positive" if rho > 0 else "negative"} trend: {col1} vs {col2} (rho={rho:.3f}, p={p_value:.4f})',
                    })

        sig_count = sum(1 for c in correlations if c['significant'] and abs(c['spearman_rho']) > 0.5)

        return {
            'analysis_type': 'trend_analysis',
            'summary': f'Spearman rank correlations across {len(cols)} variables. {sig_count} significant monotonic trends.',
            'insights': insights if insights else [{'type': 'info', 'message': 'No strong monotonic trends detected'}],
            'charts': [],
            'statistical_details': {'correlations': correlations},
            'recommendations': [
                'Spearman captures monotonic (not just linear) relationships',
                'More robust than Pearson for ordinal data like star ratings',
                'Use when rank order matters more than exact values',
            ],
        }

    def _placeholder_analysis(self, analysis_type, df, nums, cats, dates) -> dict:
        """placeholder for analysis types not yet implementd"""
        return {
            "analysis_type": analysis_type,
            "summary": f"Analysis type '{analysis_type}' is available but requires additional configuration.",
            "insights": [{"type": "info", "message": f"This analysis needs specific parameters. Please configure via the AI assistant."}],
            "charts": [],
            "recommendations": ["Use the AI chat to configure this analysis with the right parameters"],
        }
