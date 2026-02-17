"""
time series forecasting service - ported from dalbey analytics forecasting module
supports holt-winters, arima, adn auto-selection
"""

import pandas as pd
import numpy as np
from typing import Optional

from src.data_cleaning import DataCleaningService


class ForecastingService:
    """time series forecasting"""

    def __init__(self):
        self.cleaner = DataCleaningService()

    def forecast(
        self,
        filepath: str,
        date_column: str,
        value_column: str,
        periods: int = 12,
        method: str = "auto",
    ) -> dict:
        """Generate time series forcast"""
        df = self.cleaner.load_file(filepath)

        if date_column not in df.columns or value_column not in df.columns:
            raise ValueError(f"Columns {date_column} or {value_column} not found")

        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)

        # aggregate if multiple values per date
        ts = df.groupby(date_column)[value_column].mean()
        ts = ts.dropna()

        if len(ts) < 10:
            raise ValueError("Need at least 10 data points for forecasting")

        if method == "auto":
            method = "holt_winters" if len(ts) >= 24 else "linear"

        if method == "holt_winters":
            result = self._holt_winters(ts, periods)
        elif method == "arima":
            result = self._arima(ts, periods)
        else:
            result = self._linear_trend(ts, periods)

        #build chart
        historical_dates = [str(d.date()) if hasattr(d, 'date') else str(d) for d in ts.index]
        forecast_dates = result["forecast_dates"]
        all_dates = historical_dates + forecast_dates

        charts = [{
            "type": "line",
            "title": f"{value_column} Forecast ({method})",
            "xAxis": {"type": "category", "data": all_dates},
            "yAxis": {"type": "value", "name": value_column},
            "series": [
                {
                    "name": "Historical",
                    "type": "line",
                    "data": ts.values.round(2).tolist() + [None] * periods,
                },
                {
                    "name": "Forecast",
                    "type": "line",
                    "data": [None] * len(ts) + result["values"],
                    "lineStyle": {"type": "dashed"},
                },
            ],
        }]

        forecast_data = [
            {"date": d, "value": round(v, 2)}
            for d, v in zip(forecast_dates, result["values"])
        ]

        return {
            "forecast": forecast_data,
            "model_info": {"method": method, "periods": periods, "data_points": len(ts)},
            "charts": charts,
            "summary": f"Forecasted {periods} periods using {method}. Trend: {'up' if result['values'][-1] > result['values'][0] else 'down'}.",
            "accuracy_metrics": result.get("metrics"),
        }



    
    def _holt_winters(self, ts: pd.Series, periods: int) -> dict:
        """holt-winters exponential smoothign"""
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        try:
            model = ExponentialSmoothing(
                ts, trend="add", seasonal=None, damped_trend=True
            ).fit()
            forecast = model.forecast(periods)

            last_date = ts.index[-1]
            freq = pd.infer_freq(ts.index) or "MS"
            forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]

            return {
                "values": forecast.values.round(2).tolist(),
                "forecast_dates": [str(d.date()) if hasattr(d, 'date') else str(d) for d in forecast_dates],
                "metrics": {"aic": round(float(model.aic), 2) if hasattr(model, 'aic') else None},
            }
        except Exception:
            return self._linear_trend(ts, periods)

    def _arima(self, ts: pd.Series, periods: int) -> dict:
        """arima forecasting"""
        try:
            from statsmodels.tsa.arima.model import ARIMA

            model = ARIMA(ts, order=(1, 1, 1)).fit()
            forecast = model.forecast(periods)

            last_date = ts.index[-1]
            freq = pd.infer_freq(ts.index) or "MS"
            forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]

            return {
                "values": forecast.values.round(2).tolist(),
                "forecast_dates": [str(d.date()) if hasattr(d, 'date') else str(d) for d in forecast_dates],
                "metrics": {"aic": round(float(model.aic), 2)},
            }
        except Exception:
            return self._linear_trend(ts, periods)

    def _linear_trend(self, ts: pd.Series, periods: int) -> dict:
        """simple linear trend extrapolaton"""
        x = np.arange(len(ts))
        coeffs = np.polyfit(x, ts.values, 1)

        future_x = np.arange(len(ts), len(ts) + periods)
        forecast_values = np.polyval(coeffs, future_x).round(2).tolist()

        last_date = ts.index[-1]
        try:
            freq = pd.infer_freq(ts.index) or "D"
            forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
        except Exception:
            forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq="D")[1:]

        return {
            "values": forecast_values,
            "forecast_dates": [str(d.date()) if hasattr(d, 'date') else str(d) for d in forecast_dates],
            "metrics": {"slope": round(float(coeffs[0]), 4), "intercept": round(float(coeffs[1]), 4)},
        }
