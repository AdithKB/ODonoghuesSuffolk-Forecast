"""
Split conformal prediction intervals for the hourly XGBoost forecast.

Uses walk-forward CV out-of-sample residuals to produce empirical coverage
bands segmented by hour-of-day. High-variance hours (late bar) get wider
bands than predictable hours (quiet morning).

The point forecast is NEVER modified. Bands are display-only context.
"""

import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR = Path("models")
COVERAGE = 0.80


def compute_hourly_quantiles(
    target: str = "orders_count",
    coverage: float = COVERAGE,
) -> dict:
    """
    Returns {hour: {"q_lo": float, "q_hi": float}} from CV residuals.
    lower_bound = forecast + q_lo (q_lo is negative when model over-predicts)
    upper_bound = forecast + q_hi
    """
    cv_path = MODELS_DIR / f"cv_out_of_sample_preds_{target}.csv"
    if not cv_path.exists():
        return {}

    df = pd.read_csv(cv_path)
    df["timestamp_hour"] = pd.to_datetime(df["timestamp_hour"])
    df["hour"] = df["timestamp_hour"].dt.hour

    if "residual" not in df.columns:
        pred_col = next((c for c in df.columns if "predict" in c.lower()), None)
        act_col  = next((c for c in df.columns if c in ("orders_count", "food_tickets_count")), None)
        if pred_col and act_col:
            df["residual"] = df[pred_col] - df[act_col]
        else:
            return {}

    alpha    = 1 - coverage
    q_lo_lvl = alpha / 2
    q_hi_lvl = 1 - alpha / 2

    global_res = df["residual"].dropna().values
    quantiles  = {}
    for h, grp in df.groupby("hour"):
        res = grp["residual"].dropna().values
        if len(res) < 8:
            res = global_res
        quantiles[int(h)] = {
            "q_lo": float(np.quantile(res, q_lo_lvl)),
            "q_hi": float(np.quantile(res, q_hi_lvl)),
        }

    return quantiles


def apply_intervals(
    forecast: pd.DataFrame,
    quantiles: dict,
    point_col: str = "orders_count_xgb",
    out_lo: str   = "orders_ci_lo",
    out_hi: str   = "orders_ci_hi",
) -> pd.DataFrame:
    """
    Adds lower/upper CI columns to forecast. Point forecast column unchanged.
    Lower bound is clipped at 0 (can't have negative orders).
    """
    if not quantiles or point_col not in forecast.columns:
        forecast[out_lo] = forecast[point_col]
        forecast[out_hi] = forecast[point_col]
        return forecast

    forecast = forecast.copy()
    lo = forecast["hour"].map(lambda h: quantiles.get(int(h), {}).get("q_lo", 0))
    hi = forecast["hour"].map(lambda h: quantiles.get(int(h), {}).get("q_hi", 0))

    forecast[out_lo] = (forecast[point_col] + lo).clip(lower=0).round(1)
    forecast[out_hi] = (forecast[point_col] + hi).clip(lower=0).round(1)
    return forecast
