"""
Forecasting pipeline for O'Donoghues Suffolk Street demand prediction.

Two models per target (orders_count, food_tickets_count):
  1. Baseline   — same-hour-last-week (lag_168h) with rolling-mean fallback
  2. XGBoost    — gradient-boosted trees on full feature table

Evaluation via walk-forward (rolling-origin) cross-validation.
Models saved to models/ for dashboard consumption.
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from dataclasses import dataclass, field

import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore", category=FutureWarning)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

TARGETS = ["orders_count", "food_tickets_count"]

# Shift definitions: (name, start_hour_inclusive, end_hour_inclusive)
SHIFTS = [
    ("lunch",   12, 14),
    ("dinner",  17, 20),
    ("late_bar", 21, 1),   # wraps midnight; handled separately
]

# Shift segments for per-shift model training
SHIFT_SEGMENTS = {
    "lunch":    lambda h: (h >= 12) & (h <= 15),
    "evening":  lambda h: (h >= 17) & (h <= 20),
    "late_bar": lambda h: (h >= 21) | (h <= 2),
    "off_peak": lambda h: ((h >= 9) & (h <= 11)) | (h == 16),
}


def get_hour_shift(hour: int) -> str:
    """Map a single hour value to its shift bucket name."""
    if 12 <= hour <= 15:
        return "lunch"
    if 17 <= hour <= 20:
        return "evening"
    if hour >= 21 or hour <= 2:
        return "late_bar"
    return "off_peak"

# Busy-label thresholds (percentile of training shift totals per shift type)
LABEL_THRESHOLDS = {"quiet": 0.25, "normal": 0.65, "busy": 0.88}


# ---------------------------------------------------------------------------
# Feature column selection
# ---------------------------------------------------------------------------

# Features safe to use when predicting the NEXT day (no same-day info needed)
SAFE_FOR_NEXT_DAY = [
    "orders_count_lag_24h", "orders_count_lag_48h",
    "orders_count_lag_168h", "orders_count_lag_336h",
    "food_tickets_count_lag_24h", "food_tickets_count_lag_48h",
    "food_tickets_count_lag_168h", "food_tickets_count_lag_336h",
    "orders_count_roll_mean_24", "orders_count_roll_mean_168",
    "orders_count_roll_std_24",
    "food_tickets_count_roll_mean_24", "food_tickets_count_roll_mean_168",
    "hour", "weekday", "month", "quarter", "is_weekend", "is_friday_saturday",
    "week_of_year", "bank_holiday_flag", "school_holiday_flag", "payday_period_flag",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos",
    "is_food_service", "is_breakfast_window", "is_lunch_window",
    "is_dinner_window", "is_after_food_close", "hours_since_food_close",
    "is_live_music_window",
    "temp_c", "rain_mm", "wind_speed_kmh", "weather_severity_flag",
    "airport_arrivals", "airport_arrivals_lag1", "airport_arrivals_zscore",
    "cruise_ship_flag", "ships_in_port_count", "cruise_passenger_estimate",
    "major_sports_event_flag", "city_event_flag", "st_patricks_week_flag",
    "special_event_flag", "event_intensity", "tourism_pressure",
    "weekend_x_music", "rain_x_weekend",
    # Events enrichment
    "aviva_event_flag", "croke_park_event_flag", "nearby_venue_event_flag",
    "event_impact_score",
    "bloomsday_flag", "bloomsday_week_flag", "summer_tourism_flag",
    "christmas_market_flag", "new_years_eve_flag", "new_years_day_flag",
    "college_term_flag", "days_from_payday",
    "failte_event_count", "failte_free_event_count", "failte_festival_count",
    # City pedestrian footfall lags (safe — use yesterday/last-week, not current hour)
    "suffolk_footfall_lag_24h", "suffolk_footfall_lag_168h",
    "suffolk_footfall_roll_24h",
    # Same-slot seasonal averages (4-week lookback, no same-day leakage)
    "orders_count_same_slot_4w_avg",
    "food_tickets_count_same_slot_4w_avg",
]

# Full feature set (adds intra-day lags — usable for same-day nowcasting)
FULL_FEATURES = SAFE_FOR_NEXT_DAY + [
    "orders_count_lag_1", "orders_count_lag_2", "orders_count_lag_3",
    "food_tickets_count_lag_1", "food_tickets_count_lag_2", "food_tickets_count_lag_3",
    "orders_count_roll_mean_3", "orders_count_roll_mean_6", "orders_count_roll_mean_12",
    "food_tickets_count_roll_mean_3", "food_tickets_count_roll_mean_6",
    # Current-hour footfall (usable intra-day when counter data is available)
    "suffolk_nassau_footfall", "nearby_footfall_avg", "city_footfall_zscore",
    "suffolk_footfall_lag_1h", "suffolk_is_busy_flag",
]


def get_feature_cols(df: pd.DataFrame, feature_list: list[str]) -> list[str]:
    return [c for c in feature_list if c in df.columns]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold: int
    cutoff: pd.Timestamp
    n_train: int
    n_test: int
    mae: float
    rmse: float
    mape: float
    shift_accuracy: float | None = None
    y_true: np.ndarray = field(default_factory=lambda: np.array([]))
    y_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    mask = y_true > eps
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred = np.maximum(y_pred, 0)
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mape(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Shift-label helpers
# ---------------------------------------------------------------------------

def assign_shift_labels(
    df: pd.DataFrame, target: str, thresholds: dict | None = None
) -> pd.Series:
    """
    Assign busy/quiet labels at shift level.
    thresholds: dict with keys 'quiet', 'normal', 'busy' as percentile values
                derived from training data shift totals.
    """
    h = df["hour"] if "hour" in df.columns else df["timestamp_hour"].dt.hour

    # Lunch (12-14)
    lunch_mask = h.between(12, 14)
    # Dinner (17-20)
    dinner_mask = h.between(17, 20)
    # Late bar (21-23 + 0-1)
    late_mask = (h >= 21) | (h <= 1)

    shift_col = pd.Series("none", index=df.index)
    shift_col[lunch_mask] = "lunch"
    shift_col[dinner_mask] = "dinner"
    shift_col[late_mask] = "late_bar"
    return shift_col


def shift_totals(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Compute per-date per-shift demand totals."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp_hour"]).dt.date
    df["hour"] = pd.to_datetime(df["timestamp_hour"]).dt.hour
    df["shift"] = assign_shift_labels(df, target)
    return (
        df[df["shift"] != "none"]
        .groupby(["date", "shift"])[target]
        .sum()
        .reset_index()
    )


def label_from_thresholds(val: float, thresholds: dict) -> str:
    if val <= thresholds["quiet"]:
        return "quiet"
    elif val <= thresholds["normal"]:
        return "normal"
    elif val <= thresholds["busy"]:
        return "busy"
    else:
        return "slammed"


def compute_shift_label_accuracy(
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
    target: str,
    train_shift_totals: pd.DataFrame,
) -> float:
    """
    Classify predicted and actual shift totals into busy labels,
    return fraction of shifts where predicted label matches actual label.
    """
    # Build thresholds from training data
    pct = train_shift_totals.groupby("shift")[target].quantile([0.25, 0.65, 0.88])
    thresholds_by_shift = {}
    for shift_name in pct.index.get_level_values("shift").unique():
        q = pct.loc[shift_name].values  # use .values to avoid float index issues
        thresholds_by_shift[shift_name] = {
            "quiet": float(q[0]),   # 0.25 quantile
            "normal": float(q[1]),  # 0.65 quantile
            "busy": float(q[2]),    # 0.88 quantile
        }

    # Use only the columns needed to avoid duplicate-column issues
    true_totals = shift_totals(df_true[["timestamp_hour", target]].copy(), target)

    # df_pred has a "pred" column — rename cleanly without duplicates
    pred_slim = df_pred[["timestamp_hour", "pred"]].copy().rename(columns={"pred": target})
    pred_totals = shift_totals(pred_slim, target)

    merged = true_totals.merge(pred_totals, on=["date", "shift"], suffixes=("_true", "_pred"))
    if merged.empty:
        return np.nan

    correct = 0
    for _, row in merged.iterrows():
        shift_name = str(row["shift"])
        thr = thresholds_by_shift.get(shift_name, {"quiet": 10, "normal": 30, "busy": 50})
        true_label = label_from_thresholds(float(row[f"{target}_true"]), thr)
        pred_label = label_from_thresholds(float(row[f"{target}_pred"]), thr)
        correct += int(true_label == pred_label)

    return correct / len(merged)


# ---------------------------------------------------------------------------
# Baseline predictor
# ---------------------------------------------------------------------------

class BaselinePredictor:
    """
    Same-hour-last-week (lag_168h) with rolling-mean fallback.
    Mirrors what a manager does intuitively: "how was it this time last week?"
    """

    def __init__(self):
        self._history: dict[tuple, list] = {}   # (hour, weekday) -> recent values

    def fit(self, df: pd.DataFrame, target: str) -> "BaselinePredictor":
        self._history.clear()
        df = df.copy()
        df["hour"] = pd.to_datetime(df["timestamp_hour"]).dt.hour
        df["weekday"] = pd.to_datetime(df["timestamp_hour"]).dt.weekday
        for _, row in df.iterrows():
            key = (int(row["hour"]), int(row["weekday"]))
            self._history.setdefault(key, []).append(row[target])
        return self

    def predict(self, df: pd.DataFrame, target: str) -> np.ndarray:
        df = df.copy()
        df["hour"] = pd.to_datetime(df["timestamp_hour"]).dt.hour
        df["weekday"] = pd.to_datetime(df["timestamp_hour"]).dt.weekday

        # Prefer lag_168h column if present (exact same slot last week)
        lag_col = f"{target}_lag_168h"
        if lag_col in df.columns:
            lag_vals = df[lag_col].values.copy()
            # Fill remaining NaN with slot rolling mean
            for i, (_, row) in enumerate(df.iterrows()):
                if np.isnan(lag_vals[i]):
                    key = (int(row["hour"]), int(row["weekday"]))
                    hist = self._history.get(key, [])
                    lag_vals[i] = np.mean(hist[-8:]) if hist else 0.0
            return np.maximum(lag_vals, 0)

        # Fallback: slot rolling mean from fit history
        preds = []
        for _, row in df.iterrows():
            key = (int(row["hour"]), int(row["weekday"]))
            hist = self._history.get(key, [])
            preds.append(np.mean(hist[-8:]) if hist else 0.0)
        return np.maximum(np.array(preds), 0)


# ---------------------------------------------------------------------------
# XGBoost model
# ---------------------------------------------------------------------------

XGB_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
    "eval_metric": "mae",
}


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def feature_importance_df(model: xgb.XGBRegressor, feature_cols: list[str]) -> pd.DataFrame:
    scores = model.get_booster().get_score(importance_type="gain")
    rows = [{"feature": k, "importance": v} for k, v in scores.items()]
    df = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    df["importance_pct"] = df["importance"] / df["importance"].sum() * 100
    return df


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------

def walk_forward_cv(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    n_splits: int = 8,
    test_weeks: int = 4,
    min_train_weeks: int = 12,
    verbose: bool = True,
) -> tuple[list[FoldResult], list[FoldResult]]:
    """
    Rolling-origin evaluation. Each fold:
      - Train on all data up to cutoff
      - Evaluate on next `test_weeks` weeks
      - Advance cutoff by `test_weeks`

    Returns (baseline_results, xgb_results)
    """
    df = df.copy()
    df["timestamp_hour"] = pd.to_datetime(df["timestamp_hour"])
    df = df.sort_values("timestamp_hour").reset_index(drop=True)

    t_min = df["timestamp_hour"].min()
    t_max = df["timestamp_hour"].max()

    min_train = pd.Timedelta(weeks=min_train_weeks)
    test_delta = pd.Timedelta(weeks=test_weeks)

    # Generate cutoff dates
    first_cutoff = t_min + min_train
    last_cutoff = t_max - test_delta
    cutoffs = pd.date_range(first_cutoff, last_cutoff, periods=n_splits)

    baseline_results, xgb_results = [], []

    for fold_idx, cutoff in enumerate(cutoffs):
        train_df = df[df["timestamp_hour"] <= cutoff].copy()
        test_df  = df[(df["timestamp_hour"] > cutoff) &
                      (df["timestamp_hour"] <= cutoff + test_delta)].copy()

        if len(test_df) == 0:
            continue

        # Exclude rows where target is NaN
        train_df = train_df.dropna(subset=[target] + [c for c in feature_cols if c in train_df.columns])
        test_df  = test_df.dropna(subset=[target])

        y_train = train_df[target].values
        y_test  = test_df[target].values

        feat_avail = [c for c in feature_cols if c in train_df.columns]
        X_train = train_df[feat_avail].fillna(0)
        X_test  = test_df[feat_avail].fillna(0)

        # --- Baseline ---
        baseline = BaselinePredictor()
        baseline.fit(train_df, target)
        b_pred = baseline.predict(test_df, target)
        bm = compute_metrics(y_test, b_pred)

        # Shift-label accuracy
        train_st = shift_totals(train_df, target)
        test_df_copy = test_df.copy()
        test_df_copy["pred"] = b_pred
        b_shift_acc = compute_shift_label_accuracy(test_df, test_df_copy, target, train_st)

        baseline_results.append(FoldResult(
            fold=fold_idx, cutoff=cutoff,
            n_train=len(train_df), n_test=len(test_df),
            mae=bm["mae"], rmse=bm["rmse"], mape=bm["mape"],
            shift_accuracy=b_shift_acc,
            y_true=y_test, y_pred=b_pred,
            timestamps=test_df["timestamp_hour"].values
        ))

        # --- XGBoost ---
        # Use last 20% of training data as internal val set for early stopping
        val_size = max(int(len(X_train) * 0.2), 200)
        X_val_es = X_train.iloc[-val_size:]
        y_val_es = y_train[-val_size:]
        X_tr_es  = X_train.iloc[:-val_size]
        y_tr_es  = y_train[:-val_size]

        xgb_model = train_xgboost(X_tr_es, y_tr_es, X_val_es, y_val_es)
        x_pred_raw = xgb_model.predict(X_test)
        x_pred = np.maximum(x_pred_raw, 0)
        xm = compute_metrics(y_test, x_pred)

        test_df_xgb = test_df.copy()
        test_df_xgb["pred"] = x_pred
        x_shift_acc = compute_shift_label_accuracy(test_df, test_df_xgb, target, train_st)

        xgb_results.append(FoldResult(
            fold=fold_idx, cutoff=cutoff,
            n_train=len(train_df), n_test=len(test_df),
            mae=xm["mae"], rmse=xm["rmse"], mape=xm["mape"],
            shift_accuracy=x_shift_acc,
            y_true=y_test, y_pred=x_pred,
            timestamps=test_df["timestamp_hour"].values
        ))

        if verbose:
            print(
                f"  Fold {fold_idx+1:2d} | cutoff {cutoff.date()} | "
                f"train={len(train_df):5,} | test={len(test_df):4,} | "
                f"Baseline MAE={bm['mae']:.2f}  XGB MAE={xm['mae']:.2f}  "
                f"XGB shift-acc={x_shift_acc:.0%}"
            )

    return baseline_results, xgb_results


def summarise_cv(results: list[FoldResult], label: str) -> pd.DataFrame:
    rows = [
        {"model": label, "fold": r.fold, "cutoff": r.cutoff,
         "mae": r.mae, "rmse": r.rmse, "mape": r.mape,
         "shift_accuracy": r.shift_accuracy}
        for r in results
    ]
    df = pd.DataFrame(rows)
    print(f"\n{label} — CV summary:")
    print(f"  MAE:  {df['mae'].mean():.2f} ± {df['mae'].std():.2f}")
    print(f"  RMSE: {df['rmse'].mean():.2f} ± {df['rmse'].std():.2f}")
    print(f"  MAPE: {df['mape'].mean():.1f}% ± {df['mape'].std():.1f}%")
    if df["shift_accuracy"].notna().any():
        print(f"  Shift accuracy: {df['shift_accuracy'].mean():.1%} ± {df['shift_accuracy'].std():.1%}")
    return df


# ---------------------------------------------------------------------------
# Final model training (full dataset)
# ---------------------------------------------------------------------------

def train_final_models(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
) -> tuple[BaselinePredictor, xgb.XGBRegressor, list[str]]:
    """Train on the full dataset for deployment."""
    df = df.dropna(subset=[target]).copy()
    feat_avail = [c for c in feature_cols if c in df.columns]

    # Baseline
    baseline = BaselinePredictor()
    baseline.fit(df, target)

    # XGBoost — use last 10% as validation for early stopping
    y = df[target].values
    X = df[feat_avail].fillna(0)
    val_size = max(int(len(X) * 0.10), 200)
    xgb_model = train_xgboost(
        X.iloc[:-val_size], y[:-val_size],
        X.iloc[-val_size:], y[-val_size:],
    )

    return baseline, xgb_model, feat_avail


def predict_next_day(
    df_history: pd.DataFrame,
    target: str,
    baseline: BaselinePredictor,
    xgb_model: xgb.XGBRegressor,
    feature_cols: list[str],
    target_date: str | None = None,
) -> pd.DataFrame:
    """
    Given history up to today, predict all hours for the next day.
    Uses safe-for-next-day features only (no same-day intra-hour lags).
    Returns dataframe with columns: timestamp_hour, baseline_pred, xgb_pred, shift.
    """
    from src.features import build_features
    from src.synthetic import BASE_ORDERS, FOOD_TICKET_RATIO

    if target_date is None:
        target_date = (pd.Timestamp.now() + pd.Timedelta(days=1)).date()

    # Build a stub for target day rows using history features
    # In production: replace with actual weather forecast + known event flags
    last_known = df_history.sort_values("timestamp_hour").iloc[-1]
    target_dt = pd.Timestamp(target_date)
    hours = list(range(9, 24)) + [0, 1]

    stub_rows = []
    for h in hours:
        dt = target_dt + pd.Timedelta(hours=h)
        row = {"timestamp_hour": dt, "orders_count": np.nan, "food_tickets_count": np.nan}
        # Copy external/calendar signals from history (weather assumed same as yesterday)
        for col in ["temp_c", "rain_mm", "wind_speed_kmh", "weather_severity_flag",
                    "airport_arrivals", "cruise_ship_flag", "ships_in_port_count",
                    "bank_holiday_flag", "school_holiday_flag", "major_sports_event_flag",
                    "city_event_flag", "st_patricks_week_flag", "special_event_flag"]:
            row[col] = last_known.get(col, 0)
        stub_rows.append(row)

    stub_df = pd.DataFrame(stub_rows)
    combined = pd.concat([df_history, stub_df], ignore_index=True)
    combined_feat = build_features(combined, targets=TARGETS, drop_na=False)
    pred_rows = combined_feat[combined_feat["timestamp_hour"].dt.date == pd.Timestamp(target_date).date()]

    feat_avail = [c for c in feature_cols if c in pred_rows.columns]
    X_pred = pred_rows[feat_avail].fillna(0)

    b_preds = baseline.predict(pred_rows, target)
    x_preds = np.maximum(xgb_model.predict(X_pred), 0)

    pred_rows = pred_rows.copy()
    pred_rows["baseline_pred"] = b_preds
    pred_rows["xgb_pred"] = x_preds
    pred_rows["shift"] = assign_shift_labels(pred_rows, target)

    return pred_rows[["timestamp_hour", "baseline_pred", "xgb_pred", "shift"]]


# ---------------------------------------------------------------------------
# Asymmetric loss (penalise under-predictions 2× more)
# ---------------------------------------------------------------------------

def _asymmetric_obj(y_pred: np.ndarray, dtrain: xgb.DMatrix):
    """
    Custom XGBoost gradient/hessian for asymmetric squared loss.
    Under-predictions (pred < true) receive 2× the penalty of over-predictions.

    Loss:
        2*(y_true - y_pred)^2  when y_pred < y_true  (under)
        (y_true - y_pred)^2    otherwise              (over)

    Gradient  dL/dy_pred:
        Under: 4*(y_pred - y_true)   |  Hessian: 4
        Over:  2*(y_pred - y_true)   |  Hessian: 2
    """
    y_true = dtrain.get_label()
    residual = y_pred - y_true
    is_under = residual < 0
    grad = np.where(is_under, 4.0 * residual, 2.0 * residual)
    hess = np.where(is_under, 4.0, 2.0)
    return grad, hess


# ---------------------------------------------------------------------------
# Optuna hyperparameter tuning with walk-forward CV
# ---------------------------------------------------------------------------

def _make_optuna_objective(df_shift: pd.DataFrame, target: str, feature_cols: list[str]):
    """Factory: returns an Optuna objective closure for a specific shift+target."""

    df_s = df_shift.dropna(subset=[target]).sort_values("timestamp_hour").reset_index(drop=True)
    feat_avail = [c for c in feature_cols if c in df_s.columns]
    X_all = df_s[feat_avail].fillna(0).values
    y_all = df_s[target].values
    n = len(df_s)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_boost_round":  trial.suggest_int("n_estimators", 100, 800),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 3.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 3.0),
            "seed": 42,
            "verbosity": 0,
        }
        n_rounds = params.pop("num_boost_round")

        if n < 80:
            return float("inf")

        maes = []
        # 3-fold walk-forward CV
        for fold in range(3):
            train_end = int(n * (fold + 1) / 4)
            test_start = train_end
            test_end   = int(n * (fold + 2) / 4)
            if test_end > n or train_end < 30 or (test_end - test_start) < 10:
                continue

            X_tr, y_tr = X_all[:train_end], y_all[:train_end]
            X_te, y_te = X_all[test_start:test_end], y_all[test_start:test_end]

            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dtest  = xgb.DMatrix(X_te, label=y_te)

            bst = xgb.train(
                params, dtrain,
                num_boost_round=n_rounds,
                obj=_asymmetric_obj,
                verbose_eval=False,
            )
            preds = np.maximum(bst.predict(dtest), 0)
            maes.append(mean_absolute_error(y_te, preds))

        return float(np.mean(maes)) if maes else float("inf")

    return objective, feat_avail, df_s, X_all, y_all


def tune_and_train_shift_model(
    df: pd.DataFrame,
    target: str,
    shift_name: str,
    shift_mask_fn,
    feature_cols: list[str],
    n_trials: int = 50,
) -> tuple[xgb.Booster | None, dict, list[str]]:
    """
    Run Optuna (n_trials) for one shift+target, train final model on full shift data.

    Returns (booster, best_params, feat_avail) — booster is None if insufficient data.
    """
    hour_col = (
        df["timestamp_hour"].dt.hour
        if "timestamp_hour" in df.columns
        else df["hour"]
    )
    mask = shift_mask_fn(hour_col)
    df_shift = df[mask].copy()

    if len(df_shift) < 80:
        print(f"    [{shift_name}] Skipping — only {len(df_shift)} rows.")
        return None, {}, []

    obj_fn, feat_avail, df_s, X_all, y_all = _make_optuna_objective(
        df_shift, target, feature_cols
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(obj_fn, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params.copy()
    n_rounds = best.pop("n_estimators")

    # Train final model on full shift data with best params
    xgb_params = {
        "max_depth":        best["max_depth"],
        "learning_rate":    best["learning_rate"],
        "min_child_weight": best["min_child_weight"],
        "subsample":        best["subsample"],
        "colsample_bytree": best["colsample_bytree"],
        "reg_alpha":        best["reg_alpha"],
        "reg_lambda":       best["reg_lambda"],
        "seed": 42,
        "verbosity": 0,
    }
    dtrain = xgb.DMatrix(X_all, label=y_all, feature_names=feat_avail)
    bst = xgb.train(
        xgb_params, dtrain,
        num_boost_round=n_rounds,
        obj=_asymmetric_obj,
    )
    return bst, study.best_params, feat_avail


def train_shift_models(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    global_model: xgb.XGBRegressor,
    n_trials: int = 50,
) -> None:
    """
    Train and save per-shift XGBoost models with Optuna-tuned asymmetric loss.
    Prints before/after MAE per shift.
    """
    print(f"\n  --- Shift-specific models for {target} ---")

    hour_col = df["timestamp_hour"].dt.hour
    feat_avail_global = [c for c in feature_cols if c in df.columns]

    for shift_name, mask_fn in SHIFT_SEGMENTS.items():
        mask = mask_fn(hour_col)
        df_shift = df[mask].dropna(subset=[target]).copy()

        if len(df_shift) < 80:
            print(f"    [{shift_name}] Insufficient data ({len(df_shift)} rows), skipping.")
            continue

        X_shift = df_shift[feat_avail_global].fillna(0)
        y_shift = df_shift[target].values

        # Before MAE — global model on this shift's data
        before_preds = np.maximum(global_model.predict(X_shift), 0)
        before_mae   = mean_absolute_error(y_shift, before_preds)

        print(f"    [{shift_name}] {len(df_shift)} rows | Global MAE={before_mae:.2f} | tuning…")

        bst, best_params, feat_used = tune_and_train_shift_model(
            df, target, shift_name, mask_fn, feature_cols, n_trials=n_trials
        )

        if bst is None:
            continue

        # After MAE — shift model on same data
        dshift = xgb.DMatrix(df_shift[feat_used].fillna(0).values, feature_names=feat_used)
        after_preds = np.maximum(bst.predict(dshift), 0)
        after_mae   = mean_absolute_error(y_shift, after_preds)

        print(
            f"    [{shift_name}] After MAE={after_mae:.2f}  "
            f"(Δ={after_mae - before_mae:+.2f})"
        )

        # Save shift model
        model_path = MODELS_DIR / f"xgb_{target}_{shift_name}.json"
        bst.save_model(str(model_path))
        print(f"    [{shift_name}] Saved → {model_path}")

        # Save best params
        params_path = MODELS_DIR / f"best_params_{target}_{shift_name}.json"
        with open(params_path, "w") as fh:
            json.dump({"best_params": best_params, "feat_avail": feat_used}, fh, indent=2)
        print(f"    [{shift_name}] Params → {params_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    features_path = Path("data/processed/features.parquet")
    print(f"Loading {features_path} ...")
    df = pd.read_parquet(features_path)
    print(f"  Shape: {df.shape}")

    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"TARGET: {target}")
        print(f"{'='*60}")

        feature_cols = get_feature_cols(df, SAFE_FOR_NEXT_DAY)
        print(f"  Features used: {len(feature_cols)}")

        b_results, x_results = walk_forward_cv(
            df, target, feature_cols,
            n_splits=8, test_weeks=4, min_train_weeks=12,
        )

        b_summary = summarise_cv(b_results, "Baseline")
        x_summary = summarise_cv(x_results, "XGBoost")

        improvement = (b_summary["mae"].mean() - x_summary["mae"].mean()) / b_summary["mae"].mean() * 100
        print(f"\n  XGBoost MAE improvement over baseline: {improvement:+.1f}%")

        # Export out-of-sample CV predictions for rigorous EDA
        cv_preds = []
        for r in x_results:
            df_fold = pd.DataFrame({
                "timestamp_hour": r.timestamps,
                "orders_count": r.y_true,
                "predicted_orders": r.y_pred,
                "fold": r.fold
            })
            cv_preds.append(df_fold)
        if cv_preds:
            cv_df = pd.concat(cv_preds, ignore_index=True)
            cv_df["residual"] = cv_df["predicted_orders"] - cv_df["orders_count"]
            cv_out_path = MODELS_DIR / f"cv_out_of_sample_preds_{target}.csv"
            cv_df.to_csv(cv_out_path, index=False)
            print(f"  Saved CV out-of-sample predictions to {cv_out_path}")

        # Train final model on full data
        print(f"\n  Training final model on full dataset...")
        baseline_final, xgb_final, feat_used = train_final_models(df, target, feature_cols)

        # Feature importance (top 15)
        fi = feature_importance_df(xgb_final, feat_used)
        print(f"\n  Top 15 features by gain:")
        print(fi.head(15)[["feature", "importance_pct"]].to_string(index=False))

        # Save
        joblib.dump(baseline_final, MODELS_DIR / f"baseline_{target}.pkl")
        xgb_final.save_model(str(MODELS_DIR / f"xgb_{target}.json"))
        fi.to_csv(MODELS_DIR / f"feature_importance_{target}.csv", index=False)
        print(f"\n  Models saved to {MODELS_DIR}/")

        # Train shift-specific models with Optuna asymmetric tuning
        print(f"\n  Running Optuna shift-specific tuning (50 trials × 4 shifts)…")
        train_shift_models(df, target, feature_cols, xgb_final, n_trials=50)

if __name__ == "__main__":
    main()
