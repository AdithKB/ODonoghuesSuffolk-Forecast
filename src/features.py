"""
Feature engineering pipeline for O'Donoghues Suffolk Street demand forecasting.

Takes raw hourly data (synthetic or real POS export) and produces a leakage-safe
supervised forecasting table. All lag and rolling features use only past data.

Design rule: no feature at time T may use any information from T or later.
- Row-based lags (lag_1, lag_2, lag_3): previous consecutive bar-open hours.
- Timestamp-based lags (lag_24h, lag_168h): exact calendar lookups via index join.
- Rolling windows: always shifted by 1 row before applying, so window ends at T-1.
"""

import numpy as np
import pandas as pd
import holidays
from pathlib import Path



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FOOD_CLOSE_HOUR = 21
BREAKFAST_END_HOUR = 15
LUNCH_START, LUNCH_END = 12, 14
DINNER_START, DINNER_END = 17, 20
MUSIC_WEEKDAY_START = 22   # Sun-Thu
MUSIC_WEEKEND_START = 22   # Fri-Sat (10pm)
MUSIC_WEEKEND_END = 26     # 2am (hour 26 in 24h+ convention)


# ---------------------------------------------------------------------------
# Timestamp-based lag helper
# ---------------------------------------------------------------------------
def _lag_by_timedelta(
    series: pd.Series, delta: pd.Timedelta, name: str
) -> pd.Series:
    """
    For each timestamp T, look up the value at T - delta.
    This gives true calendar-aligned lags (e.g. same hour yesterday).
    Uses an index join rather than row shift to handle gaps correctly.
    """
    lookup = series.rename("value")
    shifted_index = series.index - delta
    result = shifted_index.map(lookup.to_dict())
    return pd.Series(result, index=series.index, name=name)


# ---------------------------------------------------------------------------
# Cyclic encoding
# ---------------------------------------------------------------------------
def _cyclic(series: pd.Series, period: int, name: str) -> pd.DataFrame:
    sin = np.sin(2 * np.pi * series / period).rename(f"{name}_sin")
    cos = np.cos(2 * np.pi * series / period).rename(f"{name}_cos")
    return pd.concat([sin, cos], axis=1)


# ---------------------------------------------------------------------------
# Irish holiday helpers
# ---------------------------------------------------------------------------
def _build_holiday_set(years: list[int]) -> set:
    ie = holidays.Ireland(years=years)
    return set(ie.keys())


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Two types of lags:
    - Row-based (lag_1, lag_2, lag_3): previous consecutive open hours.
      Useful for within-shift momentum.
    - Timestamp-based (lag_24h, lag_48h, lag_168h, lag_336h): same clock hour
      on prior days/weeks. Useful for day-of-week and weekly seasonality.
    """
    s = df[target]

    # Row-based: previous N bar-open hours
    for n in [1, 2, 3]:
        df[f"{target}_lag_{n}"] = s.shift(n)

    # Timestamp-based: true calendar alignment (handles overnight gaps)
    ts_series = df.set_index("timestamp_hour")[target]
    for hours, label in [(24, "lag_24h"), (48, "lag_48h"),
                         (168, "lag_168h"), (336, "lag_336h")]:
        lagged = _lag_by_timedelta(ts_series, pd.Timedelta(hours=hours), label)
        df[f"{target}_{label}"] = lagged.values

    return df


def add_rolling_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    All rolling windows are computed on the sorted row sequence, then shifted by 1
    so the window ends at T-1 (no leakage). The row sequence follows bar-open hours,
    so roll_mean_6 captures the last 6 bar-open hours, not 6 clock hours.
    """
    s = df[target]

    windows = {
        "roll_mean_3": (3, "mean"),
        "roll_mean_6": (6, "mean"),
        "roll_mean_12": (12, "mean"),
        "roll_mean_24": (24, "mean"),
        "roll_std_24": (24, "std"),
        "roll_max_24": (24, "max"),
        "roll_mean_168": (168, "mean"),   # 1-week rolling avg (bar-open hours)
    }

    for col, (w, agg) in windows.items():
        rolled = getattr(s.shift(1).rolling(w, min_periods=max(1, w // 4)), agg)()
        df[f"{target}_{col}"] = rolled

    return df


def add_calendar_features(df: pd.DataFrame, holiday_set: set) -> pd.DataFrame:
    ts = df["timestamp_hour"]

    df["hour"] = ts.dt.hour
    df["weekday"] = ts.dt.weekday   # 0=Mon, 6=Sun
    df["month"] = ts.dt.month
    df["quarter"] = ts.dt.quarter
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["is_weekend"] = ts.dt.weekday.isin([5, 6]).astype(int)
    df["is_friday_saturday"] = ts.dt.weekday.isin([4, 5]).astype(int)

    # Cyclic encodings — tree models don't need these, but useful if you try linear models
    df = pd.concat([df, _cyclic(df["hour"], 24, "hour")], axis=1)
    df = pd.concat([df, _cyclic(df["weekday"], 7, "weekday")], axis=1)
    df = pd.concat([df, _cyclic(df["month"], 12, "month")], axis=1)

    # Holiday / calendar flags
    date_col = ts.dt.date
    df["bank_holiday_flag"] = date_col.map(lambda d: d in holiday_set).astype(int)

    return df


def add_venue_features(df: pd.DataFrame) -> pd.DataFrame:
    h = df["timestamp_hour"].dt.hour

    df["is_food_service"] = (h < FOOD_CLOSE_HOUR).astype(int)
    df["is_breakfast_window"] = (h < BREAKFAST_END_HOUR).astype(int)
    df["is_lunch_window"] = ((h >= LUNCH_START) & (h <= LUNCH_END)).astype(int)
    df["is_dinner_window"] = ((h >= DINNER_START) & (h <= DINNER_END)).astype(int)
    df["is_after_food_close"] = (h >= FOOD_CLOSE_HOUR).astype(int)

    # Hours elapsed since food service closed (0 during food hours)
    df["hours_since_food_close"] = np.where(
        h >= FOOD_CLOSE_HOUR, h - FOOD_CLOSE_HOUR, 0
    )

    # Live music window — keep the flag from raw data if present, else derive
    if "live_music_flag" in df.columns:
        df["is_live_music_window"] = df["live_music_flag"].astype(int)
    else:
        wd = df["timestamp_hour"].dt.weekday
        is_fri_sat = wd.isin([4, 5])
        music = (
            (is_fri_sat & (h >= MUSIC_WEEKEND_START)) |
            (~is_fri_sat & (h >= MUSIC_WEEKDAY_START))
        )
        df["is_live_music_window"] = music.astype(int)

    return df


def add_external_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pass-through for weather, airport, cruise, footfall, and events columns."""
    external_cols = [
        # Weather
        "temp_c", "rain_mm", "wind_speed_kmh", "weather_severity_flag",
        # Airport + cruise
        "airport_arrivals", "airport_arrivals_lag1",
        "cruise_ship_flag", "ships_in_port_count", "cruise_passenger_estimate",
        # Computed calendar signals
        "school_holiday_flag", "payday_period_flag", "days_from_payday",
        "college_term_flag", "bloomsday_flag", "bloomsday_week_flag",
        "summer_tourism_flag", "christmas_market_flag",
        "new_years_eve_flag", "new_years_day_flag",
        # Events
        "major_sports_event_flag", "city_event_flag", "st_patricks_week_flag",
        "aviva_event_flag", "croke_park_event_flag", "nearby_venue_event_flag",
        "event_impact_score",
        "failte_event_count", "failte_free_event_count", "failte_festival_count",
        "special_event_flag",
        # City pedestrian footfall (hourly — join on timestamp_hour)
        "suffolk_nassau_footfall", "nearby_footfall_avg", "nearby_footfall_sum",
        "city_footfall_zscore",
        "suffolk_footfall_lag_1h", "suffolk_footfall_lag_24h", "suffolk_footfall_lag_168h",
        "suffolk_footfall_roll_24h", "suffolk_is_busy_flag",
    ]
    for col in external_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Normalised airport signal
    if "airport_arrivals" in df.columns and df["airport_arrivals"].std() > 0:
        mu = df["airport_arrivals"].mean()
        sigma = df["airport_arrivals"].std()
        df["airport_arrivals_zscore"] = (df["airport_arrivals"] - mu) / sigma

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df["weekend_x_music"] = df["is_weekend"] * df.get("is_live_music_window", 0)
    df["rain_x_weekend"] = df["rain_mm"] * df["is_weekend"] if "rain_mm" in df.columns else 0

    event_cols = [
        "bank_holiday_flag", "major_sports_event_flag",
        "city_event_flag", "st_patricks_week_flag", "special_event_flag",
    ]
    present = [c for c in event_cols if c in df.columns]
    df["event_intensity"] = df[present].sum(axis=1) if present else 0

    # Tourism pressure: combine airport and cruise signals
    if "airport_arrivals_zscore" in df.columns and "cruise_ship_flag" in df.columns:
        df["tourism_pressure"] = (
            df["airport_arrivals_zscore"] + df["cruise_ship_flag"].astype(float)
        )

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_features(
    raw: pd.DataFrame,
    targets: list[str] | None = None,
    drop_na: bool = True,
    footfall_path: Path | str | None = "data/raw/footfall_hourly.csv",
    enrichment_path: Path | str | None = "data/raw/enrichment.csv",
    failte_path: Path | str | None = "data/raw/failte_events.csv",
    weather_path: Path | str | None = "data/raw/weather_hourly.csv",
) -> pd.DataFrame:
    """
    Transform a raw hourly dataframe into a model-ready supervised table.

    Parameters
    ----------
    raw             : raw hourly dataframe (from synthetic.py or real POS export)
    targets         : columns to generate lag/rolling features for
    drop_na         : drop rows where any lag/rolling feature is NaN
    footfall_path   : path to hourly footfall CSV (from fetch_footfall.py). None to skip.
    enrichment_path : path to daily enrichment CSV (from fetch_public_data.py). None to skip.
    failte_path     : path to Fáilte Ireland events CSV. None to skip.

    Returns
    -------
    pd.DataFrame with all features + targets, sorted by timestamp_hour
    """
    if targets is None:
        targets = ["orders_count", "food_tickets_count"]

    df = raw.copy()
    df["timestamp_hour"] = pd.to_datetime(df["timestamp_hour"])
    df = df.sort_values("timestamp_hour").reset_index(drop=True)

    years = df["timestamp_hour"].dt.year.unique().tolist()
    holiday_set = _build_holiday_set(years)

    def _merge_daily(df: pd.DataFrame, enrich: pd.DataFrame, label: str) -> pd.DataFrame:
        """
        Merge a daily enrichment table into an hourly df on date.
        Real enrichment values REPLACE synthetic columns of the same name.
        New columns are added. No _x/_y suffixes.
        """
        enrich = enrich.copy()
        enrich["_date"] = pd.to_datetime(enrich.get("date", enrich.iloc[:, 0])).dt.date
        enrich = enrich.drop(columns=["date"], errors="ignore")
        df["_date"] = df["timestamp_hour"].dt.date

        # Drop columns from df that the enrichment will provide (prefer real data)
        overlap = [c for c in enrich.columns if c in df.columns and c != "_date"]
        if overlap:
            df = df.drop(columns=overlap)

        new_cols_before = len(df.columns)
        df = df.merge(enrich, on="_date", how="left")
        df = df.drop(columns=["_date"], errors="ignore")
        added = len(df.columns) - new_cols_before
        print(f"  Joined {added} columns from {label}")
        return df

    # --- Join daily enrichment (weather, events, computed calendar signals) ---
    if enrichment_path and Path(enrichment_path).exists():
        enrich = pd.read_csv(enrichment_path, low_memory=False)
        df = _merge_daily(df, enrich, str(enrichment_path))

    # --- Join Fáilte events (upcoming event counts near pub) ---
    if failte_path and Path(failte_path).exists():
        failte = pd.read_csv(failte_path, low_memory=False)
        df = _merge_daily(df, failte, str(failte_path))
        for col in ["failte_event_count", "failte_free_event_count", "failte_festival_count"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # --- Join hourly footfall (city pedestrian counts — aligns on timestamp_hour) ---
    if footfall_path and Path(footfall_path).exists():
        footfall = pd.read_csv(footfall_path, low_memory=False)
        footfall["timestamp_hour"] = pd.to_datetime(footfall["timestamp_hour"])
        before_cols = set(df.columns)
        df = df.merge(footfall, on="timestamp_hour", how="left")
        new_cols = set(df.columns) - before_cols
        print(f"  Joined {len(new_cols)} footfall columns from {footfall_path}")
        # Fill NaN footfall with 0 (missing counter data treated as no signal)
        for col in new_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- Join hourly weather ---
    if weather_path and Path(weather_path).exists():
        weather = pd.read_csv(weather_path, low_memory=False)
        weather["timestamp_hour"] = pd.to_datetime(weather["timestamp_hour"])
        
        overlap = [c for c in weather.columns if c in df.columns and c != "timestamp_hour"]
        if overlap:
            df = df.drop(columns=overlap)
            
        before_cols = set(df.columns)
        df = df.merge(weather, on="timestamp_hour", how="left")
        new_cols = set(df.columns) - before_cols
        print(f"  Joined {len(new_cols)} weather columns from {weather_path}")

    for target in targets:
        if target not in df.columns:
            continue
        df = add_lag_features(df, target)
        df = add_rolling_features(df, target)

    df = add_calendar_features(df, holiday_set)
    df = add_venue_features(df)
    df = add_external_features(df)
    df = add_interaction_features(df)

    if drop_na:
        lag_cols = [c for c in df.columns if "_lag_" in c or "_roll_" in c]
        before = len(df)
        df = df.dropna(subset=lag_cols).reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            print(f"Dropped {dropped} warm-up rows (lag/rolling NaN).")

    return df


def get_feature_columns(df: pd.DataFrame, targets: list[str]) -> list[str]:
    """Return all feature column names (everything except raw targets and timestamp)."""
    exclude = {"timestamp_hour"} | set(targets) | {
        "sales_total", "covers_count", "busy_label",
        "bar_staff_count", "kitchen_staff_count",
        "stockout_flag", "menu_change_flag", "promo_flag",
        "live_music_flag",   # replaced by is_live_music_window
        "hour", "weekday", "month", "is_weekend",  # kept in df but derived above too
    }
    return [c for c in df.columns if c not in exclude and c not in targets]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    raw_path = Path("data/synthetic/odonoghues_hourly.csv")
    out_path = Path("data/processed/features.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {raw_path} ...")
    raw = pd.read_csv(raw_path)
    print(f"  Raw shape: {raw.shape}")

    df = build_features(raw)
    df.to_parquet(out_path, index=False)

    targets = ["orders_count", "food_tickets_count"]
    feature_cols = get_feature_columns(df, targets)

    print(f"  Feature table shape: {df.shape}")
    print(f"  Feature count: {len(feature_cols)}")
    print(f"\n--- Feature groups ---")

    groups = {
        "Lag (row-based)":   [c for c in feature_cols if "_lag_1" in c or "_lag_2" in c or "_lag_3" in c],
        "Lag (calendar)":    [c for c in feature_cols if "lag_24h" in c or "lag_48h" in c or "lag_168h" in c or "lag_336h" in c],
        "Rolling":           [c for c in feature_cols if "_roll_" in c],
        "Calendar":          [c for c in feature_cols if any(x in c for x in ["hour", "weekday", "month", "weekend", "quarter", "week_of", "bank_", "school_", "payday_", "sin", "cos"])],
        "Venue":             [c for c in feature_cols if any(x in c for x in ["food_service", "breakfast", "lunch_window", "dinner_window", "music", "after_food", "hours_since"])],
        "External":          [c for c in feature_cols if any(x in c for x in ["temp_c", "rain_mm", "wind", "weather_", "airport", "cruise", "ships_"])],
        "Events":            [c for c in feature_cols if any(x in c for x in ["sports", "city_event", "patricks", "special_event"])],
        "Interactions":      [c for c in feature_cols if any(x in c for x in ["weekend_x", "rain_x", "event_intensity", "tourism_pressure"])],
    }

    for group, cols in groups.items():
        print(f"  {group} ({len(cols)}): {', '.join(cols[:6])}{'...' if len(cols) > 6 else ''}")

    print(f"\n--- Sample rows ---")
    sample_cols = ["timestamp_hour"] + targets + ["orders_count_lag_1", "orders_count_lag_168h",
                   "orders_count_roll_mean_24", "is_lunch_window", "rain_mm", "event_intensity"]
    present = [c for c in sample_cols if c in df.columns]
    print(df[present].head(8).to_string(index=False))
