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

    Also adds:
    - same_slot_4w_avg: mean of the same (hour, weekday) slot at 7, 14, 21, 28 days
      prior. Handles NaN by averaging only available weeks.
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

    # Same-slot 4-week average: mean of weeks 1–4 lookback (NaN-safe)
    col_name = f"{target}_same_slot_4w_avg"
    if col_name not in df.columns:
        week_lags = []
        for weeks in [1, 2, 3, 4]:
            lv = _lag_by_timedelta(
                ts_series,
                pd.Timedelta(hours=weeks * 168),
                f"_tmp_{target}_{weeks}w",
            )
            week_lags.append(lv.values.astype(float))
        stacked = np.vstack(week_lags).T  # (n_rows, 4)
        df[col_name] = np.nanmean(stacked, axis=1)

    # Same-weekday same-hour rolling average over last 8 occurrences.
    # Conditions on (weekday, hour) so Thursday 1pm only averages previous
    # Thursdays at 1pm — much tighter baseline than same_slot_4w_avg.
    wd_col = f"{target}_same_wd_hour_roll8"
    if wd_col not in df.columns:
        _wd = df["timestamp_hour"].dt.weekday
        _h  = df["timestamp_hour"].dt.hour
        df[wd_col] = (
            df.groupby([_wd, _h])[target]
            .transform(lambda x: x.shift(1).rolling(8, min_periods=2).mean())
        )

    return df


def add_ewma_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Exponential Weighted Moving Averages — discount older obs naturally.
    Computed on shift(1) series so window ends at T-1 (no leakage).
    Better than simple rolling means for capturing momentum in demand.
    """
    s = df[target]
    for span, label in [(6, "ewma_6"), (24, "ewma_24"), (168, "ewma_168")]:
        df[f"{target}_{label}"] = s.shift(1).ewm(span=span, min_periods=1).mean()
    return df


def add_trend_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Long-horizon trend features computed on daily totals then broadcast to hours.
    Daily aggregation avoids dead-hour (2–8am near-zero) dilution.

    - trend_90d : 90-day rolling mean of daily totals (captures seasonal arc)
    - yoy_ratio : 28-day rolling avg / same window 364 days ago (growth signal)
    """
    ts = pd.to_datetime(df["timestamp_hour"])
    date_idx = pd.to_datetime(ts.dt.date)

    daily = (
        pd.Series(df[target].values, index=date_idx)
        .groupby(level=0).sum()
        .sort_index()
    )

    # 90-day rolling mean on daily totals, shift by 1 day, min 14 days of data
    roll90 = daily.shift(1).rolling(90, min_periods=14).mean().fillna(0.0)

    # YoY: 28-day avg divided by same 28-day window 364 days prior
    roll28      = daily.shift(1).rolling(28, min_periods=7).mean()
    roll28_1y   = roll28.shift(364)
    yoy         = (roll28 / roll28_1y.replace(0.0, np.nan)).clip(0.3, 3.0).fillna(1.0)

    roll90_map = roll90.to_dict()
    yoy_map    = yoy.to_dict()

    df[f"{target}_trend_90d"] = date_idx.map(roll90_map).fillna(0.0).astype(float)
    df[f"{target}_yoy_ratio"]  = date_idx.map(yoy_map).fillna(1.0).astype(float)

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
        df["is_live_music_window"] = df["live_music_flag"].fillna(0).astype(int)
    else:
        wd = df["timestamp_hour"].dt.weekday
        is_fri_sat = wd.isin([4, 5])
        music = (
            (is_fri_sat & (h >= MUSIC_WEEKEND_START)) |
            (~is_fri_sat & (h >= MUSIC_WEEKDAY_START))
        )
        df["is_live_music_window"] = music.astype(int)

    return df


def add_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic schedule-based demand signals derived purely from the timestamp.
    No external data needed — all flags are computable for any future date.
    """
    ts = df["timestamp_hour"]
    h = ts.dt.hour
    wd = ts.dt.weekday  # 0=Mon, 6=Sun
    m = ts.dt.month
    day = ts.dt.day
    is_weekday = (wd < 5).astype(int)

    # college_term_flag: prefer enrichment CSV value; derive from month if absent
    if "college_term_flag" not in df.columns:
        df["college_term_flag"] = (
            ((m >= 9) & (m <= 12)) | ((m >= 1) & (m <= 5))
        ).astype(int)

    # post_lecture_window: weekday, term time, 13–18
    college_term = df["college_term_flag"].fillna(0).astype(int)
    df["post_lecture_window"] = (
        is_weekday
        & college_term
        & h.between(13, 18)
    ).astype(int)

    # fresher_week_flag: first full week of October (Mon–Sun containing first Monday on/after Oct 1)
    def _fresher_week_dates(year):
        oct1 = pd.Timestamp(year, 10, 1)
        days_to_mon = (7 - oct1.weekday()) % 7
        first_mon = oct1 + pd.Timedelta(days=days_to_mon)
        return pd.date_range(first_mon, periods=7, freq="D")

    fresher_dates = set()
    for y in ts.dt.year.unique():
        for d in _fresher_week_dates(y):
            fresher_dates.add(d.date())
    df["fresher_week_flag"] = ts.dt.date.map(lambda d: d in fresher_dates).astype(int)

    # rag_week_flag: approx third week of February (Feb 14–22)
    df["rag_week_flag"] = ((m == 2) & day.between(14, 22)).astype(int)

    # exam_period_flag: May only (Trinity summer exams)
    # Nov–Dec was previously included but is confounded with Christmas market season
    df["exam_period_flag"] = (m == 5).astype(int)

    # is_business_lunch_hour: weekday 12–13, not bank holiday
    bank_hol = df.get("bank_holiday_flag", pd.Series(0, index=df.index))
    df["is_business_lunch_hour"] = (
        is_weekday & h.between(12, 13) & (bank_hol == 0)
    ).astype(int)

    # is_after_work_window: weekday 17–19, not bank holiday
    df["is_after_work_window"] = (
        is_weekday & h.between(17, 19) & (bank_hol == 0)
    ).astype(int)

    # budget_day_flag: first Tuesday of October each year (Irish national Budget Day)
    def _budget_day(year):
        oct1 = pd.Timestamp(year, 10, 1)
        days_to_tue = (1 - oct1.weekday()) % 7  # Tuesday = weekday 1
        return (oct1 + pd.Timedelta(days=days_to_tue)).date()

    budget_dates = {_budget_day(y) for y in ts.dt.year.unique()}
    df["budget_day_flag"] = ts.dt.date.map(lambda d: d in budget_dates).astype(int)

    # cheltenham_festival_flag: Tue–Fri of week containing the 2nd Wednesday of March
    def _cheltenham_dates(year):
        mar1 = pd.Timestamp(year, 3, 1)
        days_to_wed = (2 - mar1.weekday()) % 7
        first_wed = mar1 + pd.Timedelta(days=days_to_wed)
        second_wed = first_wed + pd.Timedelta(weeks=1)
        tue = second_wed - pd.Timedelta(days=1)
        return pd.date_range(tue, periods=4, freq="D")

    cheltenham_dates = set()
    for y in ts.dt.year.unique():
        for d in _cheltenham_dates(y):
            cheltenham_dates.add(d.date())
    df["cheltenham_festival_flag"] = ts.dt.date.map(lambda d: d in cheltenham_dates).astype(int)

    return df


def add_sports_features(
    df: pd.DataFrame,
    sports_path: "Path | str | None" = "data/raw/sports_fixtures.csv",
    six_nations_path: "Path | str | None" = "data/raw/six_nations_fixtures.csv",
    horse_racing_path: "Path | str | None" = "data/raw/horse_racing_fixtures.csv",
    ireland_rugby_path: "Path | str | None" = "data/raw/ireland_rugby_fixtures.csv",
    window_hours: int = 3,
) -> pd.DataFrame:
    """
    Compute TV sports feature columns for each hourly row.

    For each hourly timestamp, look for qualifying fixtures within a ±window_hours
    window around that hour and compute:
      tv_sports_flag            1 if any match in the ±3h window, else 0
      tv_sports_intensity       max intensity score of any match in the window (0 if none)
      match_kickoff_proximity   hours to nearest kickoff in the window (99 if none)
      is_ireland_match_window   1 if an Ireland match (football or rugby) is in the window
      tv_sports_match_label     "Competition: Home vs Away" of most intense match (else "")

    Intensity scale: 1 = regular league match, 2 = CL/big derby/Six Nations,
                     3 = WC/EC knockout / CL Final / Ireland international

    Sources loaded (each is optional — missing files are skipped):
      sports_path        — football fixtures from football-data.org
      six_nations_path   — static Six Nations 2023–2026
      horse_racing_path  — horse racing fixtures (static + HRI PDF)
      ireland_rugby_path — Ireland non-Six-Nations rugby (Autumn Nations + Summer Tour)
    """
    frames = []
    for p in [sports_path, six_nations_path, horse_racing_path, ireland_rugby_path]:
        if p and Path(p).exists():
            try:
                frames.append(pd.read_csv(p))
            except Exception as e:
                print(f"  Warning: could not load sports fixtures from {p}: {e}")

    if not frames:
        print("  No sports fixture files found — skipping tv_sports features (zeros).")
        df["tv_sports_flag"] = 0
        df["tv_sports_intensity"] = 0
        df["match_kickoff_proximity"] = 99
        df["is_ireland_match_window"] = 0
        df["tv_sports_match_label"] = ""
        df["days_until_next_match"] = 7
        df["is_match_tomorrow"] = 0
        df["is_match_in_2_days"] = 0
        return df

    fixtures = pd.concat(frames, ignore_index=True)
    # Normalise team name columns — empty away_team (horse racing) reads back as NaN
    fixtures["home_team"] = fixtures["home_team"].fillna("")
    fixtures["away_team"] = fixtures["away_team"].fillna("")

    # Build kickoff datetime column
    def _parse_ko(row):
        try:
            return pd.Timestamp(f"{row['date']} {row['kickoff_local']}")
        except Exception:
            return pd.NaT

    fixtures["kickoff_dt"] = fixtures.apply(_parse_ko, axis=1)
    fixtures = fixtures.dropna(subset=["kickoff_dt"]).copy()
    fixtures["intensity"] = pd.to_numeric(fixtures["intensity"], errors="coerce").fillna(0).astype(int)
    fixtures["is_ireland_match"] = fixtures["is_ireland_match"].astype(bool)

    # Numpy arrays for fast vectorised loop
    ko_arr   = fixtures["kickoff_dt"].values.astype("datetime64[s]")
    int_arr  = fixtures["intensity"].values
    ire_arr  = fixtures["is_ireland_match"].values
    home_arr = fixtures["home_team"].values
    away_arr = fixtures["away_team"].values
    comp_arr = fixtures["competition"].values

    timestamps = pd.to_datetime(df["timestamp_hour"]).values.astype("datetime64[s]")
    n = len(df)
    tv_flag      = np.zeros(n, dtype=np.int8)
    tv_intensity = np.zeros(n, dtype=np.int8)
    tv_proximity = np.full(n, 99.0)
    tv_ireland   = np.zeros(n, dtype=np.int8)
    tv_label     = [""] * n

    win_sec = window_hours * 3600

    for i, ts in enumerate(timestamps):
        delta_sec = (ko_arr - ts).astype(np.int64)  # seconds, signed
        in_win = np.abs(delta_sec) <= win_sec
        if not in_win.any():
            continue

        tv_flag[i] = 1
        tv_intensity[i] = int(int_arr[in_win].max())
        abs_delta_h = np.abs(delta_sec[in_win]) / 3600.0
        tv_proximity[i] = round(float(abs_delta_h.min()), 1)
        tv_ireland[i] = int(ire_arr[in_win].any())

        # Label: most intense match; tie-break: prefer Ireland
        idxs = np.where(in_win)[0]
        intensities_win = int_arr[idxs]
        max_int = intensities_win.max()
        top_idxs = idxs[intensities_win == max_int]
        ire_top = ire_arr[top_idxs]
        pick = top_idxs[ire_top][0] if ire_top.any() else top_idxs[0]
        tv_label[i] = f"{comp_arr[pick]}: {home_arr[pick]} vs {away_arr[pick]}"

    df["tv_sports_flag"]          = tv_flag.astype(int)
    df["tv_sports_intensity"]     = tv_intensity.astype(int)
    df["match_kickoff_proximity"] = tv_proximity
    df["is_ireland_match_window"] = tv_ireland.astype(int)
    df["tv_sports_match_label"]   = tv_label

    # Pre-event buildup: days until next high-intensity fixture
    hi_fixtures = fixtures[fixtures["intensity"] >= 2].copy()
    hi_ko_dates = pd.to_datetime(hi_fixtures["kickoff_dt"]).dt.normalize().drop_duplicates().sort_values().values

    days_until = np.full(n, 7, dtype=np.int8)  # default: 7 = "none in window"
    for i, ts in enumerate(timestamps):
        ts_date = pd.Timestamp(ts).normalize().to_datetime64()
        future = hi_ko_dates[hi_ko_dates >= ts_date]
        if len(future) > 0:
            diff = int((future[0] - ts_date) / np.timedelta64(1, 'D'))
            days_until[i] = min(diff, 7)

    df["days_until_next_match"] = days_until.astype(int)
    df["is_match_tomorrow"]     = (days_until == 1).astype(int)
    df["is_match_in_2_days"]    = (days_until == 2).astype(int)

    active = int(tv_flag.sum())
    print(f"  TV sports: {active} hourly slots have a match within ±{window_hours}h")
    return df


def add_external_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pass-through for weather, airport, cruise, footfall, and events columns."""
    external_cols = [
        # Weather
        "temp_c", "rain_mm", "wind_speed_kmh", "weather_severity_flag",
        "apparent_temp_c", "wind_gusts_kmh", "sunshine_duration_min",
        "uv_index", "precip_probability",
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
        # TV sports (from fetch_sports.py)
        "tv_sports_flag", "tv_sports_intensity", "match_kickoff_proximity",
        "is_ireland_match_window",
        # Pre-event buildup (days until next high-intensity fixture)
        "days_until_next_match", "is_match_tomorrow", "is_match_in_2_days",
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

    # ── Improved rain features ────────────────────────────────────────────
    # The raw hourly rain_mm captures rain during the visit hour itself —
    # a poor signal because people decide to go out based on earlier conditions.
    # We build three complementary features instead:
    #
    #  1. rainy_day_flag       — it rained meaningfully (>3mm) today at any point.
    #                            Captures "people reduce going-out plans on wet days."
    #  2. afternoon_rain_mm    — total rain 12:00–17:00, predicting evening decisions.
    #                            Captures "people saw it raining; opted for the pub."
    #  3. outdoor_weather_flag — clear, warm, sunny weekend afternoon: people may
    #                            go to parks/outdoor venues instead of the pub.
    #
    # Keep rain_x_weekend as a legacy interaction for backwards compatibility.
    if "rain_mm" in df.columns:
        ts = pd.to_datetime(df["timestamp_hour"])
        hour_col = ts.dt.hour
        date_col = ts.dt.date

        # Daily total rain — groupby date and broadcast back to hourly rows
        rain_ser = df["rain_mm"].fillna(0)
        daily_rain = (
            pd.Series(rain_ser.values, index=pd.to_datetime(date_col))
            .groupby(level=0).sum()
        )
        df["rainy_day_flag"] = pd.to_datetime(date_col).map(
            daily_rain.__getitem__
        ).ge(3.0).astype(int)

        # Afternoon rain (12:00–17:00): average over those hours per date
        aft_mask = hour_col.between(12, 17)
        aft_rain_by_date = (
            df.loc[aft_mask, ["timestamp_hour", "rain_mm"]]
            .assign(date=pd.to_datetime(df.loc[aft_mask, "timestamp_hour"]).dt.date)
            .groupby("date")["rain_mm"].mean()
        )
        # aft_rain_by_date index is datetime.date; map receives Timestamps — use .date()
        df["afternoon_rain_mm"] = pd.to_datetime(date_col).map(
            lambda d: aft_rain_by_date.get(d.date(), 0.0)
        ).fillna(0.0)

        # Legacy interaction (kept for model backward compat)
        df["rain_x_weekend"] = rain_ser * df["is_weekend"]
    else:
        df["rainy_day_flag"]     = 0
        df["afternoon_rain_mm"]  = 0.0
        df["rain_x_weekend"]     = 0

    # Outdoor weather flag: warm + sunny + weekend afternoon → people go outside
    if "sunshine_duration_min" in df.columns and "temp_c" in df.columns:
        ts = pd.to_datetime(df["timestamp_hour"])
        hour_col = ts.dt.hour
        df["sunny_afternoon"] = (
            (df["sunshine_duration_min"] > 20) & (df["temp_c"] > 16)
        ).astype(int)
        # Specific outdoor-pull signal: prime outside hours on good weather
        df["outdoor_weather_flag"] = (
            df["sunny_afternoon"].astype(bool)
            & df["is_weekend"].astype(bool)
            & hour_col.between(13, 19)
        ).astype(int)
    else:
        df["sunny_afternoon"]     = 0
        df["outdoor_weather_flag"] = 0

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

    # Late-night × Fri/Sat: Friday and Saturday nights are an entirely different
    # demand curve to weekday late nights — explicit interaction captures this.
    h = pd.to_datetime(df["timestamp_hour"]).dt.hour
    is_late_night = ((h >= 21) | (h <= 2)).astype(int)
    fri_sat = df.get("is_friday_saturday", pd.Series(0, index=df.index))
    df["late_night_x_fri_sat"] = is_late_night * fri_sat.astype(int)

    return df


# ---------------------------------------------------------------------------
# Busyness tier — weekday-adjusted, calibrated on 938 days of Titan POS data
# ---------------------------------------------------------------------------

# Thresholds: (quiet_max, normal_max, busy_max); above busy_max = Slammed
# Based on p33/p67/p90 of daily open-hour totals per weekday
_BUSYNESS_THRESHOLDS = {
    0: (452,  614,  800),   # Mon
    1: (450,  581,  722),   # Tue
    2: (445,  578,  741),   # Wed
    3: (658,  822, 1052),   # Thu
    4: (1060, 1248, 1446),  # Fri
    5: (1454, 1692, 1946),  # Sat
    6: (735,  948, 1324),   # Sun
}

def label_busyness(predicted_daily_total: float, weekday: int) -> str:
    """
    Return a busyness tier for a predicted daily total on a given weekday (0=Mon, 6=Sun).
    Tiers: Quiet / Normal / Busy / Slammed — calibrated to historical p33/p67/p90 per weekday.
    """
    q, n, b = _BUSYNESS_THRESHOLDS[weekday % 7]
    if predicted_daily_total < q:
        return "Quiet"
    elif predicted_daily_total < n:
        return "Normal"
    elif predicted_daily_total < b:
        return "Busy"
    else:
        return "Slammed"


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
    sports_path: Path | str | None = "data/raw/sports_fixtures.csv",
    six_nations_path: Path | str | None = "data/raw/six_nations_fixtures.csv",
    horse_racing_path: Path | str | None = "data/raw/horse_racing_fixtures.csv",
    ireland_rugby_path: Path | str | None = "data/raw/ireland_rugby_fixtures.csv",
) -> pd.DataFrame:
    """
    Transform a raw hourly dataframe into a model-ready supervised table.

    Parameters
    ----------
    raw               : raw hourly dataframe (from synthetic.py or real POS export)
    targets           : columns to generate lag/rolling features for
    drop_na           : drop rows where any lag/rolling feature is NaN
    footfall_path      : path to hourly footfall CSV (from fetch_footfall.py). None to skip.
    enrichment_path    : path to daily enrichment CSV (from fetch_public_data.py). None to skip.
    failte_path        : path to Fáilte Ireland events CSV. None to skip.
    sports_path        : path to football fixtures CSV (from fetch_sports.py). None to skip.
    six_nations_path   : path to Six Nations fixtures CSV. None to skip.
    horse_racing_path  : path to horse racing fixtures CSV. None to skip.
    ireland_rugby_path : path to Ireland rugby fixtures CSV (non-Six-Nations). None to skip.

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

    # --- Join TV sports fixtures (hourly window features) ---
    df = add_sports_features(
        df,
        sports_path=sports_path,
        six_nations_path=six_nations_path,
        horse_racing_path=horse_racing_path,
        ireland_rugby_path=ireland_rugby_path,
    )

    for target in targets:
        if target not in df.columns:
            continue
        df = add_lag_features(df, target)
        df = add_rolling_features(df, target)
        df = add_ewma_features(df, target)
        df = add_trend_features(df, target)

    df = add_calendar_features(df, holiday_set)
    df = add_venue_features(df)
    df = add_schedule_features(df)
    df = add_external_features(df)
    df = add_interaction_features(df)

    if drop_na:
        lag_cols = [c for c in df.columns if "_lag_" in c or "_roll_" in c]
        before = len(df)
        # Keep future forecast rows (NaN targets) regardless of lag NaNs —
        # they need to stay for dashboard predictions. Only drop historical
        # warm-up rows (which have real targets but missing lags).
        has_target = df[[t for t in (targets or []) if t in df.columns]].notna().any(axis=1)
        drop_mask = has_target & df[lag_cols].isna().any(axis=1)
        df = df[~drop_mask].reset_index(drop=True)
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
        "live_music_flag",         # replaced by is_live_music_window
        "hour", "weekday", "month", "is_weekend",  # kept in df but derived above too
        "tv_sports_match_label",   # string column — display only, not a model feature
    }
    return [c for c in df.columns if c not in exclude and c not in targets]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import numpy as np
    from datetime import timedelta as _timedelta
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.ingest_titan import build_titan_dataset

    titan_dir = Path("data/raw/pos_titanbi")
    out_path = Path("data/processed/features.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading Titan POS data from {titan_dir} ...")
    raw = build_titan_dataset(titan_dir, verbose=True)
    print(f"  Raw shape: {raw.shape}")

    last_date = pd.to_datetime(raw["timestamp_hour"]).max().date()
    open_hours = list(range(9, 24)) + [0, 1, 2]
    future_stubs = [
        {
            "timestamp_hour": pd.Timestamp(last_date + _timedelta(days=d)) + pd.Timedelta(hours=h),
            "orders_count": np.nan,
            "food_tickets_count": np.nan,
        }
        for d in range(1, 8)
        for h in open_hours
    ]
    raw = pd.concat([raw, pd.DataFrame(future_stubs)], ignore_index=True)
    print(f"  Extended with {len(future_stubs)} future stubs → {len(raw):,} total rows")

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
