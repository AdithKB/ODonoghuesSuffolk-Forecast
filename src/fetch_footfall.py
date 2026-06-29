"""
Dublin city-centre pedestrian footfall + Fáilte Ireland events for O'Donoghues.

Two data sources:
  1. DCC Footfall Counters — Smart Dublin hourly pedestrian counts (CC-BY)
       - Counter "Grafton Street / Nassau Street / Suffolk Street" is 50m from O'Donoghues
       - Also aggregates 5 nearest counters for a broader city pressure signal
       - Source: https://data.smartdublin.ie/dataset/dublin-city-centre-footfall-counters

  2. Fáilte Ireland Events API — daily tourism event counts near Dublin city centre
       - Free, no API key, updated daily
       - Source: https://failteireland.azure-api.net/opendata-api/v2/events/csv

Run directly:
    python src/fetch_footfall.py

Outputs:
    data/raw/footfall_hourly.csv    — hourly footfall features (join to hourly model data)
    data/raw/failte_events.csv      — daily event counts near Dublin city centre
"""

import io
import math
import logging
from datetime import date, datetime
from pathlib import Path

import requests
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

RAW_DIR   = Path("data/raw")
CACHE_DIR = Path("data/raw/footfall_cache")
RAW_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

REQUESTS_TIMEOUT = 60
REQUESTS_HEADERS = {"User-Agent": "ODonoghuesForecast/1.0 (research project)"}

# ---------------------------------------------------------------------------
# Counter selection — 6 locations nearest to O'Donoghues Suffolk Street D02
# ---------------------------------------------------------------------------

# The column in the XLSX/CSV exactly as it appears (with spaces/slashes)
SUFFOLK_COL = "Grafton Street / Nassau Street / Suffolk Street"

NEARBY_COUNTERS: dict[str, str] = {
    "suffolk_nassau":    "Grafton Street / Nassau Street / Suffolk Street",
    "grafton_compub":    "Grafton Street/CompuB",
    "grafton_monsoon":   "Grafton st/Monsoon",
    "dawson_st":         "Dawson Street",
    "dawson_moles":      "Dawson Street/Molesworth Pedestrian",
    "dame_londis":       "Dame Street/Londis",
}

# ---------------------------------------------------------------------------
# Annual file index — direct download URLs (CC-BY licence)
# Ordered newest-first so the partial 2026 file is preferred last.
# ---------------------------------------------------------------------------
ANNUAL_FILES = [
    {
        "year": 2022,
        "url":  "https://data.smartdublin.ie/dataset/cc421859-1f4f-43f6-b349-f4ca0e1c60fa"
                "/resource/2beeedcc-7fe6-4ae2-b8c7-ee8179686595/download/pedestrian-counts-2022.csv",
        "fmt":  "csv",
    },
    {
        "year": 2023,
        "url":  "https://data.smartdublin.ie/dataset/cc421859-1f4f-43f6-b349-f4ca0e1c60fa"
                "/resource/0d0f0de2-d82d-404e-8da2-d238de985532"
                "/download/pedestrian-counts-1-jan-31-december-2023.csv",
        "fmt":  "csv",
    },
    {
        "year": 2024,
        "url":  "https://data.smartdublin.ie/dataset/cc421859-1f4f-43f6-b349-f4ca0e1c60fa"
                "/resource/49a7c9a7-715e-4284-b1f4-05f03cd07ddb"
                "/download/pedestrian-counts-1-jan-31-december-2024.csv",
        "fmt":  "csv",
    },
    {
        "year": 2025,
        "url":  "https://data.smartdublin.ie/dataset/cc421859-1f4f-43f6-b349-f4ca0e1c60fa"
                "/resource/ca145381-cedf-475a-9b96-6e43b76c8a98"
                "/download/pedestrian-counts-1-jan-31-december-2025.csv",
        "fmt":  "csv",
    },
    {
        "year": 2026,
        "url":  "https://data.smartdublin.ie/dataset/cc421859-1f4f-43f6-b349-f4ca0e1c60fa"
                "/resource/6559b005-38c2-4648-a926-f1a3c70a99a2"
                "/download/pedestrain-counts-1-jan-2-jun-2026.xlsx",
        "fmt":  "xlsx",
    },
]

CURRENT_YEAR = date.today().year


# ---------------------------------------------------------------------------
# 1. Download + cache one year of footfall data
# ---------------------------------------------------------------------------

def _download_year(year_info: dict, force: bool = False) -> pd.DataFrame | None:
    """
    Download a single year's footfall CSV/XLSX and return as a DataFrame
    with only the nearby counter columns + the Time column.

    Files are cached locally; pass force=True to re-download.
    """
    year = year_info["year"]
    url  = year_info["url"]
    fmt  = year_info["fmt"]

    cache_path = CACHE_DIR / f"footfall_{year}.parquet"

    if cache_path.exists() and not force and year < CURRENT_YEAR:
        log.info(f"  {year}: loading from cache ({cache_path})")
        return pd.read_parquet(cache_path)

    log.info(f"  {year}: downloading from Smart Dublin… ({url[-60:]})")
    try:
        resp = requests.get(url, headers=REQUESTS_HEADERS, timeout=REQUESTS_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"  {year}: download failed — {e}")
        return None

    try:
        if fmt == "xlsx" or resp.headers.get("content-type", "").startswith(
            "application/vnd.openxmlformats"
        ):
            raw = pd.read_excel(io.BytesIO(resp.content), sheet_name=0, header=0)
        else:
            # utf-8-sig strips BOM on some files; fall back to latin-1 if needed
            try:
                raw = pd.read_csv(io.BytesIO(resp.content), encoding="utf-8-sig", low_memory=False)
            except Exception:
                raw = pd.read_csv(io.BytesIO(resp.content), encoding="latin-1", low_memory=False)
    except Exception as e:
        log.warning(f"  {year}: parse failed — {e}")
        return None

    # Strip BOM and whitespace from all column names
    raw.columns = [c.strip("﻿￾").strip() for c in raw.columns]

    # Normalise the Time column
    if "Time" not in raw.columns:
        log.warning(f"  {year}: no 'Time' column. Columns: {list(raw.columns[:5])}")
        return None

    raw["timestamp_hour"] = pd.to_datetime(
        raw["Time"], dayfirst=True, errors="coerce"
    ).dt.floor("h")
    raw = raw.dropna(subset=["timestamp_hour"])

    # Extract only relevant counter cols
    selected = {"timestamp_hour": raw["timestamp_hour"]}
    for alias, col_name in NEARBY_COUNTERS.items():
        if col_name in raw.columns:
            selected[alias] = pd.to_numeric(raw[col_name], errors="coerce")
        else:
            selected[alias] = np.nan

    df = pd.DataFrame(selected).sort_values("timestamp_hour").reset_index(drop=True)

    # Cache completed full years
    if year < CURRENT_YEAR and not df.empty:
        df.to_parquet(cache_path, index=False)
        log.info(f"  {year}: cached {len(df)} hourly rows → {cache_path}")

    return df


# ---------------------------------------------------------------------------
# 2. Fetch all years and build hourly footfall features
# ---------------------------------------------------------------------------

def fetch_footfall(
    start: str | None = None,
    end: str | None = None,
    force_refresh_current_year: bool = False,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    Download all available years of Dublin footfall data and return hourly
    features for the date range [start, end].

    Feature columns produced:
        timestamp_hour
        suffolk_nassau_footfall      — total peds at Grafton/Nassau/Suffolk St counter
        nearby_footfall_avg          — mean of up to 6 nearest counters
        nearby_footfall_sum          — sum of up to 6 nearest counters
        city_footfall_zscore         — z-score of suffolk_nassau_footfall
        suffolk_footfall_lag_1h      — previous hour
        suffolk_footfall_lag_24h     — same hour yesterday
        suffolk_footfall_lag_168h    — same hour last week
        suffolk_footfall_roll_24h    — rolling 24-hour mean (leakage-safe when used with lag)
        suffolk_is_busy_flag         — 1 if footfall > 70th percentile for that hour-of-day

    Parameters
    ----------
    start, end : optional date filter ("YYYY-MM-DD"). If not given, returns all available data.
    force_refresh_current_year : re-download the current year's file even if cached.
    """
    dfs = []
    for info in ANNUAL_FILES:
        force = force_refresh_current_year and info["year"] == CURRENT_YEAR
        df = _download_year(info, force=force)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        log.warning("No footfall data downloaded. Returning empty DataFrame.")
        return pd.DataFrame(columns=["timestamp_hour"])

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates("timestamp_hour").sort_values("timestamp_hour").reset_index(drop=True)

    # Filter date range
    if start:
        combined = combined[combined["timestamp_hour"] >= pd.Timestamp(start)]
    if end:
        combined = combined[combined["timestamp_hour"] <= pd.Timestamp(end) + pd.Timedelta(hours=23)]

    # Nearby aggregate (most reliable — tolerant of individual counter outages)
    nearby_cols = list(NEARBY_COUNTERS.keys())
    avail_cols  = [c for c in nearby_cols if c in combined.columns]
    combined["nearby_footfall_avg"] = combined[avail_cols].mean(axis=1).round(1)
    combined["nearby_footfall_sum"] = combined[avail_cols].sum(axis=1).astype(int)

    # Suffolk St counter — keep NaN where counter is offline, then impute from nearby
    combined["suffolk_nassau_footfall"] = combined["suffolk_nassau"].astype(float)
    suffolk_avail = combined["suffolk_nassau_footfall"].notna()
    combined["suffolk_counter_is_live"] = suffolk_avail.astype(int)
    if suffolk_avail.sum() > 100:
        # Scale factor = median of suffolk/nearby ratio during live periods
        mask = suffolk_avail & (combined["nearby_footfall_avg"] > 0)
        scale = (
            combined.loc[mask, "suffolk_nassau_footfall"]
            / combined.loc[mask, "nearby_footfall_avg"]
        ).median()
        combined["suffolk_nassau_footfall"] = combined["suffolk_nassau_footfall"].fillna(
            (combined["nearby_footfall_avg"] * scale).round(0)
        )
    else:
        combined["suffolk_nassau_footfall"] = combined["nearby_footfall_avg"]
    combined["suffolk_nassau_footfall"] = combined["suffolk_nassau_footfall"].fillna(0).astype(int)

    # Primary lag/roll signal: nearby_footfall_avg (consistent across all years)
    s = combined["nearby_footfall_avg"].fillna(0)

    # Z-score
    mu, sigma = s.mean(), s.std()
    combined["city_footfall_zscore"] = ((s - mu) / sigma).round(3)

    # Lagged features — leakage-safe (shifted before join to model data)
    combined["suffolk_footfall_lag_1h"]   = s.shift(1).fillna(0).round(0).astype(int)
    combined["suffolk_footfall_lag_24h"]  = s.shift(24).fillna(0).round(0).astype(int)
    combined["suffolk_footfall_lag_168h"] = s.shift(168).fillna(0).round(0).astype(int)

    # Rolling 24h mean (leakage-safe)
    combined["suffolk_footfall_roll_24h"] = s.shift(1).rolling(24, min_periods=4).mean().round(1)

    # Busy flag: above 70th percentile for that hour-of-week
    hour_of_week = combined["timestamp_hour"].dt.dayofweek * 24 + combined["timestamp_hour"].dt.hour
    p70 = combined.groupby(hour_of_week)["nearby_footfall_avg"].transform(
        lambda x: x.quantile(0.70)
    )
    combined["suffolk_is_busy_flag"] = (combined["nearby_footfall_avg"] > p70).astype(int)

    # Keep only engineered feature columns
    keep = [
        "timestamp_hour",
        "suffolk_nassau_footfall",
        "suffolk_counter_is_live",
        "nearby_footfall_avg",
        "nearby_footfall_sum",
        "city_footfall_zscore",
        "suffolk_footfall_lag_1h",
        "suffolk_footfall_lag_24h",
        "suffolk_footfall_lag_168h",
        "suffolk_footfall_roll_24h",
        "suffolk_is_busy_flag",
    ]
    result = combined[[c for c in keep if c in combined.columns]]

    if save_path:
        result.to_csv(save_path, index=False)
        log.info(f"Saved {len(result)} hourly footfall rows → {save_path}")
        log.info(f"  Date range: {result['timestamp_hour'].min()} → {result['timestamp_hour'].max()}")

    return result


# ---------------------------------------------------------------------------
# 3. Fáilte Ireland Events API — daily event counts near Dublin city centre
# ---------------------------------------------------------------------------

FAILTE_EVENTS_URL = "https://failteireland.azure-api.net/opendata-api/v2/events/csv"

# O'Donoghues location — events within FAILTE_RADIUS_KM are flagged as nearby
ODONOGHUES_LAT = 53.3434
ODONOGHUES_LON = -6.2601
FAILTE_RADIUS_KM = 5.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fetch_failte_events(
    start: str | None = None,
    end: str | None = None,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    Fetch the Fáilte Ireland events feed and produce a daily count of events
    happening within FAILTE_RADIUS_KM of O'Donoghues (default 5km).

    Daily columns:
        date,
        failte_event_count       — total events within radius on this date
        failte_free_event_count  — free events within radius
        failte_festival_count    — events tagged as Festival within radius
    """
    log.info(f"Downloading Fáilte Ireland events CSV…")
    try:
        resp = requests.get(
            FAILTE_EVENTS_URL,
            headers=REQUESTS_HEADERS,
            timeout=REQUESTS_TIMEOUT,
        )
        resp.raise_for_status()
        raw = pd.read_csv(io.StringIO(resp.text), low_memory=False, on_bad_lines="skip")
    except Exception as e:
        log.warning(f"Fáilte Ireland events fetch failed: {e}")
        return pd.DataFrame(columns=["date", "failte_event_count", "failte_free_event_count", "failte_festival_count"])

    log.info(f"  Downloaded {len(raw)} events, {len(raw.columns)} columns")

    # Normalise column names
    raw.columns = [c.strip() for c in raw.columns]

    # Parse dates (DD/MM/YYYY format)
    for col in ["Start Date", "End Date"]:
        if col in raw.columns:
            raw[col] = pd.to_datetime(raw[col], dayfirst=True, errors="coerce")

    # Parse coordinates
    raw["Latitude"]  = pd.to_numeric(raw.get("Latitude",  pd.Series(dtype=float)), errors="coerce")
    raw["Longitude"] = pd.to_numeric(raw.get("Longitude", pd.Series(dtype=float)), errors="coerce")

    # Filter to events with valid dates and coordinates
    raw = raw.dropna(subset=["Start Date", "Latitude", "Longitude"])

    # Distance filter: events within FAILTE_RADIUS_KM of pub
    raw["dist_km"] = raw.apply(
        lambda r: _haversine_km(ODONOGHUES_LAT, ODONOGHUES_LON, r["Latitude"], r["Longitude"]),
        axis=1,
    )
    nearby = raw[raw["dist_km"] <= FAILTE_RADIUS_KM].copy()
    log.info(f"  Events within {FAILTE_RADIUS_KM}km of O'Donoghues: {len(nearby)}")

    if nearby.empty:
        return pd.DataFrame(columns=["date", "failte_event_count", "failte_free_event_count", "failte_festival_count"])

    # If no End Date, event is single-day
    nearby["End Date"] = nearby["End Date"].fillna(nearby["Start Date"])

    # Expand each event to cover its date range
    records = []
    for _, ev in nearby.iterrows():
        s = ev["Start Date"]
        e = min(ev["End Date"], pd.Timestamp("2030-12-31"))
        if pd.isna(s):
            continue
        is_free     = str(ev.get("Is Free To Visit", "")).strip().lower() in ("yes", "true", "1")
        event_type  = str(ev.get("Event Type", "")).lower()
        is_festival = "festival" in event_type or "festival" in str(ev.get("Name", "")).lower()
        for day in pd.date_range(s, e, freq="D"):
            records.append({
                "date":        day.date(),
                "is_free":     is_free,
                "is_festival": is_festival,
            })

    if not records:
        return pd.DataFrame(columns=["date", "failte_event_count", "failte_free_event_count", "failte_festival_count"])

    ev_df = pd.DataFrame(records)
    daily = ev_df.groupby("date").agg(
        failte_event_count      =("date",        "count"),
        failte_free_event_count =("is_free",     "sum"),
        failte_festival_count   =("is_festival", "sum"),
    ).reset_index()
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["failte_free_event_count"]  = daily["failte_free_event_count"].astype(int)
    daily["failte_festival_count"]    = daily["failte_festival_count"].astype(int)

    # Filter date range
    if start:
        daily = daily[daily["date"] >= pd.Timestamp(start).date()]
    if end:
        daily = daily[daily["date"] <= pd.Timestamp(end).date()]

    if save_path:
        daily.to_csv(save_path, index=False)
        log.info(f"Saved {len(daily)} daily event rows → {save_path}")

    return daily


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    start = sys.argv[1] if len(sys.argv) > 1 else "2023-01-01"
    end   = sys.argv[2] if len(sys.argv) > 2 else str(date.today())

    print(f"\n=== Dublin Footfall Counters ({start} → {end}) ===\n")
    footfall = fetch_footfall(
        start=start,
        end=end,
        force_refresh_current_year=True,
        save_path=RAW_DIR / "footfall_hourly.csv",
    )
    print(f"Rows: {len(footfall)}")
    print(footfall.tail(12).to_string(index=False))

    print(f"\n\n=== Fáilte Ireland Events ===\n")
    events = fetch_failte_events(
        start=start,
        end=end,
        save_path=RAW_DIR / "failte_events.csv",
    )
    print(f"Days with events near pub: {len(events)}")
    if not events.empty:
        print(events.head(20).to_string(index=False))
