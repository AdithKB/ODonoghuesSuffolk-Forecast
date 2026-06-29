"""
Public data fetchers for O'Donoghues demand prediction.

All sources are free, require no API keys, and can be called daily.

Sources:
  Weather   → Open-Meteo (free REST API, no key, historical + 7-day forecast)
  Airport   → Smart Dublin CSV (downloadable, CC-BY licence)
  Cruise    → Dublin Port HTML table (scraped)
  Holidays  → `holidays` Python library (always current)

Run directly:
    python src/fetch_public_data.py

Or call individual functions from features.py / a cron job.
"""

import io
import time
import json
import logging
from datetime import date, timedelta
from pathlib import Path

import requests
import pandas as pd
import holidays
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

DUBLIN_LAT  =  53.3498
DUBLIN_LON  =  -6.2603
DUBLIN_TZ   = "Europe/Dublin"

REQUESTS_TIMEOUT = 20
REQUESTS_HEADERS = {"User-Agent": "ODonoghuesForecast/1.0 (research project)"}


# ---------------------------------------------------------------------------
# 1. Weather — Open-Meteo (free, no key)
#    https://open-meteo.com/en/docs
# ---------------------------------------------------------------------------

def fetch_weather_historical(
    start: str,
    end: str,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    Fetch hourly weather from Open-Meteo historical archive.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   DUBLIN_LAT,
        "longitude":  DUBLIN_LON,
        "start_date": start,
        "end_date":   end,
        "hourly": ",".join([
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "weathercode",
        ]),
        "timezone": DUBLIN_TZ,
    }
    log.info(f"Fetching historical hourly weather {start} → {end} from Open-Meteo…")
    resp = requests.get(url, params=params, timeout=REQUESTS_TIMEOUT, headers=REQUESTS_HEADERS)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    df = pd.DataFrame({
        "timestamp_hour":    pd.to_datetime(data["time"]),
        "temp_c":            data["temperature_2m"],
        "rain_mm":           data["precipitation"],
        "wind_speed_kmh":    data["wind_speed_10m"],
        "weather_code":      data["weathercode"],
    })
    # WMO weather codes >= 61 = rain/storm/snow; >= 95 = thunderstorm
    df["weather_severity_flag"] = (df["weather_code"] >= 61).astype(int)
    df = df.drop(columns=["weather_code"])

    if save_path:
        df.to_csv(save_path, index=False)
        log.info(f"Saved {len(df)} rows → {save_path}")
    return df


def fetch_weather_forecast(days: int = 7, save_path: Path | None = None) -> pd.DataFrame:
    """
    Fetch next N days hourly weather forecast from Open-Meteo.
    Same schema as fetch_weather_historical so the two can be concatenated.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":   DUBLIN_LAT,
        "longitude":  DUBLIN_LON,
        "hourly": ",".join([
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "weathercode",
        ]),
        "timezone":     DUBLIN_TZ,
        "forecast_days": days,
    }
    log.info(f"Fetching {days}-day hourly weather forecast from Open-Meteo…")
    resp = requests.get(url, params=params, timeout=REQUESTS_TIMEOUT, headers=REQUESTS_HEADERS)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    df = pd.DataFrame({
        "timestamp_hour": pd.to_datetime(data["time"]),
        "temp_c":         data["temperature_2m"],
        "rain_mm":        data["precipitation"],
        "wind_speed_kmh": data["wind_speed_10m"],
        "weather_code":   data["weathercode"],
    })
    df["weather_severity_flag"] = (df["weather_code"] >= 61).astype(int)
    df = df.drop(columns=["weather_code"])

    if save_path:
        df.to_csv(save_path, index=False)
        log.info(f"Saved hourly forecast → {save_path}")
    return df

def build_hourly_weather_table(
    start: str,
    end: str | None = None,
    include_forecast_days: int = 7,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """
    Fetch hourly historical + forecast weather and merge into a single table.
    """
    if end is None:
        end = str(date.today())
        
    try:
        hist_wx = fetch_weather_historical(
            start, end,
            save_path=RAW_DIR / "weather_historical.csv",
        )
        fcast_wx = fetch_weather_forecast(
            include_forecast_days,
            save_path=RAW_DIR / "weather_forecast.csv",
        )
        weather = pd.concat([hist_wx, fcast_wx], ignore_index=True).drop_duplicates("timestamp_hour")
    except Exception as e:
        log.warning(f"Weather fetch failed: {e}. Using empty weather.")
        weather = pd.DataFrame(columns=["timestamp_hour","temp_c","rain_mm","wind_speed_kmh","weather_severity_flag"])

    # Forward-fill weather for forecast hours that might not have data yet
    wx_cols = ["temp_c","rain_mm","wind_speed_kmh"]
    for col in wx_cols:
        if col in weather.columns:
            weather[col] = weather[col].ffill()

    if out_path:
        weather.to_csv(out_path, index=False)
        log.info(f"\nHourly weather table saved → {out_path}")
        log.info(f"  Shape: {weather.shape}")

    return weather


# ---------------------------------------------------------------------------
# 2. Airport arrivals — Smart Dublin open dataset (CC-BY)
#    Updated periodically by Dublin City Council / Smart Dublin
# ---------------------------------------------------------------------------

SMART_DUBLIN_AIRPORT_CSV = (
    "https://data.smartdublin.ie/dataset/4997223b-13b2-4c97-9e88-cd94c6d35aec"
    "/resource/fc6e6f0f-b6a9-4ed6-b9c3-d2db1e872244"
    "/download/copy-of-indicator-9-dublin-airport.csv"
)


def fetch_airport_arrivals(save_path: Path | None = None) -> pd.DataFrame:
    """
    Download Smart Dublin Dublin Airport passenger arrivals dataset.

    The Smart Dublin CSV is quarterly data (e.g. 'Q3 25', total pax in thousands).
    This function broadcasts each quarter's total across its daily dates, giving
    a daily airport_arrivals proxy (quarterly total ÷ days in quarter).

    Returns daily DataFrame:
        date, airport_arrivals, airport_arrivals_lag1
    """
    log.info("Downloading Smart Dublin airport arrivals CSV…")
    resp = requests.get(
        SMART_DUBLIN_AIRPORT_CSV,
        timeout=REQUESTS_TIMEOUT,
        headers=REQUESTS_HEADERS,
    )
    resp.raise_for_status()

    raw = pd.read_csv(io.StringIO(resp.text))

    # Normalise column names
    raw.columns = [c.strip().replace("\n", " ").replace("  ", " ") for c in raw.columns]

    # The quarterly CSV has: Quarter | ... | Total (000) | ...
    # Quarter format: "Q3 25" = Q3 2025
    quarter_col = raw.columns[0]   # first col = quarter ID
    total_col   = next(
        (c for c in raw.columns if "total" in c.lower() and "000" in c),
        None,
    )
    if total_col is None:
        # fallback: second numeric-looking column
        total_col = raw.columns[2]

    log.info(f"Using quarter col='{quarter_col}', total col='{total_col}'")

    qdf = raw[[quarter_col, total_col]].copy()
    qdf.columns = ["quarter", "total_000"]
    qdf = qdf.dropna(subset=["quarter"])

    # Parse "Q3 25" → quarter start date
    def parse_quarter(q: str):
        q = str(q).strip()
        import re
        m = re.match(r"Q(\d)\s*(\d{2,4})", q)
        if not m:
            return None, None
        qnum, yr = int(m.group(1)), int(m.group(2))
        if yr < 100:
            yr += 2000
        q_start_month = {1: 1, 2: 4, 3: 7, 4: 10}[qnum]
        q_start = pd.Timestamp(yr, q_start_month, 1)
        q_end   = (q_start + pd.DateOffset(months=3) - pd.Timedelta(days=1))
        return q_start, q_end

    records = []
    for _, row in qdf.iterrows():
        q_start, q_end = parse_quarter(row["quarter"])
        if q_start is None:
            continue
        total_pax = pd.to_numeric(str(row["total_000"]).replace(",", ""), errors="coerce")
        if pd.isna(total_pax):
            continue
        total_pax = int(total_pax * 1000)   # convert from thousands
        dates_in_q = pd.date_range(q_start, q_end, freq="D")
        daily_pax = total_pax // len(dates_in_q)
        for d in dates_in_q:
            records.append({"date": d.date(), "airport_arrivals": daily_pax})

    if not records:
        log.warning("Could not parse quarterly airport data. Returning empty frame.")
        return pd.DataFrame(columns=["date", "airport_arrivals", "airport_arrivals_lag1"])

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    df["airport_arrivals_lag1"] = df["airport_arrivals"].shift(1).bfill().astype(int)

    if save_path:
        df.to_csv(save_path, index=False)
        log.info(f"Saved {len(df)} daily rows (from quarterly data) → {save_path}")
    return df


# ---------------------------------------------------------------------------
# 3. Cruise schedule — Dublin Port (scraped HTML table)
#    https://www.dublinport.ie/information-centre/next-100-arrivals/
# ---------------------------------------------------------------------------

DUBLIN_PORT_URL = "https://www.dublinport.ie/information-centre/next-100-arrivals/"

# Known large passenger vessels (used to identify cruise ships vs cargo)
CRUISE_KEYWORDS = [
    "CRUISE", "CELEBRITY", "ROYAL CARIBBEAN", "MSC ", "NORWEGIAN",
    "CARNIVAL", "PRINCESS", "CUNARD", "SILVERSEA", "VIKING",
    "MARELLA", "TUI ", "HURTIGRUTEN", "SEABOURN", "AZAMARA",
    "EVRIMA", "SCENIC", "PONANT", "WINDSTAR",
]


def fetch_cruise_schedule(
    save_path: Path | None = None,
    fallback_on_error: bool = True,
) -> pd.DataFrame:
    """
    Scrape Dublin Port next-100-arrivals page for cruise ship dates.

    Returns daily DataFrame:
        date, cruise_ship_flag, ships_in_port_count, cruise_passenger_estimate
    """
    log.info(f"Scraping Dublin Port arrivals: {DUBLIN_PORT_URL}")
    try:
        resp = requests.get(
            DUBLIN_PORT_URL,
            timeout=REQUESTS_TIMEOUT,
            headers=REQUESTS_HEADERS,
        )
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"Dublin Port scrape failed: {e}")
        if fallback_on_error:
            return pd.DataFrame(columns=["date","cruise_ship_flag","ships_in_port_count","cruise_passenger_estimate"])
        raise

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find the arrivals table — typically the first or only table on the page
    tables = soup.find_all("table")
    if not tables:
        log.warning("No tables found on Dublin Port page — page structure may have changed.")
        return pd.DataFrame(columns=["date","cruise_ship_flag","ships_in_port_count","cruise_passenger_estimate"])

    rows = []
    for table in tables:
        for tr in table.find_all("tr")[1:]:  # skip header
            cells = [td.get_text(strip=True) for td in tr.find_all(["td","th"])]
            if not cells:
                continue
            rows.append(cells)

    if not rows:
        log.warning("No table rows extracted from Dublin Port page.")
        return pd.DataFrame(columns=["date","cruise_ship_flag","ships_in_port_count","cruise_passenger_estimate"])

    # Identify date and vessel columns
    raw_df = pd.DataFrame(rows)
    log.info(f"Scraped {len(raw_df)} rows, {len(raw_df.columns)} columns from Dublin Port.")

    # Dublin Port table columns: Date | Time | Vessel | Vessel Type | Berth | From Location
    # Cruise ships have Vessel Type = "Cruise Liners"
    cruise_dates: dict[date, int] = {}

    for _, row in raw_df.iterrows():
        vals = list(row.values)
        if len(vals) < 4:
            continue

        # Column 3 = Vessel Type (0-indexed)
        vessel_type = str(vals[3]).strip().lower() if len(vals) > 3 else ""
        vessel_name = str(vals[2]).strip().upper() if len(vals) > 2 else ""
        is_cruise = (
            "cruise" in vessel_type
            or any(kw in vessel_name for kw in CRUISE_KEYWORDS)
        )

        # Column 0 = Date in DD/MM/YYYY format
        arrival_date = None
        date_str = str(vals[0]).strip()
        try:
            parsed = pd.to_datetime(date_str, dayfirst=True, errors="coerce")
            if pd.notna(parsed):
                arrival_date = parsed.date()
        except Exception:
            pass

        if arrival_date and is_cruise:
            cruise_dates[arrival_date] = cruise_dates.get(arrival_date, 0) + 1

    if not cruise_dates:
        log.info("No cruise ships identified. Returning empty schedule.")
        return pd.DataFrame(columns=["date","cruise_ship_flag","ships_in_port_count","cruise_passenger_estimate"])

    records = []
    for d, count in sorted(cruise_dates.items()):
        pax_estimate = count * 2500  # rough average passenger count per vessel
        records.append({
            "date":                     d,
            "cruise_ship_flag":         1,
            "ships_in_port_count":      count,
            "cruise_passenger_estimate": pax_estimate,
        })

    df = pd.DataFrame(records)
    if save_path:
        df.to_csv(save_path, index=False)
        log.info(f"Saved {len(df)} cruise days → {save_path}")
    return df


# ---------------------------------------------------------------------------
# 4. Irish bank holidays — `holidays` library (always current)
# ---------------------------------------------------------------------------

def generate_holidays(years: list[int], save_path: Path | None = None) -> pd.DataFrame:
    ie = holidays.Ireland(years=years)
    df = pd.DataFrame([
        {"date": d, "bank_holiday_flag": 1, "holiday_name": name}
        for d, name in sorted(ie.items())
    ])
    if save_path:
        df.to_csv(save_path, index=False)
        log.info(f"Saved {len(df)} Irish holidays → {save_path}")
    return df


# ---------------------------------------------------------------------------
# 5. Build enrichment table — merge all public signals into one daily CSV
# ---------------------------------------------------------------------------

def build_enrichment_table(
    start: str,
    end: str | None = None,
    include_forecast_days: int = 7,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """
    Fetch all public signals and merge into a single daily enrichment table.
    Covers `start` → `end` (historical) plus the next `include_forecast_days` days.

    This table is joined onto the hourly feature table in features.py by date.

    Parameters
    ----------
    start : "YYYY-MM-DD" — start of historical range
    end   : "YYYY-MM-DD" — end of historical range (usually today)
    include_forecast_days : how many future days of weather forecast to append

    Output columns:
        date, airport_arrivals, airport_arrivals_lag1,
        cruise_ship_flag, ships_in_port_count, cruise_passenger_estimate,
        bank_holiday_flag
    """
    if end is None:
        end = str(date.today())
        
    end_dt = pd.Timestamp(end).date()
    future_end = end_dt + timedelta(days=include_forecast_days)

    # Full date spine
    all_dates = pd.DataFrame({
        "date": pd.date_range(start, future_end, freq="D").date
    })

    # --- Airport ---
    try:
        airport = fetch_airport_arrivals(save_path=RAW_DIR / "airport_arrivals.csv")
    except Exception as e:
        log.warning(f"Airport fetch failed: {e}. Using empty airport data.")
        airport = pd.DataFrame(columns=["date","airport_arrivals","airport_arrivals_lag1"])

    # --- Cruise ---
    cruise = fetch_cruise_schedule(save_path=RAW_DIR / "cruise_schedule.csv")

    # --- Holidays ---
    years = list(range(pd.Timestamp(start).year, future_end.year + 1))
    hols = generate_holidays(years, save_path=RAW_DIR / "irish_holidays.csv")
    hols = hols[["date","bank_holiday_flag"]]

    # --- Events (static calendar + computed signals) ---
    try:
        try:
            from src.fetch_events import build_event_enrichment
        except ModuleNotFoundError:
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            from fetch_events import build_event_enrichment
        events = build_event_enrichment(
            start=start,
            end=str(future_end),
            out_path=RAW_DIR / "events_enrichment.csv",
        )
    except Exception as e:
        log.warning(f"Events enrichment failed: {e}. Continuing without event signals.")
        events = pd.DataFrame(columns=["date"])

    # --- Merge ---
    enrichment = all_dates.copy()
    for df, cols in [
        (airport, ["date","airport_arrivals","airport_arrivals_lag1"]),
        (cruise,  ["date","cruise_ship_flag","ships_in_port_count","cruise_passenger_estimate"]),
        (hols,    ["date","bank_holiday_flag"]),
        (events,  [c for c in events.columns]),   # all event columns
    ]:
        if not df.empty and len(df.columns) > 1:
            avail = [c for c in cols if c in df.columns]
            enrichment = enrichment.merge(df[avail], on="date", how="left")

    # Forward-fill airport arrivals for quarters not yet published (e.g. current quarter).
    # The quarterly cadence means the most recent quarter may be missing; use the last
    # known daily value rather than leaving NaN or zeroing out.
    for col in ["airport_arrivals", "airport_arrivals_lag1"]:
        if col in enrichment.columns:
            enrichment[col] = enrichment[col].ffill().bfill()
            enrichment[col] = pd.to_numeric(enrichment[col], errors="coerce").fillna(0).astype(int)

    # Fill missing flags with 0
    all_flag_cols = [c for c in enrichment.columns
                     if c.endswith("_flag") or c in [
                         "ships_in_port_count", "cruise_passenger_estimate",
                         "event_impact_score", "days_from_payday",
                     ]]
    for col in all_flag_cols:
        if col in enrichment.columns:
            enrichment[col] = enrichment[col].fillna(0)
            if col not in ("event_impact_score", "days_from_payday"):
                enrichment[col] = enrichment[col].astype(int)

    if out_path:
        enrichment.to_csv(out_path, index=False)
        log.info(f"\nEnrichment table saved → {out_path}")
        log.info(f"  Shape: {enrichment.shape}")
        log.info(f"  Date range: {enrichment['date'].min()} → {enrichment['date'].max()}")

    return enrichment


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from datetime import date as date_cls

    start = sys.argv[1] if len(sys.argv) > 1 else "2023-01-01"
    end   = sys.argv[2] if len(sys.argv) > 2 else str(date_cls.today())

    print(f"\nFetching public data: {start} → {end} + 7-day forecast\n")

    enrichment = build_enrichment_table(
        start=start,
        end=end,
        include_forecast_days=7,
        out_path=RAW_DIR / "enrichment.csv",
    )

    print(f"\n{'='*50}")
    print(enrichment.tail(14).to_string(index=False))
    print(f"\nAll sources fetched. Enrichment table ready for features.py.")
