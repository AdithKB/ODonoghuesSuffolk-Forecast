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
            "apparent_temperature",
            "wind_gusts_10m",
            "sunshine_duration",
        ]),
        "timezone": DUBLIN_TZ,
    }
    log.info(f"Fetching historical hourly weather {start} → {end} from Open-Meteo…")
    resp = requests.get(url, params=params, timeout=REQUESTS_TIMEOUT, headers=REQUESTS_HEADERS)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    df = pd.DataFrame({
        "timestamp_hour":      pd.to_datetime(data["time"]),
        "temp_c":              data["temperature_2m"],
        "rain_mm":             data["precipitation"],
        "wind_speed_kmh":      data["wind_speed_10m"],
        "weather_code":        data["weathercode"],
        "apparent_temp_c":     data["apparent_temperature"],
        "wind_gusts_kmh":      data["wind_gusts_10m"],
        "sunshine_duration_min": [v / 60 if v is not None else 0.0
                                  for v in data["sunshine_duration"]],
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
            "apparent_temperature",
            "wind_gusts_10m",
            "sunshine_duration",
            "uv_index",
            "precipitation_probability",
        ]),
        "timezone":     DUBLIN_TZ,
        "forecast_days": days,
    }
    log.info(f"Fetching {days}-day hourly weather forecast from Open-Meteo…")
    resp = requests.get(url, params=params, timeout=REQUESTS_TIMEOUT, headers=REQUESTS_HEADERS)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    df = pd.DataFrame({
        "timestamp_hour":      pd.to_datetime(data["time"]),
        "temp_c":              data["temperature_2m"],
        "rain_mm":             data["precipitation"],
        "wind_speed_kmh":      data["wind_speed_10m"],
        "weather_code":        data["weathercode"],
        "apparent_temp_c":     data["apparent_temperature"],
        "wind_gusts_kmh":      data["wind_gusts_10m"],
        "sunshine_duration_min": [v / 60 if v is not None else 0.0
                                  for v in data["sunshine_duration"]],
        "uv_index":            data.get("uv_index", [0.0] * len(data["time"])),
        "precip_probability":  data.get("precipitation_probability", [0.0] * len(data["time"])),
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

    # New columns only available from forecast; fill 0 for historical rows
    new_wx_cols = [
        "apparent_temp_c", "wind_gusts_kmh", "sunshine_duration_min",
        "uv_index", "precip_probability",
    ]
    for col in new_wx_cols:
        if col not in weather.columns:
            weather[col] = 0.0
        else:
            weather[col] = weather[col].fillna(0.0)

    # Forward-fill weather for forecast hours that might not have data yet
    wx_cols = ["temp_c", "rain_mm", "wind_speed_kmh"] + new_wx_cols
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

# CruiseTimetables.com — correct URL slug format (year-specific pages)
CRUISETIMETABLES_BASE = "https://www.cruisetimetables.com/dublin-ireland-cruise-ship-schedule-{year}.html"
DUN_LAOGHAIRE_URL     = "https://www.cruisetimetables.com/dun-laoghaire-ireland-cruise-ship-schedule-{year}.html"
# Current (no-year) page — lists upcoming season(s); used for forward-looking top-up
CRUISETIMETABLES_CURRENT = "https://www.cruisetimetables.com/dublin-ireland-cruise-ship-schedule.html"

# Known large passenger vessels (used to identify cruise ships vs cargo)
CRUISE_KEYWORDS = [
    "CRUISE", "CELEBRITY", "ROYAL CARIBBEAN", "MSC ", "NORWEGIAN",
    "CARNIVAL", "PRINCESS", "CUNARD", "SILVERSEA", "VIKING",
    "MARELLA", "TUI ", "HURTIGRUTEN", "SEABOURN", "AZAMARA",
    "EVRIMA", "SCENIC", "PONANT", "WINDSTAR",
]


_CT_MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


def _parse_ct_cell(text: str) -> list:
    """
    Parse a CruiseTimetables table cell.
    Cell format (newline-separated):
        Month YYYY
        DD
        DD
        ...
    Returns list of date objects.
    """
    import re, datetime as dt
    dates, cur_m, cur_y = [], None, None
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(
            r"^(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+(\d{4})$", line
        )
        if m:
            cur_m, cur_y = _CT_MONTHS[m.group(1)], int(m.group(2))
            continue
        if re.match(r"^\d{1,2}$", line) and cur_m and cur_y:
            try:
                dates.append(dt.date(cur_y, cur_m, int(line)))
            except ValueError:
                pass
    return dates


def _scrape_cruisetimetables(url: str) -> dict:
    """
    Scrape a cruisetimetables.com page and return {date: ship_count}.
    Handles both the per-row format (older pages) and the cell-based monthly
    calendar format used on year-specific slug pages.
    """
    cruise_dates: dict = {}
    try:
        resp = requests.get(url, timeout=REQUESTS_TIMEOUT, headers=REQUESTS_HEADERS)
        resp.raise_for_status()
    except Exception as e:
        log.debug(f"cruisetimetables fetch failed ({url}): {e}")
        return cruise_dates

    soup = BeautifulSoup(resp.text, "html.parser")
    for table in soup.find_all("table"):
        for td in table.find_all("td"):
            cell_text = td.get_text("\n")
            for d in _parse_ct_cell(cell_text):
                cruise_dates[d] = cruise_dates.get(d, 0) + 1
    return cruise_dates


def fetch_cruise_schedule(
    save_path: Path | None = None,
    fallback_on_error: bool = True,
    start_year: int = 2024,
) -> pd.DataFrame:
    """
    Build a historical + forward-looking cruise schedule for Dublin / Dun Laoghaire.

    Strategy:
    1. Load existing save_path CSV as historical base (pre-populated from
       official DLRCOCO PDFs: data/pubdata/cruise_schedule_dublin.csv).
    2. Top up with CruiseTimetables.com current-season page for future dates.
    3. Merge: existing rows win for past dates; online data adds new future dates.

    Returns daily DataFrame:
        date, cruise_ship_flag, ships_in_port_count, cruise_passenger_estimate
    """
    import datetime as dt
    today = dt.date.today()
    cruise_dates: dict = {}  # {date: ship_count}

    # ── 1. Load existing historical CSV ────────────────────────────────────
    historical_path = save_path or (RAW_DIR / "cruise_schedule.csv")
    if Path(historical_path).exists():
        hist = pd.read_csv(historical_path)
        hist["date"] = pd.to_datetime(hist["date"]).dt.date
        for _, row in hist.iterrows():
            cruise_dates[row["date"]] = int(row.get("ships_in_port_count", 1))
        log.info(f"Loaded {len(hist)} historical cruise dates from {historical_path}")

    # ── 2. Top up with CruiseTimetables current-season page ────────────────
    found_online = _scrape_cruisetimetables(CRUISETIMETABLES_CURRENT)
    if not found_online:
        # fall back to year-specific URL
        for url_tpl in [CRUISETIMETABLES_BASE, DUN_LAOGHAIRE_URL]:
            found_online.update(_scrape_cruisetimetables(url_tpl.format(year=today.year)))
            found_online.update(_scrape_cruisetimetables(url_tpl.format(year=today.year + 1)))

    for d, cnt in found_online.items():
        if d > today:  # only add future dates from online source
            cruise_dates[d] = max(cruise_dates.get(d, 0), cnt)

    log.info(f"Online top-up: {sum(1 for d in found_online if d > today)} future cruise dates")

    # ── 3. Dublin Port forward-looking (next 100 arrivals) ─────────────────
    try:
        resp = requests.get(DUBLIN_PORT_URL, timeout=REQUESTS_TIMEOUT, headers=REQUESTS_HEADERS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for table in soup.find_all("table"):
            for tr in table.find_all("tr")[1:]:
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if len(cells) < 4:
                    continue
                vessel_type = str(cells[3]).strip().lower()
                vessel_name = str(cells[2]).strip().upper() if len(cells) > 2 else ""
                is_cruise = (
                    "cruise" in vessel_type
                    or any(kw in vessel_name for kw in CRUISE_KEYWORDS)
                )
                if not is_cruise:
                    continue
                try:
                    parsed = pd.to_datetime(cells[0].strip(), dayfirst=True, errors="coerce")
                    if pd.notna(parsed) and parsed.date() > today:
                        d = parsed.date()
                        cruise_dates[d] = cruise_dates.get(d, 0) + 1
                except Exception:
                    continue
    except Exception as e:
        log.debug(f"Dublin Port forward-looking scrape failed: {e}")

    if not cruise_dates:
        log.warning("No cruise data found from any source.")
        if not fallback_on_error:
            raise RuntimeError("No cruise data found")
        return pd.DataFrame(columns=["date","cruise_ship_flag","ships_in_port_count","cruise_passenger_estimate"])

    # Build final DataFrame — use actual pax estimate from historical where available
    hist_pax: dict = {}
    if Path(historical_path).exists():
        hist_df = pd.read_csv(historical_path)
        hist_df["date"] = pd.to_datetime(hist_df["date"]).dt.date
        if "cruise_passenger_estimate" in hist_df.columns:
            hist_pax = dict(zip(hist_df["date"], hist_df["cruise_passenger_estimate"]))

    records = []
    for d, count in sorted(cruise_dates.items()):
        pax = hist_pax.get(d, count * 2000)
        records.append({
            "date":                      d,
            "cruise_ship_flag":          1,
            "ships_in_port_count":       count,
            "cruise_passenger_estimate": pax,
        })

    df = pd.DataFrame(records)
    log.info(f"Total cruise days: {len(df)} across {df['date'].min()} → {df['date'].max()}")
    if save_path:
        df.to_csv(save_path, index=False)
        log.info(f"Saved → {save_path}")
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
