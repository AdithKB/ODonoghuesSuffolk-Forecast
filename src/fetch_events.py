"""
Event calendar + computed signals for O'Donoghues demand enrichment.

Three layers:
  1. Static events  — hardcoded calendar of confirmed Dublin events (rugby, GAA, concerts, festivals)
  2. Computed flags — school holidays, payday period, Bloomsday (rule-based, always current)
  3. Optional API   — Ticketmaster Discovery API (set TICKETMASTER_API_KEY env var to enable)

All produce daily DataFrames that merge into the main enrichment table.

Run directly:
    python src/fetch_events.py
"""

import os
import logging
from datetime import date, timedelta
from pathlib import Path

import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. STATIC EVENTS CALENDAR
#
# Source: manually verified from official venue/IRFU pages June 2026.
# Add new events as they are announced. The model uses these as binary flags
# on the day of the event.
#
# Venue impact on O'Donoghues Suffolk Street D02:
#   Aviva Stadium       4km south     VERY HIGH (rugby fans = pub culture, walk city)
#   Croke Park          3km north     HIGH      (GAA All-Ireland draws huge crowds)
#   3Arena              2km east      MEDIUM    (concert-goers pass city centre)
#   Olympia Theatre     0.3km         HIGH      (pre/post-show drinks on Suffolk St)
#   Gaiety Theatre      0.4km         HIGH      (ditto)
#   Marlay/Malahide     10km+         LOW       (crowd stays in own area)
# ---------------------------------------------------------------------------

STATIC_EVENTS: list[dict] = [
    # ── SIX NATIONS 2026 (all Ireland HOME matches at Aviva Stadium) ────────
    {"date": "2026-02-14", "event_name": "Ireland vs Italy (Six Nations)",   "venue": "Aviva Stadium", "category": "rugby",    "impact": "very_high"},
    {"date": "2026-03-06", "event_name": "Ireland vs Wales (Six Nations)",   "venue": "Aviva Stadium", "category": "rugby",    "impact": "very_high"},
    {"date": "2026-03-14", "event_name": "Ireland vs Scotland (Six Nations)","venue": "Aviva Stadium", "category": "rugby",    "impact": "very_high"},

    # ── AUTUMN NATIONS / NATIONS CHAMPIONSHIP 2026 ─────────────────────────
    {"date": "2026-11-06", "event_name": "Ireland vs Argentina",             "venue": "Aviva Stadium", "category": "rugby",    "impact": "very_high"},
    {"date": "2026-11-14", "event_name": "Ireland vs Fiji",                  "venue": "Aviva Stadium", "category": "rugby",    "impact": "very_high"},
    {"date": "2026-11-21", "event_name": "Ireland vs South Africa",          "venue": "Aviva Stadium", "category": "rugby",    "impact": "very_high"},

    # ── SIX NATIONS 2027 (Ireland home dates confirmed) ─────────────────────
    {"date": "2027-02-05", "event_name": "Ireland vs England (Six Nations)", "venue": "Aviva Stadium", "category": "rugby",    "impact": "very_high"},
    {"date": "2027-03-13", "event_name": "Ireland vs France (Six Nations)",  "venue": "Aviva Stadium", "category": "rugby",    "impact": "very_high"},

    # ── CONCERTS AT AVIVA STADIUM ───────────────────────────────────────────
    {"date": "2026-07-04", "event_name": "Take That / OneRepublic",          "venue": "Aviva Stadium", "category": "concert",  "impact": "high"},

    # ── GAA ALL-IRELAND 2026 (Croke Park) ──────────────────────────────────
    {"date": "2026-07-04", "event_name": "GAA Hurling All-Ireland SF (Galway v Cork)",   "venue": "Croke Park", "category": "gaa", "impact": "high"},
    {"date": "2026-07-05", "event_name": "GAA Hurling All-Ireland SF (Limerick v Clare)","venue": "Croke Park", "category": "gaa", "impact": "high"},
    {"date": "2026-08-02", "event_name": "LGFA TG4 All-Ireland Finals",                 "venue": "Croke Park", "category": "gaa", "impact": "medium"},
    {"date": "2026-09-05", "event_name": "Katie Taylor vs Flora Pili (boxing)",         "venue": "Croke Park", "category": "sports","impact": "high"},

    # ── CONCERTS AT CROKE PARK ──────────────────────────────────────────────
    {"date": "2026-08-30", "event_name": "Bon Jovi",                         "venue": "Croke Park",    "category": "concert",  "impact": "high"},

    # ── 3ARENA CONCERTS ────────────────────────────────────────────────────
    {"date": "2026-06-30", "event_name": "Lily Allen",                       "venue": "3Arena",        "category": "concert",  "impact": "medium"},
    {"date": "2026-07-01", "event_name": "Lily Allen",                       "venue": "3Arena",        "category": "concert",  "impact": "medium"},
    {"date": "2026-07-03", "event_name": "Wolfe Tones",                      "venue": "3Arena",        "category": "concert",  "impact": "medium"},
    {"date": "2026-07-04", "event_name": "Wolfe Tones",                      "venue": "3Arena",        "category": "concert",  "impact": "medium"},

    # ── OUTDOOR FESTIVALS ───────────────────────────────────────────────────
    {"date": "2026-06-28", "event_name": "Calvin Harris — Malahide Castle",  "venue": "Malahide Castle","category": "concert",  "impact": "low"},
    {"date": "2026-06-28", "event_name": "Florence + The Machine — Marlay Park","venue": "Marlay Park","category": "concert",   "impact": "low"},
    {"date": "2026-06-30", "event_name": "Empire of the Sun — Fairview Park","venue": "Fairview Park", "category": "concert",  "impact": "low"},

    # ── BLOOMSDAY / LITERARY FESTIVAL ───────────────────────────────────────
    # Full week June 11-16 near Suffolk Street / Trinity College area
    {"date": "2026-06-11", "event_name": "Bloomsday Festival begins",        "venue": "City Centre",   "category": "festival", "impact": "medium"},
    {"date": "2026-06-12", "event_name": "Bloomsday Festival",               "venue": "City Centre",   "category": "festival", "impact": "medium"},
    {"date": "2026-06-13", "event_name": "Bloomsday Festival",               "venue": "City Centre",   "category": "festival", "impact": "medium"},
    {"date": "2026-06-14", "event_name": "Bloomsday Festival",               "venue": "City Centre",   "category": "festival", "impact": "medium"},
    {"date": "2026-06-15", "event_name": "Bloomsday Festival",               "venue": "City Centre",   "category": "festival", "impact": "medium"},
    {"date": "2026-06-16", "event_name": "Bloomsday (main day — Ulysses June 16)","venue": "City Centre","category": "festival","impact": "high"},

    # ── NEW MUSIC DUBLIN ────────────────────────────────────────────────────
    {"date": "2026-04-15", "event_name": "New Music Dublin Festival",        "venue": "National Concert Hall","category": "festival","impact": "low"},
    {"date": "2026-04-16", "event_name": "New Music Dublin Festival",        "venue": "National Concert Hall","category": "festival","impact": "low"},
    {"date": "2026-04-17", "event_name": "New Music Dublin Festival",        "venue": "National Concert Hall","category": "festival","impact": "low"},
    {"date": "2026-04-18", "event_name": "New Music Dublin Festival",        "venue": "National Concert Hall","category": "festival","impact": "low"},
    {"date": "2026-04-19", "event_name": "New Music Dublin Festival",        "venue": "National Concert Hall","category": "festival","impact": "low"},

    # ── DUN LAOGHAIRE SUMMERFEST ────────────────────────────────────────────
    *[{"date": f"2026-07-{d:02d}", "event_name": "Dun Laoghaire Summerfest",
       "venue": "Dun Laoghaire", "category": "festival", "impact": "low"}
      for d in range(3, 13)],   # July 3-12
]


IMPACT_SCORE = {"very_high": 4, "high": 3, "medium": 2, "low": 1}


def build_static_event_calendar(save_path: Path | None = None) -> pd.DataFrame:
    """
    Compile all known static Dublin events into a daily DataFrame.

    Columns per day:
        date, aviva_event_flag, croke_park_event_flag, nearby_venue_event_flag,
        city_event_flag, event_impact_score, event_names
    """
    if not STATIC_EVENTS:
        return pd.DataFrame(columns=["date"])

    df = pd.DataFrame(STATIC_EVENTS)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    def agg_day(group):
        venues = set(group["venue"].tolist())
        max_impact = max(IMPACT_SCORE.get(i, 1) for i in group["impact"])
        return pd.Series({
            "aviva_event_flag":        int("Aviva Stadium" in venues),
            "croke_park_event_flag":   int("Croke Park" in venues),
            "nearby_venue_event_flag": int(any(v in venues for v in [
                "3Arena", "Olympia Theatre", "Gaiety Theatre",
                "National Concert Hall", "RDS",
            ])),
            "city_event_flag":         1,
            "event_impact_score":      max_impact,
            "event_names":             " | ".join(group["event_name"].tolist()),
        })

    daily = df.groupby("date").apply(agg_day, include_groups=False).reset_index()
    daily = daily.sort_values("date").reset_index(drop=True)

    if save_path:
        daily.to_csv(save_path, index=False)
        log.info(f"Saved {len(daily)} event days → {save_path}")
    return daily


# ---------------------------------------------------------------------------
# 2. SCHOOL HOLIDAY CALENDAR (Irish Department of Education pattern)
#
# Irish school terms (primary & secondary) follow a consistent pattern.
# These dates are approximated — the exact boundaries shift ±3 days per year
# but the rough periods are stable enough to be useful signals.
# ---------------------------------------------------------------------------

def _irish_school_holidays(year: int) -> list[tuple[date, date]]:
    """
    Return list of (start, end) inclusive ranges of Irish school holiday periods
    for the given year.
    """
    from dateutil.easter import easter

    easter_sun = easter(year)
    holidays = []

    # Summer: last Friday of June → last Monday of August
    # Approximate: June 26 → Aug 31 (safe window that covers all years)
    holidays.append((date(year, 6, 25), date(year, 8, 31)))

    # Christmas: ~Dec 23 → ~Jan 6
    holidays.append((date(year, 12, 23), date(year, 12, 31)))
    holidays.append((date(year, 1, 1),   date(year, 1, 6)))

    # October mid-term: ~Oct 27 → ~Nov 1 (Halloween week)
    holidays.append((date(year, 10, 26), date(year, 11, 1)))

    # February mid-term: ~Week 3 of February (approx Feb 16-20)
    holidays.append((date(year, 2, 16), date(year, 2, 20)))

    # Easter break: Good Friday (Easter - 2) through ~2 weeks after Easter
    good_friday = easter_sun - timedelta(days=2)
    easter_end  = easter_sun + timedelta(days=11)
    holidays.append((good_friday, easter_end))

    return holidays


def build_school_holiday_calendar(
    years: list[int],
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    Generate a daily school holiday flag for the given years.

    Columns: date, school_holiday_flag
    """
    holiday_dates: set[date] = set()
    for year in years:
        for start, end in _irish_school_holidays(year):
            d = start
            while d <= end:
                holiday_dates.add(d)
                d += timedelta(days=1)

    all_dates = pd.date_range(
        date(min(years), 1, 1), date(max(years), 12, 31), freq="D"
    ).date

    df = pd.DataFrame({
        "date": all_dates,
        "school_holiday_flag": [int(d in holiday_dates) for d in all_dates],
    })

    if save_path:
        df.to_csv(save_path, index=False)
        log.info(f"Saved school holiday calendar → {save_path}")
    return df


def build_trinity_term_calendar(
    years: list[int],
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    TCD term indicator — students in the Suffolk Street area drive lunch/evening demand.
    Trinity is 300m from O'Donoghues.

    Term pattern (approximate):
      Michaelmas: Early Sept → mid-Dec
      Hilary:     Mid-Jan → late April (with reading week)
      Trinity:    Early May → late May (revision/exams only)
    """
    term_dates: set[date] = set()
    for year in years:
        # Michaelmas term: Sept 7 → Dec 12
        for month, start_day, end_month, end_day in [
            (9, 7,   12, 12),   # Michaelmas
            (1, 15,   4, 28),   # Hilary
            (5, 1,    5, 29),   # Trinity (exam month)
        ]:
            s = date(year, month, start_day)
            try:
                e = date(year, end_month, end_day)
            except ValueError:
                e = date(year, end_month, 28)
            d = s
            while d <= e:
                term_dates.add(d)
                d += timedelta(days=1)

    all_dates = pd.date_range(
        date(min(years), 1, 1), date(max(years), 12, 31), freq="D"
    ).date
    df = pd.DataFrame({
        "date": all_dates,
        "college_term_flag": [int(d in term_dates) for d in all_dates],
    })
    if save_path:
        df.to_csv(save_path, index=False)
    return df


# ---------------------------------------------------------------------------
# 3. PAYDAY PERIOD FLAG
#
# Most Irish employees are paid monthly (end of month or 25th).
# The period around payday (last ~7 days of the month + first 5 days of next)
# correlates with higher discretionary spend in hospitality.
# ---------------------------------------------------------------------------

def build_payday_calendar(
    years: list[int],
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    Flag the payday-spending window: 25th → month end + 1st → 5th.
    Columns: date, payday_period_flag, days_from_payday
    """
    all_dates = pd.date_range(
        date(min(years), 1, 1), date(max(years), 12, 31), freq="D"
    ).date

    records = []
    for d in all_dates:
        dom = d.day
        is_payday = (dom >= 25) or (dom <= 5)
        # days_from_payday: negative = days before payday, positive = days after
        if dom >= 25:
            days_from = dom - 25  # 0 on the 25th
        else:
            days_from = dom - 5   # negative values early in month
        records.append({
            "date": d,
            "payday_period_flag": int(is_payday),
            "days_from_payday": days_from,
        })

    df = pd.DataFrame(records)
    if save_path:
        df.to_csv(save_path, index=False)
        log.info(f"Saved payday calendar → {save_path}")
    return df


# ---------------------------------------------------------------------------
# 4. RECURRING ANNUAL FLAGS (no scraping needed)
# ---------------------------------------------------------------------------

def build_annual_flags(years: list[int], save_path: Path | None = None) -> pd.DataFrame:
    """
    Recurring annual events near O'Donoghues that are always on fixed dates.

    Columns: date, st_patricks_week_flag, bloomsday_flag, summer_tourism_flag,
             new_years_eve_flag, christmas_market_flag
    """
    records = []
    all_dates = pd.date_range(
        date(min(years), 1, 1), date(max(years), 12, 31), freq="D"
    ).date

    for d in all_dates:
        # St Patrick's week: March 13-17
        st_pats = int(d.month == 3 and 13 <= d.day <= 17)

        # Bloomsday: June 16 (main day) and festival June 11-16
        bloomsday_main  = int(d.month == 6 and d.day == 16)
        bloomsday_week  = int(d.month == 6 and 11 <= d.day <= 16)

        # Peak summer tourism: June 20 → August 31
        summer_tourism  = int((d.month == 6 and d.day >= 20) or
                              d.month in [7, 8] or
                              (d.month == 8 and d.day <= 31))

        # Dublin St Patrick's Festival period also includes March 13-17 parade build-up
        # Bonus: proximity to Grafton Street Christmas markets (late Nov → Dec 23)
        christmas_market = int(
            (d.month == 11 and d.day >= 20) or
            (d.month == 12 and d.day <= 23)
        )

        # New Year's Eve (massive for a city-centre pub)
        nye = int(d.month == 12 and d.day == 31)

        # New Year's Day (hangover / quieter)
        nya = int(d.month == 1 and d.day == 1)

        records.append({
            "date":                 d,
            "st_patricks_week_flag":st_pats,
            "bloomsday_flag":       bloomsday_main,
            "bloomsday_week_flag":  bloomsday_week,
            "summer_tourism_flag":  summer_tourism,
            "christmas_market_flag":christmas_market,
            "new_years_eve_flag":   nye,
            "new_years_day_flag":   nya,
        })

    df = pd.DataFrame(records)
    if save_path:
        df.to_csv(save_path, index=False)
        log.info(f"Saved annual flags → {save_path}")
    return df


# ---------------------------------------------------------------------------
# 5. OPTIONAL: TICKETMASTER DISCOVERY API
#
# Set env var TICKETMASTER_API_KEY to enable.
# Free tier: 5000 calls/day. Register at: https://developer.ticketmaster.com
# Fetches major events within radius_km of O'Donoghues.
# ---------------------------------------------------------------------------

ODONOGHUES_LAT  =  53.3434
ODONOGHUES_LON  =  -6.2601


def fetch_ticketmaster_events(
    start: str,
    end: str,
    api_key: str | None = None,
    radius_km: int = 5,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    Fetch major events near O'Donoghues using Ticketmaster Discovery API.

    Parameters
    ----------
    start, end  : "YYYY-MM-DD"
    api_key     : Ticketmaster API key. Defaults to TICKETMASTER_API_KEY env var.
    radius_km   : Search radius in km from the pub (default 5)

    Returns daily DataFrame with: date, ticketmaster_event_flag, ticketmaster_event_names
    """
    api_key = api_key or os.environ.get("TICKETMASTER_API_KEY")
    if not api_key:
        log.info("No TICKETMASTER_API_KEY set. Skipping Ticketmaster fetch.")
        return pd.DataFrame(columns=["date", "ticketmaster_event_flag", "ticketmaster_event_names"])

    url = "https://app.ticketmaster.com/discovery/v2/events.json"
    params = {
        "apikey":       api_key,
        "latlong":      f"{ODONOGHUES_LAT},{ODONOGHUES_LON}",
        "radius":       radius_km,
        "unit":         "km",
        "startDateTime":f"{start}T00:00:00Z",
        "endDateTime":  f"{end}T23:59:59Z",
        "size":         200,
        "countryCode":  "IE",
    }

    log.info(f"Fetching Ticketmaster events {start} → {end} within {radius_km}km of pub…")
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning(f"Ticketmaster API error: {e}")
        return pd.DataFrame(columns=["date", "ticketmaster_event_flag", "ticketmaster_event_names"])

    events = data.get("_embedded", {}).get("events", [])
    log.info(f"  → {len(events)} events returned")

    records: dict[date, list[str]] = {}
    for ev in events:
        name = ev.get("name", "Unknown")
        dates_info = ev.get("dates", {}).get("start", {})
        local_date = dates_info.get("localDate")
        if not local_date:
            continue
        d = pd.Timestamp(local_date).date()
        records.setdefault(d, []).append(name)

    if not records:
        return pd.DataFrame(columns=["date", "ticketmaster_event_flag", "ticketmaster_event_names"])

    rows = [
        {"date": d, "ticketmaster_event_flag": 1, "ticketmaster_event_names": " | ".join(names)}
        for d, names in sorted(records.items())
    ]
    df = pd.DataFrame(rows)
    if save_path:
        df.to_csv(save_path, index=False)
        log.info(f"Saved {len(df)} Ticketmaster event days → {save_path}")
    return df


# ---------------------------------------------------------------------------
# 6. BUILD COMBINED EVENT ENRICHMENT TABLE
# ---------------------------------------------------------------------------

def build_event_enrichment(
    start: str,
    end: str,
    ticketmaster_api_key: str | None = None,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """
    Merge all event signals into a single daily enrichment table.

    Output columns:
        date,
        aviva_event_flag, croke_park_event_flag, nearby_venue_event_flag,
        city_event_flag, event_impact_score, event_names,
        school_holiday_flag, college_term_flag,
        payday_period_flag, days_from_payday,
        st_patricks_week_flag, bloomsday_flag, bloomsday_week_flag,
        summer_tourism_flag, christmas_market_flag, new_years_eve_flag, new_years_day_flag,
        ticketmaster_event_flag, ticketmaster_event_names
    """
    start_dt = pd.Timestamp(start).date()
    end_dt   = pd.Timestamp(end).date()
    years    = list(range(start_dt.year, end_dt.year + 1))

    # Full date spine
    all_dates = pd.DataFrame({
        "date": pd.date_range(start, end, freq="D").date
    })

    # --- Static events ---
    static = build_static_event_calendar(save_path=RAW_DIR / "events_static.csv")

    # --- Computed signals ---
    school  = build_school_holiday_calendar(years, save_path=RAW_DIR / "school_holidays.csv")
    college = build_trinity_term_calendar(years, save_path=RAW_DIR / "college_term.csv")
    payday  = build_payday_calendar(years, save_path=RAW_DIR / "payday.csv")
    annual  = build_annual_flags(years, save_path=RAW_DIR / "annual_flags.csv")

    # --- Optional Ticketmaster ---
    tmev = fetch_ticketmaster_events(
        start, end,
        api_key=ticketmaster_api_key,
        save_path=RAW_DIR / "ticketmaster_events.csv",
    )

    # --- Merge everything ---
    enrichment = all_dates.copy()
    for df in [static, school, college, payday, annual, tmev]:
        if not df.empty:
            enrichment = enrichment.merge(df, on="date", how="left")

    # Fill flag columns with 0
    flag_cols = [c for c in enrichment.columns if c.endswith("_flag") or c == "event_impact_score"]
    for col in flag_cols:
        if col in enrichment.columns:
            enrichment[col] = enrichment[col].fillna(0)
            if col != "event_impact_score":
                enrichment[col] = enrichment[col].astype(int)

    if "days_from_payday" in enrichment.columns:
        enrichment["days_from_payday"] = enrichment["days_from_payday"].fillna(0).astype(int)

    # Composite: any major event today
    sport_cols  = [c for c in enrichment.columns if c in ["aviva_event_flag", "croke_park_event_flag"]]
    if sport_cols:
        enrichment["major_sports_event_flag"] = (enrichment[sport_cols].sum(axis=1) > 0).astype(int)

    if out_path:
        enrichment.to_csv(out_path, index=False)
        log.info(f"\nEvent enrichment saved → {out_path}")
        log.info(f"  Shape: {enrichment.shape}")
        log.info(f"  Columns: {list(enrichment.columns)}")

    return enrichment


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from datetime import date as date_cls

    start = sys.argv[1] if len(sys.argv) > 1 else "2026-01-01"
    end   = sys.argv[2] if len(sys.argv) > 2 else "2026-12-31"

    print(f"\nBuilding event enrichment: {start} → {end}\n")
    enrichment = build_event_enrichment(
        start=start,
        end=end,
        out_path=RAW_DIR / "events_enrichment.csv",
    )

    # Show event-rich days
    event_days = enrichment[enrichment["city_event_flag"].fillna(0) > 0].copy()
    print(f"\nEvent days found: {len(event_days)}\n")
    show_cols = ["date", "aviva_event_flag", "croke_park_event_flag",
                 "event_impact_score", "event_names"]
    show_cols = [c for c in show_cols if c in event_days.columns]
    if not event_days.empty:
        print(event_days[show_cols].to_string(index=False))

    # Show payday + school holiday stats
    print(f"\nPayday period days:       {enrichment['payday_period_flag'].sum()}")
    print(f"School holiday days:      {enrichment['school_holiday_flag'].sum()}")
    print(f"College term days:        {enrichment['college_term_flag'].sum()}")
    print(f"St Patrick's week days:   {enrichment['st_patricks_week_flag'].sum()}")
    print(f"Summer tourism days:      {enrichment['summer_tourism_flag'].sum()}")
    print(f"Christmas market days:    {enrichment['christmas_market_flag'].sum()}")
