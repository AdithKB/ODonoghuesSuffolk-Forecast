"""
TV sports fixtures fetcher for O'Donoghues demand forecasting.

Pulls fixtures from football-data.org free API (v4) for major leagues watched by
Irish pub-goers, computes intensity scores, and saves to data/raw/sports_fixtures.csv.

Also writes data/raw/six_nations_fixtures.csv as a static hardcoded list
(2023-2026 Six Nations schedule — public knowledge, no API needed).

Environment variables
---------------------
    FOOTBALL_DATA_API_KEY — required for football fixtures fetch.
                            If absent, fetch_sports_fixtures() returns an empty
                            DataFrame and logs a warning — the pipeline will NOT crash.

Competition codes (football-data.org v4):
    PL  — Premier League
    BL1 — Bundesliga
    CL  — UEFA Champions League
    EL  — UEFA Europa League
    PD  — La Liga
    WC  — FIFA World Cup
    EC  — UEFA European Championship
    UNL — UEFA Nations League (if available on free tier)

Free tier rate limit: 10 req/min → 7s sleep between requests to stay safe.
"""

import os
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Load .env at import time so FOOTBALL_DATA_API_KEY is available in all call sites
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — rely on env vars being set externally

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Competition configuration
# ---------------------------------------------------------------------------
# EL is restricted on the free tier (403); WC uses season 2026; EC uses season 2024.
# Each entry: (competition_code, [seasons_to_fetch])
# season = start year of competition (e.g., 2024 for 2024-25 PL season, 2024 for Euro 2024)
COMPETITION_SEASONS: list[tuple[str, list[int]]] = [
    ("PL",  [2023, 2024, 2025]),   # Premier League
    ("BL1", [2023, 2024, 2025]),   # Bundesliga
    ("CL",  [2023, 2024, 2025]),   # Champions League
    ("PD",  [2023, 2024, 2025]),   # La Liga
    ("EC",  [2024]),               # UEFA Euros 2024
    ("WC",  [2026]),               # FIFA World Cup 2026
    # EL (Europa League) is NOT accessible on free tier — skipped
]

COMPETITION_NAMES = {
    "PL":  "Premier League",
    "BL1": "Bundesliga",
    "CL":  "Champions League",
    "EL":  "Europa League",
    "PD":  "La Liga",
    "WC":  "World Cup",
    "EC":  "Euros",
    "UNL": "Nations League",
}

# Premier League big-6 clubs for derby detection.
# The API returns full names like "Arsenal FC", "Manchester City FC".
PL_BIG_6 = {
    "Arsenal FC", "Chelsea FC", "Liverpool FC", "Manchester City FC",
    "Manchester United FC", "Tottenham Hotspur FC",
}

# API stage values that indicate knockout rounds (actual values from the API)
KNOCKOUT_STAGES = {
    "FINAL", "SEMI_FINALS", "QUARTER_FINALS", "LAST_16", "LAST_32",
    "PLAYOFFS", "THIRD_PLACE",
}

# Ireland football team identifiers as returned by the API
IRELAND_TEAM_IDS = {
    "Republic of Ireland", "Ireland", "Rep. of Ireland", "Rep of Ireland",
}


# ---------------------------------------------------------------------------
# Six Nations static fixture list (public knowledge — no API needed)
# ---------------------------------------------------------------------------
# Format: (date_str, kickoff_local_IST, home_team, away_team, is_ireland_match)
# Kickoff times are approximate local Dublin times (UTC/UTC+1 depending on season).
# Feb–March matches are in GMT (UTC+0); we record in Dublin local time.

SIX_NATIONS_FIXTURES = [
    # ── 2023 Six Nations ──────────────────────────────────────────────────
    ("2023-02-04", "14:15", "Ireland",  "Wales",    True),
    ("2023-02-04", "16:45", "France",   "Italy",    False),
    ("2023-02-05", "14:15", "Scotland", "England",  False),
    ("2023-02-11", "14:15", "Wales",    "Scotland", False),
    ("2023-02-11", "16:45", "Italy",    "England",  False),
    ("2023-02-12", "14:15", "France",   "Ireland",  True),
    ("2023-02-25", "14:15", "Ireland",  "Scotland", True),
    ("2023-02-25", "16:45", "England",  "Italy",    False),
    ("2023-02-26", "14:15", "Wales",    "France",   False),
    ("2023-03-11", "14:15", "Scotland", "France",   False),
    ("2023-03-11", "16:45", "Italy",    "Ireland",  True),
    ("2023-03-12", "15:00", "England",  "Wales",    False),
    ("2023-03-18", "14:15", "Scotland", "Italy",    False),
    ("2023-03-18", "16:45", "Ireland",  "England",  True),   # Grand Slam decider
    ("2023-03-18", "20:00", "France",   "Wales",    False),
    # ── 2024 Six Nations ──────────────────────────────────────────────────
    ("2024-02-02", "19:30", "France",   "Ireland",  True),
    ("2024-02-03", "14:15", "Scotland", "Wales",    False),
    ("2024-02-03", "16:45", "England",  "Italy",    False),
    ("2024-02-10", "14:15", "Ireland",  "Italy",    True),
    ("2024-02-10", "16:45", "Wales",    "England",  False),
    ("2024-02-11", "14:15", "Scotland", "France",   False),
    ("2024-02-24", "14:15", "England",  "Scotland", False),
    ("2024-02-24", "16:45", "Ireland",  "Wales",    True),
    ("2024-02-25", "14:15", "Italy",    "France",   False),
    ("2024-03-09", "14:15", "France",   "England",  False),
    ("2024-03-09", "16:45", "Wales",    "Ireland",  True),
    ("2024-03-09", "19:00", "Italy",    "Scotland", False),
    ("2024-03-16", "14:15", "Italy",    "Wales",    False),
    ("2024-03-16", "16:45", "Scotland", "Ireland",  True),
    ("2024-03-16", "19:00", "England",  "France",   False),
    # ── 2025 Six Nations ──────────────────────────────────────────────────
    ("2025-02-01", "14:15", "Ireland",  "England",  True),
    ("2025-02-01", "16:45", "France",   "Wales",    False),
    ("2025-02-02", "14:15", "Scotland", "Italy",    False),
    ("2025-02-08", "14:15", "Wales",    "Scotland", False),
    ("2025-02-08", "16:45", "Italy",    "Ireland",  True),
    ("2025-02-09", "14:15", "England",  "France",   False),
    ("2025-02-22", "14:15", "Ireland",  "France",   True),
    ("2025-02-22", "16:45", "England",  "Wales",    False),
    ("2025-02-23", "14:15", "Italy",    "Scotland", False),
    ("2025-03-08", "14:15", "Scotland", "England",  False),
    ("2025-03-08", "16:45", "France",   "Italy",    False),
    ("2025-03-08", "19:00", "Wales",    "Ireland",  True),
    ("2025-03-15", "14:15", "Ireland",  "Scotland", True),   # potential Grand Slam
    ("2025-03-15", "16:45", "Wales",    "Italy",    False),
    ("2025-03-15", "19:00", "France",   "England",  False),
    # ── 2026 Six Nations ──────────────────────────────────────────────────
    ("2026-01-31", "14:15", "England",  "France",   False),
    ("2026-01-31", "16:45", "Wales",    "Ireland",  True),
    ("2026-02-01", "14:15", "Scotland", "Italy",    False),
    ("2026-02-07", "14:15", "France",   "Scotland", False),
    ("2026-02-07", "16:45", "Ireland",  "Italy",    True),
    ("2026-02-08", "14:15", "England",  "Wales",    False),
    ("2026-02-21", "14:15", "Scotland", "Wales",    False),
    ("2026-02-21", "16:45", "Italy",    "France",   False),
    ("2026-02-22", "14:15", "Ireland",  "England",  True),
    ("2026-03-07", "14:15", "Wales",    "France",   False),
    ("2026-03-07", "16:45", "England",  "Italy",    False),
    ("2026-03-08", "14:15", "Ireland",  "Scotland", True),
    ("2026-03-14", "14:15", "Italy",    "England",  False),
    ("2026-03-14", "16:45", "France",   "Ireland",  True),
    ("2026-03-14", "19:00", "Scotland", "Wales",    False),
]

# Final-round Ireland matches that are potential Grand Slam / championship deciders
HIGH_STAKES_IRELAND_DATES = {
    "2023-03-18",  # Ireland vs England — Grand Slam decider
    "2025-03-15",  # Ireland vs Scotland — Grand Slam possible
}


# ---------------------------------------------------------------------------
# Intensity scoring
# ---------------------------------------------------------------------------
def _score_intensity(competition: str, stage: str, home: str, away: str) -> int:
    """
    Return an intensity score 1–3 for a fixture using the API's actual stage values.

    3 — WC/EC knockout (LAST_16+), CL FINAL, Ireland national team match
    2 — CL any round, WC/EC GROUP_STAGE, PL big-6 derby
    1 — Regular PL / Bundesliga / La Liga match
    """
    stage_norm = (stage or "").upper().strip()
    is_knockout = stage_norm in KNOCKOUT_STAGES
    is_final = stage_norm == "FINAL"

    # Ireland national team — automatic 3
    if home in IRELAND_TEAM_IDS or away in IRELAND_TEAM_IDS:
        return 3

    if competition in ("WC", "EC"):
        return 3 if is_knockout else 2  # group stage = 2, knockout = 3

    if competition == "CL":
        return 3 if is_final else 2  # CL Final = 3, everything else = 2

    if competition == "EL":
        return 2 if is_knockout else 1

    if competition == "PL":
        if home in PL_BIG_6 and away in PL_BIG_6:
            return 2  # big-6 derby
        return 1

    # BL1, PD, and others — standard league match
    return 1


# ---------------------------------------------------------------------------
# API fetch helpers
# ---------------------------------------------------------------------------
def _fetch_competition_season(
    api_key: str,
    code: str,
    season: int,
    session: requests.Session,
) -> list[dict]:
    """
    Fetch all fixtures for one competition + season from football-data.org.

    Uses the season parameter (start year) instead of dateFrom/dateTo because
    the free tier limits date-range queries to 10 days maximum.
    Returns raw match dicts or an empty list on any error.
    """
    url = f"https://api.football-data.org/v4/competitions/{code}/matches"
    headers = {"X-Auth-Token": api_key}
    params = {"season": season}

    try:
        resp = session.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 404:
            logger.info("Competition %s / season %d not found (404) — skipping.", code, season)
            return []
        if resp.status_code == 403:
            logger.info("Competition %s not on free tier (403) — skipping.", code)
            return []
        if resp.status_code == 429:
            logger.warning("Rate limited on %s/%d — sleeping 65s and retrying.", code, season)
            time.sleep(65)
            resp = session.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Failed to fetch %s season %d: %s", code, season, exc)
        return []

    return resp.json().get("matches", [])


def _parse_match(match: dict, competition: str) -> dict | None:
    """Parse a single match dict into a fixture row dict, or None if invalid."""
    utc_date = match.get("utcDate", "")
    if not utc_date:
        return None

    try:
        dt_utc = datetime.fromisoformat(utc_date.replace("Z", "+00:00"))
    except ValueError:
        return None

    # Convert to Dublin local time (IST = UTC+1 all year; close enough for pub forecasting)
    dt_ist = dt_utc + timedelta(hours=1)

    home = (match.get("homeTeam") or {}).get("name", "Unknown")
    away = (match.get("awayTeam") or {}).get("name", "Unknown")
    stage = match.get("stage", "") or ""
    group = match.get("group", "") or ""
    stage_str = f"{stage} {group}".strip()

    intensity = _score_intensity(competition, stage_str, home, away)
    is_ireland = home in IRELAND_TEAM_IDS or away in IRELAND_TEAM_IDS

    return {
        "date": dt_ist.date().isoformat(),
        "kickoff_local": dt_ist.strftime("%H:%M"),
        "competition": COMPETITION_NAMES.get(competition, competition),
        "home_team": home,
        "away_team": away,
        "intensity": intensity,
        "is_ireland_match": is_ireland,
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def fetch_sports_fixtures(
    date_from: str = "2023-01-01",
    date_to: str | None = None,
    out_path: str | Path = "data/raw/sports_fixtures.csv",
) -> pd.DataFrame:
    """
    Fetch TV sports fixtures from football-data.org API (free tier).

    Strategy: fetches per-competition, per-season using the ``season`` parameter
    rather than date ranges (the free tier limits date-range queries to 10 days).
    Fetches seasons 2023, 2024, 2025 for PL / BL1 / CL / PD; season 2024 for EC;
    season 2026 for WC. EL is not accessible on the free tier (403) and is skipped.

    If ``FOOTBALL_DATA_API_KEY`` env var is absent, logs a warning and returns
    an empty DataFrame — the pipeline must not crash.

    Parameters
    ----------
    date_from : ISO date string; matches earlier than this date are filtered out.
    date_to   : ISO date string; matches later than this date are filtered out.
                Defaults to today + 90 days.
    out_path  : path to write the output CSV.

    Returns
    -------
    pd.DataFrame with columns:
        date, kickoff_local, competition, home_team, away_team,
        intensity, is_ireland_match
    """
    _empty_cols = ["date", "kickoff_local", "competition",
                   "home_team", "away_team", "intensity", "is_ireland_match"]

    api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "").strip()
    if not api_key:
        logger.warning(
            "FOOTBALL_DATA_API_KEY not set — skipping sports fixtures fetch. "
            "Set the env var and rerun to enable the TV sports demand signal."
        )
        print(
            "WARNING: FOOTBALL_DATA_API_KEY not set — sports fixtures fetch skipped.\n"
            "         Set the env var to enable the TV sports demand signal."
        )
        return pd.DataFrame(columns=_empty_cols)

    if date_to is None:
        date_to = (datetime.today() + timedelta(days=90)).strftime("%Y-%m-%d")

    date_from_dt = datetime.strptime(date_from, "%Y-%m-%d")
    date_to_dt   = datetime.strptime(date_to,   "%Y-%m-%d")

    print(f"Fetching sports fixtures {date_from} → {date_to} ...")
    session = requests.Session()
    all_rows: list[dict] = []
    fixture_counts: dict[str, int] = {}

    # Total requests = sum of seasons per competition
    total_reqs = sum(len(seasons) for _, seasons in COMPETITION_SEASONS)
    req_num = 0

    for code, seasons in COMPETITION_SEASONS:
        label = COMPETITION_NAMES.get(code, code)
        comp_rows = 0
        for season in seasons:
            req_num += 1
            print(
                f"  [{req_num}/{total_reqs}] {code} ({label}) season {season} ...",
                end="", flush=True,
            )
            matches = _fetch_competition_season(api_key, code, season, session)
            rows = []
            for m in matches:
                r = _parse_match(m, code)
                if r is None:
                    continue
                # Filter to the requested date window
                match_dt = datetime.strptime(r["date"], "%Y-%m-%d")
                if match_dt < date_from_dt or match_dt > date_to_dt:
                    continue
                rows.append(r)
            all_rows.extend(rows)
            comp_rows += len(rows)
            print(f" {len(rows)} fixtures (season {season})")
            # Free tier: 10 req/min → sleep 7s between every call
            if req_num < total_reqs:
                time.sleep(7)

        fixture_counts[label] = comp_rows

    if not all_rows:
        print("No fixtures fetched (check API key and network connectivity).")
        return pd.DataFrame(columns=_empty_cols)

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["date", "kickoff_local", "home_team", "away_team"])
    df = df.sort_values(["date", "kickoff_local"]).reset_index(drop=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} football fixtures → {out_path}")
    print("Counts by competition:")
    for comp, n in fixture_counts.items():
        print(f"  {comp}: {n}")

    return df


def build_six_nations_csv(
    out_path: str | Path = "data/raw/six_nations_fixtures.csv",
) -> pd.DataFrame:
    """
    Write the static Six Nations 2023–2026 fixture list to CSV.

    Intensity scoring:
      3 — Ireland match AND a Grand Slam / championship decider round
      2 — Ireland match (non-decider) OR non-Ireland match in Six Nations
          (all Six Nations matches drive high pub viewership in Ireland)

    Returns
    -------
    pd.DataFrame with the fixture rows.
    """
    rows = []
    for date_str, kickoff, home, away, is_ireland in SIX_NATIONS_FIXTURES:
        if is_ireland and date_str in HIGH_STAKES_IRELAND_DATES:
            intensity = 3
        else:
            intensity = 2  # all Six Nations have strong pub viewership in Ireland

        rows.append({
            "date": date_str,
            "kickoff_local": kickoff,
            "competition": "Six Nations",
            "home_team": home,
            "away_team": away,
            "intensity": intensity,
            "is_ireland_match": is_ireland,
        })

    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} Six Nations fixtures → {out_path}")
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from datetime import date as _date

    print("=" * 60)
    print("Building Six Nations CSV (static hardcoded list) ...")
    print("=" * 60)
    sn = build_six_nations_csv()
    print(f"Six Nations: {len(sn)} fixtures written\n")

    print("=" * 60)
    print("Fetching football fixtures from football-data.org ...")
    print("=" * 60)
    end_date = (_date.today() + timedelta(days=90)).strftime("%Y-%m-%d")
    fixtures = fetch_sports_fixtures(
        date_from="2023-01-01",
        date_to=end_date,
    )

    if not fixtures.empty:
        print("\n=== Fixture counts by competition ===")
        counts = fixtures.groupby("competition").size()
        for comp, n in counts.items():
            print(f"  {comp}: {n}")
        print(f"\nTotal football fixtures: {len(fixtures)}")
        print(f"Date range: {fixtures['date'].min()} → {fixtures['date'].max()}")
    else:
        print("No football fixtures fetched (FOOTBALL_DATA_API_KEY not set or API error).")

    sys.exit(0)
