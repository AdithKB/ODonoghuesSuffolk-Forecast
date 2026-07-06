"""
Horse racing and Ireland international rugby fixture builder for O'Donoghues demand forecasting.

Provides:
  build_horse_racing_csv(out_path)  — writes data/raw/horse_racing_fixtures.csv
  build_ireland_rugby_csv(out_path) — writes data/raw/ireland_rugby_fixtures.csv

Both use the same column schema as six_nations_fixtures.csv:
  date, kickoff_local, competition, home_team, away_team, intensity, is_ireland_match

Horse racing: home_team = venue name, away_team = "" (no opposing team), is_ireland_match = True
Ireland rugby: is_ireland_match = True always (all Ireland national team fixtures)
"""

import io
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column schema (must match six_nations_fixtures.csv exactly)
# ---------------------------------------------------------------------------
COLS = ["date", "kickoff_local", "competition", "home_team", "away_team",
        "intensity", "is_ireland_match"]


# ===========================================================================
# HORSE RACING
# ===========================================================================

# Known Irish racecourses — used to match venue lines in the HRI PDF
IRISH_RACECOURSES = {
    "Leopardstown", "Punchestown", "The Curragh", "Curragh",
    "Fairyhouse", "Galway", "Dundalk", "Limerick", "Cork",
    "Navan", "Thurles", "Killarney", "Tipperary", "Sligo",
    "Roscommon", "Ballinrobe", "Kilbeggan", "Clonmel",
    "Down Royal", "Downpatrick", "Gowran Park", "Laytown",
    "Tramore", "Wexford", "Bellewstown", "Listowel",
}

# Date regex for HRI PDF text: "Tuesday 12th March" or "12 March 2026"
_DATE_RE = re.compile(
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)?"
    r"\s*(\d{1,2})(?:st|nd|rd|th)?\s+"
    r"(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)"
    r"(?:\s+(\d{4}))?",
    re.IGNORECASE,
)
_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _make_racing_row(venue: str, d: date, intensity: int,
                     kickoff: str = "14:00") -> dict:
    return {
        "date": d.isoformat(),
        "kickoff_local": kickoff,
        "competition": "Horse Racing",
        "home_team": venue,
        "away_team": "",
        "intensity": intensity,
        "is_ireland_match": True,
    }


def _add_range(rows: list, venue: str, start: date, n_days: int,
               intensity: int, kickoff: str = "14:00") -> None:
    for i in range(n_days):
        rows.append(_make_racing_row(venue, start + timedelta(days=i), intensity, kickoff))


def _static_racing_fixtures() -> list[dict]:
    """Return hard-coded high-profile Irish/UK horse racing fixtures for 2023–2026."""
    rows: list[dict] = []

    # ── Galway Festival (7 days, intensity 3, last Tuesday of July) ────────
    for start in [date(2023, 7, 25), date(2024, 7, 29),
                  date(2025, 7, 28), date(2026, 7, 27)]:
        _add_range(rows, "Galway Racecourse", start, 7, 3)

    # ── Leopardstown Christmas Festival (Dec 26–29, intensity 3) ──────────
    for year in [2023, 2024, 2025, 2026]:
        _add_range(rows, "Leopardstown", date(year, 12, 26), 4, 3, "13:00")

    # ── Punchestown Festival (5 days, late April, intensity 3) ────────────
    for start in [date(2023, 4, 25), date(2024, 4, 23),
                  date(2025, 4, 22), date(2026, 4, 28)]:
        _add_range(rows, "Punchestown", start, 5, 3)

    # ── Dublin Racing Festival / Leopardstown (first full weekend, intensity 2) ─
    for d in [
        date(2023, 2, 4), date(2023, 2, 5),
        date(2024, 2, 3), date(2024, 2, 4),
        date(2025, 2, 1), date(2025, 2, 2),
        date(2026, 2, 7), date(2026, 2, 8),
    ]:
        rows.append(_make_racing_row("Leopardstown", d, 2))

    # ── Irish Grand National – Fairyhouse (Easter Monday, intensity 2) ─────
    for d in [date(2023, 4, 10), date(2024, 4, 1),
              date(2025, 4, 21), date(2026, 4, 6)]:
        rows.append(_make_racing_row("Fairyhouse", d, 2))

    # ── Cheltenham Festival (4 days, intensity 3, kickoff 13:30) ──────────
    # Also covered by cheltenham_festival_flag in features.py; adding here too
    # so it flows through tv_sports features with the correct hour proximity.
    for start in [date(2023, 3, 14), date(2024, 3, 12),
                  date(2025, 3, 11), date(2026, 3, 10)]:
        _add_range(rows, "Cheltenham", start, 4, 3, "13:30")

    # ── Irish Derby – The Curragh (last Saturday of June, intensity 2) ─────
    for d in [date(2023, 7, 1), date(2024, 6, 29),
              date(2025, 6, 28), date(2026, 6, 27)]:
        rows.append(_make_racing_row("The Curragh", d, 2))

    # ── Grand National – Aintree UK (first Saturday of April, intensity 2) ─
    for d in [date(2023, 4, 15), date(2024, 4, 13),
              date(2025, 4, 5), date(2026, 4, 11)]:
        rows.append(_make_racing_row("Aintree", d, 2))

    return rows


def _hri_pdf_fixtures() -> list[dict]:
    """
    Best-effort fetch and parse of the HRI 2026 weekly fixture PDF.
    Requires pdfplumber (NOT in requirements.txt — optional).
    Returns [] on any failure so the pipeline never blocks.
    """
    try:
        import pdfplumber  # type: ignore  # noqa: F401
    except ImportError:
        logger.info(
            "pdfplumber not installed — skipping HRI PDF fetch "
            "(install with: pip install pdfplumber)"
        )
        return []

    url = (
        "https://www.hri.ie/HRI/media/HRI/Comms/Documents/"
        "2026-Irish-Racing-Fixture-List-(Weekly).pdf"
    )
    try:
        resp = requests.get(
            url, timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (compatible; fixture-fetcher/1.0)"},
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("HRI PDF fetch failed (%s) — using static list only", exc)
        return []

    rows: list[dict] = []
    default_year = 2026

    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            current_date: date | None = None
            for page in pdf.pages:
                text = page.extract_text() or ""
                for line in text.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    # Try to extract a date
                    m = _DATE_RE.search(line)
                    if m:
                        try:
                            day_num = int(m.group(1))
                            month_num = _MONTH_MAP[m.group(2).lower()]
                            year = int(m.group(3)) if m.group(3) else default_year
                            current_date = date(year, month_num, day_num)
                        except (ValueError, KeyError):
                            pass  # malformed — keep previous date context

                    if current_date is None:
                        continue

                    # Check for a racecourse name on this line
                    for venue in IRISH_RACECOURSES:
                        if venue.lower() in line.lower():
                            rows.append(_make_racing_row(venue, current_date, 1))
                            break  # one row per line

    except Exception as exc:
        logger.warning("HRI PDF parse failed (%s) — using static list only", exc)
        return []

    logger.info("HRI PDF: parsed %d fixture rows", len(rows))
    return rows


def build_horse_racing_csv(
    out_path: "str | Path" = "data/raw/horse_racing_fixtures.csv",
) -> pd.DataFrame:
    """
    Build and save the horse racing fixtures CSV.

    Merges static high-profile meetings (2023–2026) with best-effort HRI PDF rows.
    Static entries take priority on deduplication (date, home_team).

    Returns
    -------
    pd.DataFrame with columns: date, kickoff_local, competition, home_team,
                               away_team, intensity, is_ireland_match
    """
    static_rows = _static_racing_fixtures()
    hri_rows    = _hri_pdf_fixtures()

    df_static = pd.DataFrame(static_rows, columns=COLS)
    df_static["_priority"] = 0  # 0 = higher priority (static)

    if hri_rows:
        df_hri = pd.DataFrame(hri_rows, columns=COLS)
        df_hri["_priority"] = 1
        combined = pd.concat([df_static, df_hri], ignore_index=True)
        combined = combined.sort_values("_priority")
        combined = combined.drop_duplicates(subset=["date", "home_team"], keep="first")
        combined = combined.drop(columns=["_priority"])
        print(f"  HRI PDF contributed {len(df_hri)} rows (before dedup)")
    else:
        combined = df_static.drop(columns=["_priority"], errors="ignore")

    combined = combined.sort_values(["date", "kickoff_local"]).reset_index(drop=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"Saved {len(combined)} horse racing fixtures → {out_path}")
    return combined


# ===========================================================================
# IRELAND RUGBY (non-Six Nations internationals)
# ===========================================================================

# Opponent labels for Autumn Nations (informational; doesn't need to be exact)
_AUTUMN_OPPONENTS: dict[int, list[str]] = {
    2023: ["South Africa", "Fiji", "New Zealand"],
    2024: ["New Zealand", "Argentina", "Australia"],
    2025: ["South Africa", "Fiji", "Australia"],
    2026: ["New Zealand", "Argentina", "South Africa"],
}

_AUTUMN_NATIONS_DATES: dict[int, list[date]] = {
    2023: [date(2023, 11, 11), date(2023, 11, 19), date(2023, 11, 25)],
    2024: [date(2024, 11, 9),  date(2024, 11, 16), date(2024, 11, 23)],
    2025: [date(2025, 11, 8),  date(2025, 11, 15), date(2025, 11, 22)],
    2026: [date(2026, 11, 7),  date(2026, 11, 14), date(2026, 11, 21)],
}

_SUMMER_TOUR_DATES: dict[int, list[date]] = {
    2023: [date(2023, 6, 24), date(2023, 7, 1),  date(2023, 7, 8)],
    2024: [date(2024, 7, 6),  date(2024, 7, 13), date(2024, 7, 20)],
    2025: [date(2025, 6, 28), date(2025, 7, 5),  date(2025, 7, 12)],
    2026: [date(2026, 6, 27), date(2026, 7, 4),  date(2026, 7, 11)],
}


def _static_rugby_fixtures() -> list[dict]:
    """Hard-coded Ireland non-Six-Nations internationals 2023–2026."""
    rows: list[dict] = []

    # ── Autumn Nations Series (home at Aviva Stadium, intensity 3, 20:00) ──
    for year, match_dates in _AUTUMN_NATIONS_DATES.items():
        opponents = _AUTUMN_OPPONENTS.get(year, ["TBA", "TBA", "TBA"])
        for i, d in enumerate(match_dates):
            opp = opponents[i] if i < len(opponents) else "TBA"
            rows.append({
                "date": d.isoformat(),
                "kickoff_local": "20:00",
                "competition": "Autumn Nations Series",
                "home_team": "Ireland",
                "away_team": opp,
                "intensity": 3,
                "is_ireland_match": True,
            })

    # ── Ireland Summer Tour (away, intensity 2, ~11:00 Dublin time) ────────
    for year, match_dates in _SUMMER_TOUR_DATES.items():
        for d in match_dates:
            rows.append({
                "date": d.isoformat(),
                "kickoff_local": "11:00",
                "competition": "Ireland Summer Tour",
                "home_team": "Touring Side",
                "away_team": "Ireland",
                "intensity": 2,
                "is_ireland_match": True,
            })

    return rows


def _scrape_irfu_fixtures() -> list[dict]:
    """
    Best-effort scrape of the IRFU fixture list page.
    The page is likely a JavaScript SPA, in which case this returns [].
    Falls back cleanly on any network or parse error.
    """
    url = "https://www.irishrugby.ie/fixture/list/91"
    try:
        from bs4 import BeautifulSoup  # already in requirements.txt
    except ImportError:
        logger.warning("beautifulsoup4 not installed — skipping IRFU scrape")
        return []

    try:
        resp = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-IE,en;q=0.9",
            },
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("IRFU scrape request failed (%s) — using static list", exc)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # Try common fixture container selectors for rugby club sites
    fixture_containers = (
        soup.select(".fixture-item")
        or soup.select(".fixture-card")
        or soup.select(".fixtures__item")
        or soup.select("[data-fixture]")
        or soup.select(".c-fixture")
        or soup.select("article.fixture")
    )

    if not fixture_containers:
        logger.warning(
            "IRFU: no fixture containers found (page %d chars) — likely a JS SPA, "
            "using static list",
            len(resp.text),
        )
        return []

    rows: list[dict] = []
    for container in fixture_containers:
        try:
            # Extract date element
            date_el = (
                container.select_one("[class*=date]")
                or container.select_one("time")
                or container.select_one("[datetime]")
            )
            if date_el is None:
                continue
            raw_date = (
                date_el.get("datetime")
                or date_el.get("data-date")
                or date_el.get_text(strip=True)
            )
            if not raw_date:
                continue

            parsed_date = None
            for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d %B %Y", "%B %d, %Y",
                        "%Y-%m-%dT%H:%M:%S"]:
                try:
                    parsed_date = datetime.strptime(str(raw_date)[:20], fmt).date()
                    break
                except (ValueError, TypeError):
                    continue
            if parsed_date is None:
                continue

            # Extract team names
            teams = (
                container.select("[class*=team]")
                or container.select("[class*=club]")
                or container.select("[class*=participant]")
            )
            if len(teams) < 2:
                continue
            home_text = teams[0].get_text(strip=True)
            away_text = teams[1].get_text(strip=True)

            # Only Ireland national team fixtures
            if "ireland" not in home_text.lower() and "ireland" not in away_text.lower():
                continue

            # Classify competition
            comp_el = (
                container.select_one("[class*=competition]")
                or container.select_one("[class*=league]")
            )
            comp_text = comp_el.get_text(strip=True).lower() if comp_el else ""

            if "autumn" in comp_text or parsed_date.month == 11:
                competition, intensity, kickoff = "Autumn Nations Series", 3, "20:00"
            elif "summer" in comp_text or "tour" in comp_text or parsed_date.month in (6, 7):
                competition, intensity, kickoff = "Ireland Summer Tour", 2, "11:00"
            else:
                competition, intensity, kickoff = "Ireland Rugby", 2, "20:00"

            rows.append({
                "date": parsed_date.isoformat(),
                "kickoff_local": kickoff,
                "competition": competition,
                "home_team": home_text,
                "away_team": away_text,
                "intensity": intensity,
                "is_ireland_match": True,
            })

        except Exception:
            continue

    logger.info("IRFU scrape: extracted %d fixtures", len(rows))
    return rows


def build_ireland_rugby_csv(
    out_path: "str | Path" = "data/raw/ireland_rugby_fixtures.csv",
) -> pd.DataFrame:
    """
    Build and save Ireland non-Six-Nations rugby fixtures CSV.

    Tries to scrape the IRFU fixture page first; falls back to the hard-coded
    static list (Autumn Nations Series + Summer Tour 2023–2026) on any failure.

    Returns
    -------
    pd.DataFrame with columns: date, kickoff_local, competition, home_team,
                               away_team, intensity, is_ireland_match
    """
    static_rows  = _static_rugby_fixtures()
    scraped_rows = _scrape_irfu_fixtures()

    df_static = pd.DataFrame(static_rows, columns=COLS)

    if scraped_rows:
        df_scraped = pd.DataFrame(scraped_rows, columns=COLS)
        # Static takes priority (first in concat, kept by dedup)
        combined = pd.concat([df_static, df_scraped], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["date", "competition", "home_team", "away_team"],
            keep="first",
        )
        print(f"  IRFU scrape contributed {len(df_scraped)} rows (before dedup)")
    else:
        print("  IRFU scrape returned no results — using static fallback list")
        combined = df_static.copy()

    combined = combined.sort_values(["date", "kickoff_local"]).reset_index(drop=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"Saved {len(combined)} Ireland rugby fixtures → {out_path}")
    return combined


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("Building horse racing fixtures CSV ...")
    print("=" * 60)
    hr = build_horse_racing_csv()
    print(f"Horse racing: {len(hr)} fixture rows written\n")

    print("=" * 60)
    print("Building Ireland rugby fixtures CSV ...")
    print("=" * 60)
    rb = build_ireland_rugby_csv()
    print(f"Ireland rugby: {len(rb)} fixture rows written\n")

    sys.exit(0)
