"""
Scrape hourly POS data from titanbi.net — Performance By Hour Detailed.

Usage:
    python scripts/scrape_titanbi.py                         # full backfill 2024-01-01 → today
    python scripts/scrape_titanbi.py --start 2025-06-01     # from a specific date
    python scripts/scrape_titanbi.py --date 2026-07-16      # single day
    python scripts/scrape_titanbi.py --resume               # skip dates already in output CSV

Prerequisites:
    pip install requests beautifulsoup4 browser-cookie3

Output:
    data/raw/pos_titanbi/hourly_YYYYMMDD_YYYYMMDD.csv
    Columns: date, hour, quantity, gross_eur
"""

import argparse
import csv
import re
import time
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

try:
    import browser_cookie3
    _HAS_BROWSER_COOKIE3 = True
except ImportError:
    _HAS_BROWSER_COOKIE3 = False

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "data/raw/pos_titanbi"

from titan_form import BASE_URL, CRITERIA_URL, STATIC_PAIRS
REPORT_URL = f"{BASE_URL}/Analysis/AnalyzePerformance/PerformanceByHourDetailed"
PRINT_URL = f"{REPORT_URL}?printView=yes"


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": REPORT_URL,
    })
    if _HAS_BROWSER_COOKIE3:
        try:
            cookies = browser_cookie3.chrome(domain_name="titanbi.net")
            session.cookies.update(cookies)
            print("  Loaded Chrome cookies for titanbi.net")
        except Exception as e:
            print(f"  WARNING: browser_cookie3 failed: {e}")
            print("  Make sure Chrome is closed, or export cookies manually.")
    else:
        print("  WARNING: browser_cookie3 not installed. Run: pip install browser-cookie3")
        print("  Continuing without auth — will likely get 302 redirects.")
    return session


def get_csrf_token(session: requests.Session) -> str:
    # The CSRF token lives inside the criteria dialog form, not the main page.
    # Fetch the form via the same GET the browser uses when clicking "Criteria".
    nonce = random.random()
    resp = session.get(f"{CRITERIA_URL}?_={nonce}", timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    token_input = soup.find("input", {"name": "__RequestVerificationToken"})
    if not token_input:
        raise ValueError("Could not find __RequestVerificationToken — are you logged in to titanbi.net in Chrome?")
    return token_input["value"]


def build_post_data(day: date, csrf_token: str) -> list[tuple[str, str]]:
    prev = day - timedelta(days=1)
    fmt = lambda d: d.strftime("%d/%m/%Y")
    date_fields = [
        ("Model.StartDate", fmt(day)),
        ("Model.EndDate", fmt(day)),
        ("Model.PreviousStartDate", fmt(prev)),
        ("Model.PreviousEndDate", fmt(prev)),
    ]
    return date_fields + list(STATIC_PAIRS) + [("__RequestVerificationToken", csrf_token)]


def set_criteria(session: requests.Session, day: date, csrf_token: str) -> bool:
    nonce = random.random()
    url = f"{CRITERIA_URL}?s={nonce}"
    data = build_post_data(day, csrf_token)
    resp = session.post(url, data=data, timeout=30, allow_redirects=True)
    if resp.status_code != 200:
        print(f"    POST criteria failed: {resp.status_code}")
        return False
    return True


def parse_hour_table(html: str, day: date) -> list[dict]:
    """Extract per-hour totals from the printView HTML.

    The page renders 58 single-row tables (one per 24 hours + PLU sub-tables).
    Hour summary tables are identified by the first cell having class _titan_number.
    That cell contains 3 <a> tags (show/expand/collapse); the show-in-output one
    holds the real hour number.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for t in soup.find_all("table"):
        trs = t.find_all("tr")
        if len(trs) != 1:
            continue
        cells = trs[0].find_all(["td", "th"])
        if not cells or "_titan_number" not in cells[0].get("class", []):
            continue
        a = cells[0].find("a", class_="show-in-output")
        if not a:
            continue
        try:
            hour = int(a.get_text(strip=True))
            qty_str = cells[2].get_text(strip=True).replace(",", "") if len(cells) > 2 else "0"
            gross_str = cells[3].get_text(strip=True).replace("€", "").replace(",", "") if len(cells) > 3 else "0"
            rows.append({
                "date": day.isoformat(),
                "hour": hour,
                "quantity": float(qty_str) if qty_str else 0.0,
                "gross_eur": float(gross_str) if gross_str else 0.0,
            })
        except (ValueError, IndexError):
            pass
    return rows


def scrape_day(session: requests.Session, day: date, csrf_token: str) -> list[dict] | None:
    if not set_criteria(session, day, csrf_token):
        return None
    time.sleep(0.5)
    resp = session.get(PRINT_URL, timeout=30)
    if resp.status_code != 200:
        print(f"    GET print view failed: {resp.status_code}")
        return None
    rows = parse_hour_table(resp.text, day)
    return rows


def date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def load_already_done(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(row["date"])
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--date", default=None, help="Single date YYYY-MM-DD")
    parser.add_argument("--resume", action="store_true", help="Skip dates already in output")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between requests")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.date:
        start_date = end_date = date.fromisoformat(args.date)
    else:
        start_date = date.fromisoformat(args.start)
        # Use 2 days ago when running before 10am — Titan BI serves the previous
        # session boundary (bar closes 2am) until at least ~8-9am, so scraping
        # "yesterday" before 10am returns stale session data from the night before.
        from datetime import datetime
        safe_lag = 2 if datetime.now().hour < 10 else 1
        end_date = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=safe_lag)

    out_path = OUT_DIR / f"hourly_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    print(f"Output: {out_path}")

    already_done = load_already_done(out_path) if args.resume else set()
    if already_done:
        print(f"Resuming — {len(already_done)} dates already done")

    session = make_session()

    print("Fetching CSRF token...")
    try:
        csrf_token = get_csrf_token(session)
        print(f"  Token OK ({csrf_token[:20]}...)")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    dates = list(date_range(start_date, end_date))
    total = len(dates)
    failed = []

    write_header = not out_path.exists() or not args.resume
    with open(out_path, "a" if args.resume else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "hour", "quantity", "gross_eur"])
        if write_header:
            writer.writeheader()

        for i, day in enumerate(dates, 1):
            day_str = day.isoformat()
            if day_str in already_done:
                continue

            print(f"[{i}/{total}] {day_str}", end="  ", flush=True)
            try:
                rows = scrape_day(session, day, csrf_token)
                if rows is None:
                    # CSRF may have expired — refresh token and retry once
                    print("retrying with fresh token...", end=" ", flush=True)
                    csrf_token = get_csrf_token(session)
                    rows = scrape_day(session, day, csrf_token)

                if rows:
                    writer.writerows(rows)
                    f.flush()
                    total_qty = sum(r["quantity"] for r in rows)
                    print(f"OK ({len(rows)} hours, {total_qty:.0f} items)")
                else:
                    print("EMPTY (possible closed day)")
                    # Write a placeholder row so --resume doesn't retry it
                    writer.writerow({"date": day_str, "hour": -1, "quantity": 0, "gross_eur": 0})
                    f.flush()

            except KeyboardInterrupt:
                print("\nInterrupted — output saved so far.")
                break
            except Exception as e:
                print(f"ERROR: {e}")
                failed.append(day_str)

            time.sleep(args.delay + random.uniform(0, 0.5))

    print(f"\nDone. Output: {out_path}")
    if failed:
        print(f"Failed dates ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
