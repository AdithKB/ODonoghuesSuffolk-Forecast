"""
Scrape daily transaction summary from titanbi.net — Performance By Transaction Report.

Outputs one row per day:
  date, transactions, gross_eur, avg_transaction_eur, cash_count, card_count,
  first_transaction, last_transaction

Run AFTER (not during) the hourly backfill — both use the same session and
the same CriteriaPerformanceByHour POST endpoint (same server-side session state).

Usage:
    python scripts/scrape_titanbi_txn.py                        # 2024-01-01 → yesterday
    python scripts/scrape_titanbi_txn.py --start 2025-01-01
    python scripts/scrape_titanbi_txn.py --date 2026-07-16      # single day test
    python scripts/scrape_titanbi_txn.py --resume               # skip done dates

Prerequisites (same as scrape_titanbi.py):
    pip install requests beautifulsoup4 browser-cookie3
"""

import argparse
import csv
import random
import re
import sys
import time
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
TXN_REPORT_URL = f"{BASE_URL}/Analysis/AnalyzePerformance/TransactionReport"
PRINT_URL = f"{TXN_REPORT_URL}?printView=yes"

FIELDS = ["date", "transactions", "gross_eur", "avg_transaction_eur",
          "cash_count", "card_count", "first_transaction", "last_transaction"]


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": TXN_REPORT_URL,
    })
    if _HAS_BROWSER_COOKIE3:
        try:
            s.cookies.update(browser_cookie3.chrome(domain_name="titanbi.net"))
            print("  Loaded Chrome cookies for titanbi.net")
        except Exception as e:
            print(f"  WARNING: browser_cookie3 failed: {e}")
    else:
        print("  WARNING: browser_cookie3 not installed. Run: pip install browser-cookie3")
    return s


def get_csrf_token(session: requests.Session) -> str:
    r = session.get(f"{CRITERIA_URL}?_={random.random()}", timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    inp = soup.find("input", {"name": "__RequestVerificationToken"})
    if not inp:
        raise ValueError("CSRF token not found — are you logged in to titanbi.net in Chrome?")
    return inp["value"]


def set_criteria(session: requests.Session, day: date, token: str) -> bool:
    prev = day - timedelta(days=1)
    fmt = lambda d: d.strftime("%d/%m/%Y")
    data = [
        ("Model.StartDate", fmt(day)),
        ("Model.EndDate", fmt(day)),
        ("Model.PreviousStartDate", fmt(prev)),
        ("Model.PreviousEndDate", fmt(prev)),
    ] + list(STATIC_PAIRS) + [("__RequestVerificationToken", token)]
    r = session.post(f"{CRITERIA_URL}?s={random.random()}", data=data, timeout=30)
    return r.status_code == 200


def _num(s: str) -> float:
    return float(s.replace("€", "").replace(",", "").strip()) if s.strip() else 0.0


def parse_txn_report(html: str, day: date) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    lines = [l for l in text.split("\n") if l.strip()]

    result = {
        "date": day.isoformat(),
        "transactions": None,
        "gross_eur": None,
        "avg_transaction_eur": None,
        "cash_count": None,
        "card_count": None,
        "first_transaction": None,
        "last_transaction": None,
    }

    for i, line in enumerate(lines):
        # "Transaction\n275 / €6,008.15"
        if line == "Transaction" and i + 1 < len(lines):
            m = re.match(r"(\d+)\s*/\s*€([\d,]+\.[\d]+)", lines[i + 1])
            if m:
                result["transactions"] = int(m.group(1))
                result["gross_eur"] = _num(m.group(2))

        # "Average\n€21.85"
        elif line == "Average" and i + 1 < len(lines):
            val = lines[i + 1]
            if val.startswith("€"):
                result["avg_transaction_eur"] = _num(val)

        # "First\n16/07/2026 09:31"
        elif line == "First" and i + 1 < len(lines):
            result["first_transaction"] = lines[i + 1]

        # "Last\n17/07/2026 00:18"
        elif line == "Last" and i + 1 < len(lines):
            result["last_transaction"] = lines[i + 1]

        # Tender table rows: "CASH\n129\n€2,916.65\n..."
        elif line == "CASH" and i + 1 < len(lines):
            try:
                result["cash_count"] = int(lines[i + 1])
            except ValueError:
                pass

        elif line == "CARD" and i + 1 < len(lines):
            try:
                result["card_count"] = int(lines[i + 1])
            except ValueError:
                pass

    # Gross fallback from "Gross\n€6,008.15" if transaction line didn't capture it
    if result["gross_eur"] is None:
        for i, line in enumerate(lines):
            if line == "Gross" and i + 1 < len(lines):
                val = lines[i + 1]
                if val.startswith("€"):
                    result["gross_eur"] = _num(val)
                    break

    return result if result["transactions"] is not None else None


def scrape_day(session: requests.Session, day: date, token: str) -> dict | None:
    if not set_criteria(session, day, token):
        return None
    time.sleep(0.4)
    r = session.get(PRINT_URL, timeout=30)
    if r.status_code != 200:
        return None
    return parse_txn_report(r.text, day)


def date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path) as f:
        return {row["date"] for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--delay", type=float, default=1.5)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.date:
        start_date = end_date = date.fromisoformat(args.date)
    else:
        start_date = date.fromisoformat(args.start)
        end_date = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)

    out_path = OUT_DIR / f"daily_txn_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    print(f"Output: {out_path}")

    done = load_done(out_path) if args.resume else set()
    if done:
        print(f"Resuming — {len(done)} dates already done")

    session = make_session()
    print("Fetching CSRF token...")
    try:
        token = get_csrf_token(session)
        print(f"  Token OK ({token[:20]}...)")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    dates = list(date_range(start_date, end_date))
    total = len(dates)
    failed = []

    with open(out_path, "a" if args.resume else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not args.resume:
            writer.writeheader()

        for i, day in enumerate(dates, 1):
            if day.isoformat() in done:
                continue

            print(f"[{i}/{total}] {day.isoformat()}", end="  ", flush=True)
            try:
                row = scrape_day(session, day, token)
                if row is None:
                    print("retrying with fresh token...", end=" ", flush=True)
                    token = get_csrf_token(session)
                    row = scrape_day(session, day, token)

                if row:
                    writer.writerow(row)
                    f.flush()
                    print(f"OK  txn={row['transactions']}  avg=€{row['avg_transaction_eur']}  gross=€{row['gross_eur']}")
                else:
                    print("EMPTY")
                    writer.writerow({"date": day.isoformat(), **{k: None for k in FIELDS if k != "date"}})
                    f.flush()

            except KeyboardInterrupt:
                print("\nInterrupted — output saved.")
                break
            except Exception as e:
                print(f"ERROR: {e}")
                failed.append(day.isoformat())

            time.sleep(args.delay + random.uniform(0, 0.5))

    print(f"\nDone. Output: {out_path}")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
