"""
Daily data refresh — run once per day (e.g., 6am cron) to pull live signals.

Steps:
  1. Fetch weather (historical backfill + 7-day forecast)
  2. Fetch DCC pedestrian footfall counters
  3. Rebuild enriched feature table
  4. (Optional) Retrain models if new POS data is present

Usage:
    python refresh_data.py
    python refresh_data.py --retrain   # also retrain XGBoost
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def step(label: str):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain XGBoost models after refreshing data")
    args = parser.parse_args()

    t0 = time.time()

    # Load .env so FOOTBALL_DATA_API_KEY is available for fetch_sports
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # 1. Weather + enrichment (includes airport, cruise, events, holidays)
    step("1/3  Fetching weather, airport, cruise, events…")
    from src.fetch_public_data import build_enrichment_table, build_hourly_weather_table
    
    wx = build_hourly_weather_table(
        start="2023-01-01",
        out_path=ROOT / "data/raw/weather_hourly.csv",
    )
    print(f"     Hourly weather: {wx.shape[0]} rows, {wx['timestamp_hour'].min()} → {wx['timestamp_hour'].max()}")
    
    enrich = build_enrichment_table(start="2023-01-01")
    print(f"     Enrichment table: {enrich.shape[0]} rows × {enrich.shape[1]} cols")
    enrich.to_csv(ROOT / "data/raw/enrichment.csv", index=False)
    print("     Saved → data/raw/enrichment.csv")

    # 1b. Sports fixtures
    step("1b/3  Fetching sports fixtures (football-data.org + Six Nations static)…")
    try:
        from src.fetch_sports import fetch_sports_fixtures, build_six_nations_csv
        from datetime import date as _date, timedelta as _timedelta
        sn = build_six_nations_csv(out_path=ROOT / "data/raw/six_nations_fixtures.csv")
        print(f"     Six Nations: {len(sn)} fixtures saved")
        sports_end = (_date.today() + _timedelta(days=90)).strftime("%Y-%m-%d")
        fixtures = fetch_sports_fixtures(
            date_from="2023-01-01",
            date_to=sports_end,
            out_path=ROOT / "data/raw/sports_fixtures.csv",
        )
        if not fixtures.empty:
            print(f"     Football fixtures: {len(fixtures)} total")
            counts = fixtures.groupby("competition").size()
            for comp, n in counts.items():
                print(f"       {comp}: {n}")
        else:
            print("     No football fixtures fetched (FOOTBALL_DATA_API_KEY not set?)")
    except Exception as exc:
        print(f"     WARNING: Sports fetch failed: {exc} — continuing without sports data.")

    # 1c. Horse racing + Ireland rugby fixtures
    step("1c/3  Building horse racing and Ireland rugby fixtures…")
    try:
        from src.fetch_horse_racing import build_horse_racing_csv, build_ireland_rugby_csv
        hr = build_horse_racing_csv(out_path=ROOT / "data/raw/horse_racing_fixtures.csv")
        print(f"     Horse racing: {len(hr)} fixtures saved")
        rb = build_ireland_rugby_csv(out_path=ROOT / "data/raw/ireland_rugby_fixtures.csv")
        print(f"     Ireland rugby: {len(rb)} fixtures saved")
    except Exception as exc:
        print(f"     WARNING: Horse racing/rugby fixtures failed: {exc} — continuing.")

    # 2. DCC footfall counters
    step("2/3  Fetching DCC pedestrian footfall counters…")
    from src.fetch_footfall import fetch_footfall, fetch_failte_events
    ff = fetch_footfall(start="2023-01-01", end=None, force_refresh_current_year=True)
    print(f"     Footfall: {ff.shape[0]} rows × {ff.shape[1]} cols")
    ff.to_csv(ROOT / "data/raw/footfall_hourly.csv", index=False)
    print("     Saved → data/raw/footfall_hourly.csv")

    from datetime import date
    today = date.today().isoformat()
    failte = fetch_failte_events(start=today, end=None)
    if failte is not None:
        failte.to_csv(ROOT / "data/raw/failte_events.csv", index=False)
        print(f"     Fáilte events: {len(failte)} rows saved → data/raw/failte_events.csv")

    # 3. Rebuild feature table
    step("3/3  Rebuilding feature table…")
    import pandas as pd
    from src.features import build_features

    raw_path = ROOT / "data/synthetic/odonoghues_hourly.csv"
    raw = pd.read_csv(raw_path)
    df = build_features(raw)
    df.to_parquet(ROOT / "data/processed/features.parquet", index=False)
    print(f"     Features: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print("     Saved → data/processed/features.parquet")

    # 4. Optional retrain
    if args.retrain:
        step("4/4  Retraining XGBoost models…")
        from src.model import main as train_main
        train_main()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Refresh complete in {elapsed:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
