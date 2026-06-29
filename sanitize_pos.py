"""
Sanitize a raw POS export from Des's till system into the clean hourly
format the forecasting pipeline expects.

Run this LOCALLY before committing anything. The raw input file is
.gitignored — only the aggregated output is safe to push.

Usage:
    python sanitize_pos.py --input data/raw/des_pos_export.csv
    python sanitize_pos.py --input data/raw/des_pos_export.csv --output data/synthetic/odonoghues_hourly.csv

The script:
  1. Detects which column holds the transaction timestamp
  2. Detects which column holds food/kitchen ticket flag (if any)
  3. Aggregates to one row per open hour: orders_count, food_tickets_count
  4. Drops every column that could identify a person or payment
  5. Prints a PII audit so you can confirm nothing slipped through
  6. Writes the clean hourly CSV — safe to commit

Supported raw formats (auto-detected):
  - Single timestamp column + one row per transaction
  - Pre-aggregated with a count column
  - Excel (.xlsx) or CSV (.csv)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Columns that are always PII — drop immediately, no questions asked
# ---------------------------------------------------------------------------
PII_PATTERNS = [
    "name", "staff", "employee", "operator", "cashier", "server",
    "customer", "member", "loyalty", "card", "pan", "last4", "last_4",
    "payment_ref", "transaction_id", "txn_id", "receipt", "auth_code",
    "email", "phone", "mobile", "address", "postcode", "eircode",
    "ip_", "device_id", "terminal_id",
]


def _is_pii_column(col: str) -> bool:
    c = col.lower().replace(" ", "_")
    return any(pat in c for pat in PII_PATTERNS)


def _load_raw(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
    else:
        # Try CSV as fallback
        df = pd.read_csv(path, low_memory=False)
    print(f"Loaded: {path.name}  →  {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def _detect_timestamp_col(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if any(
        kw in c.lower() for kw in ["time", "date", "datetime", "created", "opened", "ts"]
    )]
    if not candidates:
        raise ValueError(
            "Could not auto-detect a timestamp column. "
            f"Available columns: {list(df.columns)}\n"
            "Pass --timestamp-col <name> to specify it manually."
        )
    col = candidates[0]
    print(f"  Timestamp column: '{col}'")
    return col


def _detect_food_col(df: pd.DataFrame) -> str | None:
    candidates = [c for c in df.columns if any(
        kw in c.lower() for kw in ["food", "kitchen", "ticket", "meal", "cover"]
    )]
    if candidates:
        print(f"  Food/kitchen flag column: '{candidates[0]}'")
        return candidates[0]
    print("  No food/kitchen column found — food_tickets_count will be estimated "
          "as orders during food service hours (09:00–21:00).")
    return None


def _audit_pii(df: pd.DataFrame) -> list[str]:
    """Print and return list of PII columns detected."""
    found = [c for c in df.columns if _is_pii_column(c)]
    if found:
        print(f"\n  PII AUDIT — dropping {len(found)} column(s):")
        for c in found:
            print(f"    - {c}")
    else:
        print("  PII audit: no flagged columns found.")
    return found


def sanitize(
    input_path: Path,
    output_path: Path,
    timestamp_col: str | None = None,
    food_col: str | None = None,
) -> pd.DataFrame:
    df = _load_raw(input_path)

    # --- Step 1: Detect columns ---
    ts_col = timestamp_col or _detect_timestamp_col(df)
    food_flag_col = food_col or _detect_food_col(df)

    # --- Step 2: PII audit and drop ---
    pii_cols = _audit_pii(df)
    df = df.drop(columns=pii_cols, errors="ignore")

    # --- Step 3: Parse timestamp, floor to hour ---
    df["timestamp_hour"] = pd.to_datetime(df[ts_col], errors="coerce")
    bad_ts = df["timestamp_hour"].isna().sum()
    if bad_ts:
        print(f"  Warning: {bad_ts:,} rows had unparseable timestamps — dropped.")
    df = df.dropna(subset=["timestamp_hour"])
    df["timestamp_hour"] = df["timestamp_hour"].dt.floor("h")

    # Filter to pub open hours only (9am – 2am, i.e. hour 9–23 and 0–1)
    h = df["timestamp_hour"].dt.hour
    df = df[((h >= 9) & (h <= 23)) | (h <= 1)].copy()
    print(f"  After filtering to open hours: {len(df):,} rows")

    # --- Step 4: Aggregate to hourly ---
    if food_flag_col and food_flag_col in df.columns:
        # Treat truthy values in the food column as kitchen tickets
        df["_is_food"] = pd.to_numeric(
            df[food_flag_col].astype(str)
              .str.lower()
              .map({"true": 1, "yes": 1, "1": 1, "food": 1}),
            errors="coerce"
        ).fillna(0).astype(int)
    else:
        # Estimate: any transaction in food service hours (09:00–21:00) counts
        df["_is_food"] = ((df["timestamp_hour"].dt.hour >= 9) &
                          (df["timestamp_hour"].dt.hour < 21)).astype(int)

    hourly = (
        df.groupby("timestamp_hour")
          .agg(
              orders_count=("timestamp_hour", "count"),
              food_tickets_count=("_is_food", "sum"),
          )
          .reset_index()
    )

    # Ensure food tickets are zero outside food service hours
    h_out = hourly["timestamp_hour"].dt.hour
    hourly.loc[~((h_out >= 9) & (h_out < 21)), "food_tickets_count"] = 0
    hourly["food_tickets_count"] = hourly["food_tickets_count"].astype(int)

    print(f"\n  Aggregated to {len(hourly):,} hourly rows")
    print(f"  Date range: {hourly['timestamp_hour'].min()} → {hourly['timestamp_hour'].max()}")
    print(f"  Mean orders/hour: {hourly['orders_count'].mean():.1f}")
    print(f"  Mean food tickets/hour (food hours only): "
          f"{hourly.loc[hourly['food_tickets_count'] > 0, 'food_tickets_count'].mean():.1f}")

    # --- Step 5: Final PII check on output ---
    output_cols = set(hourly.columns)
    safe_cols = {"timestamp_hour", "orders_count", "food_tickets_count"}
    unexpected = output_cols - safe_cols
    if unexpected:
        print(f"\n  WARNING: unexpected columns in output: {unexpected} — dropping.")
        hourly = hourly[list(safe_cols)]

    # --- Step 6: Write clean output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output_path, index=False)
    print(f"\n  Saved clean hourly CSV → {output_path}")
    print("  This file is safe to commit. The raw input is .gitignored.")

    return hourly


def main():
    parser = argparse.ArgumentParser(
        description="Sanitize raw POS export into clean hourly demand data."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to raw POS export (.csv or .xlsx). Will NOT be committed (gitignored)."
    )
    parser.add_argument(
        "--output", "-o",
        default="data/synthetic/odonoghues_hourly.csv",
        help="Output path for clean hourly CSV. Default: data/synthetic/odonoghues_hourly.csv"
    )
    parser.add_argument(
        "--timestamp-col",
        help="Name of the timestamp column (auto-detected if omitted)."
    )
    parser.add_argument(
        "--food-col",
        help="Name of the column indicating a food/kitchen ticket (auto-detected if omitted)."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)

    print("=" * 60)
    print("  POS Data Sanitizer")
    print("=" * 60)

    hourly = sanitize(
        input_path=input_path,
        output_path=output_path,
        timestamp_col=args.timestamp_col,
        food_col=args.food_col,
    )

    print("\n" + "=" * 60)
    print("  Done. Sample output:")
    print("=" * 60)
    print(hourly.head(10).to_string(index=False))
    print("\nNext steps:")
    print("  python src/features.py          # rebuild feature table")
    print("  python src/model.py             # retrain models on real data")
    print("  python src/run_eda.py           # regenerate EDA charts")
    print("  git add data/ models/ && git commit -m 'retrain on real POS data' && git push")


if __name__ == "__main__":
    main()
