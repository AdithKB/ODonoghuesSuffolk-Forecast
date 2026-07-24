"""
Load TitanBI scraper output and produce a model-ready hourly dataframe.

Reads:
  data/raw/pos_titanbi/hourly_*.csv    — actual per-hour item quantities
  data/raw/pos_titanbi/daily_txn_*.csv — daily transaction counts (optional)

Outputs DataFrame with:
  timestamp_hour    (datetime)
  orders_count      (real hourly items sold — Titan "quantity")
  food_tickets_count (derived: daily avg × kitchen-hour fraction, scaled by day intensity)
  gross_eur         (hourly revenue from Titan)
  transactions      (daily total transactions, broadcast to all hours of that day)
  basket_size       (items / transactions per day)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Kitchen-hours food fraction — calibrated from Titan5.pdf
# ~14,787 main food orders over 182 days → 81.2 food tickets/day average.
# Distributed by hour: kitchen open 12-20, weights match lunch/dinner peaks.
# ---------------------------------------------------------------------------
_FOOD_PER_DAY = 14787 / 182   # ≈ 81.2 food tickets/day

_FOOD_HOUR_WEIGHTS = {
    12: 1.20, 13: 1.50, 14: 1.30, 15: 0.80,
    16: 0.40, 17: 1.05, 18: 1.40, 19: 1.25, 20: 0.90,
}
_FOOD_TOTAL_WEIGHT = sum(_FOOD_HOUR_WEIGHTS.values())
_FOOD_HOUR_FRACS = {h: w / _FOOD_TOTAL_WEIGHT for h, w in _FOOD_HOUR_WEIGHTS.items()}


def load_titan_hourly(dir_path: Path | str = "data/raw/pos_titanbi") -> pd.DataFrame:
    """
    Load and deduplicate all hourly_*.csv files.

    Returns DataFrame with columns: date (str), hour (int), quantity (float), gross_eur (float)
    """
    dir_path = Path(dir_path)
    files = sorted(dir_path.glob("hourly_*.csv"))
    if not files:
        raise FileNotFoundError(f"No hourly_*.csv files found in {dir_path}")

    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")

    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize().dt.strftime("%Y-%m-%d")
    raw["hour"] = pd.to_numeric(raw["hour"], errors="coerce")
    raw = raw.dropna(subset=["date", "hour"])
    raw["hour"] = raw["hour"].astype(int)
    # Dedup: keep last (so if a day appears in both the test file and backfill, backfill wins)
    raw = raw.sort_values(["date", "hour"]).drop_duplicates(subset=["date", "hour"], keep="last")
    return raw.reset_index(drop=True)


def load_titan_txn(dir_path: Path | str = "data/raw/pos_titanbi") -> pd.DataFrame:
    """
    Load and deduplicate all daily_txn_*.csv files.

    Returns DataFrame with columns: date (str), transactions (int), gross_eur (float)
    """
    dir_path = Path(dir_path)
    files = sorted(dir_path.glob("daily_txn_*.csv"))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")

    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize().dt.strftime("%Y-%m-%d")
    raw = raw.drop_duplicates(subset=["date"], keep="last")
    for col in ["transactions", "gross_eur", "avg_transaction_eur"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return raw[["date"] + [c for c in ["transactions", "gross_eur", "avg_transaction_eur"] if c in raw.columns]].reset_index(drop=True)


def build_titan_dataset(
    dir_path: Path | str = "data/raw/pos_titanbi",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Merge hourly quantity + daily transaction data into a model-ready DataFrame.

    Returns DataFrame with columns:
      timestamp_hour, orders_count, food_tickets_count, gross_eur,
      transactions, basket_size

    orders_count        = Titan hourly "quantity" (real items sold per hour)
    food_tickets_count  = _FOOD_PER_DAY × day_intensity × kitchen_hour_fraction
                          (day_intensity = that day's total items / avg daily items)
    basket_size         = daily_items / daily_transactions
    """
    dir_path = Path(dir_path)

    hourly = load_titan_hourly(dir_path)
    txn    = load_titan_txn(dir_path)

    if verbose:
        print(f"  Hourly rows loaded:  {len(hourly):,}  "
              f"({hourly['date'].min()} → {hourly['date'].max()})")
        if not txn.empty:
            print(f"  Daily txn rows:      {len(txn):,}  "
                  f"({txn['date'].min()} → {txn['date'].max()})")

    # Build timestamp_hour: hour 0 = midnight on that calendar date (same date, 00:00)
    hourly["timestamp_hour"] = (
        pd.to_datetime(hourly["date"]) + pd.to_timedelta(hourly["hour"], unit="h")
    )

    df = hourly.rename(columns={"quantity": "orders_count"}).copy()

    # ── Daily totals ──────────────────────────────────────────────────────────
    daily_qty = (
        df.groupby("date")["orders_count"].sum()
        .rename("daily_qty")
        .reset_index()
    )
    df = df.merge(daily_qty, on="date", how="left")

    # ── Merge transactions (left join — txn data may lag behind hourly) ───────
    if not txn.empty:
        df = df.merge(
            txn[["date", "transactions"]],
            on="date", how="left"
        )
        df["basket_size"] = (
            df["daily_qty"] / df["transactions"].replace(0, np.nan)
        ).fillna(0)
    else:
        df["transactions"] = np.nan
        df["basket_size"]  = np.nan

    # ── Derive food_tickets_count ─────────────────────────────────────────────
    # Scale _FOOD_PER_DAY by each day's intensity relative to the average day,
    # then distribute by kitchen hour fraction.
    avg_daily_qty = max(df["daily_qty"].mean(), 1.0)
    df["_day_intensity"] = df["daily_qty"] / avg_daily_qty

    df["food_tickets_count"] = (
        df["hour"].map(_FOOD_HOUR_FRACS).fillna(0.0) * df["_day_intensity"] * _FOOD_PER_DAY
    ).round(1)

    # ── Final column selection ────────────────────────────────────────────────
    keep = ["timestamp_hour", "orders_count", "food_tickets_count",
            "gross_eur", "transactions", "basket_size"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.sort_values("timestamp_hour").reset_index(drop=True)

    if verbose:
        days = df["timestamp_hour"].dt.date.nunique()
        avg_items = df.groupby(df["timestamp_hour"].dt.date)["orders_count"].sum().mean()
        avg_food  = df.groupby(df["timestamp_hour"].dt.date)["food_tickets_count"].sum().mean()
        print(f"  Dataset: {len(df):,} rows | {days} days | "
              f"avg {avg_items:.0f} items/day | {avg_food:.0f} food tickets/day")

    return df


if __name__ == "__main__":
    df = build_titan_dataset(verbose=True)
    print(df[df["orders_count"] > 0].head(8).to_string(index=False))
    print(f"\nDate range: {df['timestamp_hour'].min()} → {df['timestamp_hour'].max()}")
    print(f"orders_count:       min={df['orders_count'].min():.0f}  "
          f"max={df['orders_count'].max():.0f}  mean={df['orders_count'].mean():.1f}")
    print(f"food_tickets_count: daily avg ≈ "
          f"{df.groupby(df['timestamp_hour'].dt.date)['food_tickets_count'].sum().mean():.1f}")
    if df["basket_size"].gt(0).any():
        bs = df["basket_size"].replace(0, np.nan).dropna()
        print(f"basket_size:        min={bs.min():.2f}  max={bs.max():.2f}  mean={bs.mean():.2f}")
