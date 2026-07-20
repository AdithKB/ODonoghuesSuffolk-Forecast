"""
Ingest real POS data from Titan system Z-read exports and produce
hourly demand estimates for model training.

Data sources:
  data/pubdata/resultsBTR.csv — daily Z-read per terminal (gitignored)
  Hourly distribution: extracted from Titan2.pdf "Performance By Hour"

Output:
  data/synthetic/odonoghues_hourly.csv — replaces synthetic baseline

Methodology:
  Daily transaction totals (summed across terminals) are disaggregated
  into hourly estimates using real revenue distributions from Titan2.pdf,
  segmented by day-of-week.  Food ticket counts use the product-level
  daily average (~82/day) scaled by a kitchen-hours distribution.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Hourly item quantities — extracted from Titan6.pdf
# "Performance By Hour Detailed", Jan 1 2024 → Jul 20 2026, all terminals
# Full 2.5-year period; used to compute overall hourly shape fractions.
# ---------------------------------------------------------------------------
_HOURLY_ITEMS_FULL = {
     0: 25149,  1: 3710,   2: 254,   3: 104,   4: 35,
     5: 23,     6: 8,      7: 0,     8: 1,
     9: 2937,  10: 16826, 11: 32324,
    12: 54227, 13: 64717, 14: 54965, 15: 50417,
    16: 55849, 17: 66499, 18: 71802, 19: 69459,
    20: 62592, 21: 55844, 22: 68430, 23: 48967,
}
_TOTAL_ITEMS = sum(_HOURLY_ITEMS_FULL.values())
_HOURLY_FRACS_FULL = {h: v / _TOTAL_ITEMS for h, v in _HOURLY_ITEMS_FULL.items()}

# ---------------------------------------------------------------------------
# Hourly revenue by day-of-week — extracted from Titan2.pdf
# "Performance By Hour" report, Jan 1 2024 → Jul 1 2024, all terminals
# Keys: hour (int), Values: [Mon, Tue, Wed, Thu, Fri, Sat, Sun] revenue (€)
# Weekday index matches pandas .dt.weekday (0=Monday, 6=Sunday)
# Used to compute per-DOW shape; scaled to match full-period hourly fractions.
# ---------------------------------------------------------------------------
_DOW_HOURLY_REVENUE = {
    #  hour:  [Mon,      Tue,      Wed,      Thu,       Fri,       Sat,       Sun     ]
     9:       [2071.85,  1821.10,  1063.50,  1448.20,   4827.75,   5531.45,   3825.35],
    10:       [2071.85,  1821.10,  1063.50,  1448.20,   4827.75,   5531.45,   3825.35],
    11:       [6195.60,  4886.70,  3440.70,  3453.35,   9607.65,  11189.70,  11798.15],
    12:       [10787.85, 7222.90,  7066.85,  7019.70,  11787.85,  16243.45,  15481.70],
    13:       [11530.25, 8280.35,  8477.75,  9201.20,  13033.10,  18949.80,  15888.10],
    14:       [9806.40,  7238.40,  6513.75,  8601.05,  12497.20,  17759.15,  15045.00],
    15:       [8320.55,  5247.45,  4903.60,  5238.10,   9568.20,  17516.75,  12964.55],
    16:       [8772.90,  6276.15,  5241.30,  8779.40,  12348.20,  18238.05,  13342.05],
    17:       [7771.55,  7461.95,  7667.50,  11500.65, 14517.50,  20294.75,  14429.70],
    18:       [9364.45,  9128.15,  9324.40,  14035.70, 16748.00,  20229.80,  12657.35],
    19:       [8218.95,  8571.10,  8121.25,  12768.65, 18308.05,  22829.50,  11624.55],
    20:       [6810.25,  6099.70,  7970.35,  12494.30, 15692.45,  21091.40,   9085.60],
    21:       [4077.60,  5119.55,  5629.65,   8917.10, 13152.75,  18584.95,   8125.25],
    22:       [4938.20,  6181.00,  5274.90,  10622.85, 18748.55,  24178.20,   9187.45],
    23:       [1771.75,  2032.15,  2881.20,   4119.60, 15950.30,  21745.25,   7523.65],
     0:       [309.20,    175.00,    52.25,     37.35,  9483.60,  16842.80,   1573.20],
     1:       [90.00,     174.00,     0.00,      0.00,   840.25,   2167.70,     27.50],
     2:       [48.00,      72.00,     0.00,      0.00,     0.00,     78.30,      0.00],
     3:       [10.00,       0.00,     0.00,      0.00,     0.00,     80.00,     14.15],
     4:       [0.00,        0.00,     0.00,      0.00,     0.00,     35.00,      0.00],
     5:       [0.00,        0.00,     0.00,      0.00,     0.00,     23.00,      0.00],
     6:       [0.00,        0.00,     0.00,      0.00,     0.00,      8.00,      0.00],
}

# Hours that belong to the "late night" portion of the previous business day
_LATE_HOURS = {0, 1, 2}

# Pub operating hours (business-day perspective)
_PUB_HOURS = list(range(9, 24)) + [0, 1, 2]

# ---------------------------------------------------------------------------
# Food ticket daily rate and kitchen-hours distribution
# Derived from Titan5.pdf (PLU detail): ~14,787 main food orders over 182 days
# Kitchen hours only: 12-20; shape follows lunch/dinner revenue peaks
# ---------------------------------------------------------------------------
_FOOD_PER_DAY = 14787 / 182  # ≈ 81.2 food tickets/day average

_FOOD_HOUR_WEIGHTS = {
    12: 1.20, 13: 1.50, 14: 1.30, 15: 0.80,
    16: 0.40, 17: 1.05, 18: 1.40, 19: 1.25, 20: 0.90,
}
_FOOD_TOTAL_WEIGHT = sum(_FOOD_HOUR_WEIGHTS.values())
_FOOD_HOUR_FRACS = {h: w / _FOOD_TOTAL_WEIGHT for h, w in _FOOD_HOUR_WEIGHTS.items()}


def _build_dow_hour_fracs() -> dict[int, dict[int, float]]:
    """
    For each day-of-week (0=Mon), compute fraction of daily revenue
    that falls in each hour.  Returns {dow: {hour: fraction}}.
    """
    fracs: dict[int, dict[int, float]] = {}
    for dow in range(7):
        revenues = {h: vals[dow] for h, vals in _DOW_HOURLY_REVENUE.items()}
        total = sum(revenues.values())
        if total == 0:
            fracs[dow] = {h: 0.0 for h in revenues}
        else:
            fracs[dow] = {h: v / total for h, v in revenues.items()}
    return fracs


_DOW_HOUR_FRACS = _build_dow_hour_fracs()


def disaggregate_daily(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily orders_count totals into hourly estimates.

    Args:
        daily_df: columns [date, orders_count, revenue]

    Returns:
        Hourly DataFrame with columns:
          timestamp_hour, orders_count, food_tickets_count, sales_total
    """
    avg_daily_orders = daily_df["orders_count"].replace(0, np.nan).mean()
    rows = []

    for _, day in daily_df.iterrows():
        date = pd.Timestamp(day["date"])
        dow = date.weekday()            # 0=Mon … 6=Sun
        daily_orders = float(day["orders_count"])
        daily_revenue = float(day["revenue"])

        if daily_orders == 0:
            continue

        hour_fracs = _DOW_HOUR_FRACS[dow]

        # Scale food tickets by day intensity relative to the average day
        intensity = daily_orders / avg_daily_orders
        daily_food = _FOOD_PER_DAY * intensity

        for hour in _PUB_HOURS:
            frac = hour_fracs.get(hour, 0.0)
            if frac == 0.0:
                continue

            hourly_orders = max(0, round(daily_orders * frac))
            hourly_revenue = daily_revenue * frac
            food_frac = _FOOD_HOUR_FRACS.get(hour, 0.0)
            hourly_food = max(0, round(daily_food * food_frac))

            # Late-night hours belong to the next calendar date
            if hour in _LATE_HOURS:
                ts = date + pd.Timedelta(days=1, hours=hour)
            else:
                ts = date + pd.Timedelta(hours=hour)

            rows.append({
                "timestamp_hour":     ts,
                "orders_count":       hourly_orders,
                "food_tickets_count": hourly_food,
                "sales_total":        round(hourly_revenue, 2),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp_hour").drop_duplicates("timestamp_hour").reset_index(drop=True)
    return df


def load_pos_daily(pos_path: Path) -> pd.DataFrame:
    """
    Aggregate per-terminal Z-read CSV into daily pub-level totals.
    """
    raw = pd.read_csv(pos_path, skiprows=1)
    raw["date"] = pd.to_datetime(raw["TimeStamp"]).dt.date
    daily = (
        raw.groupby("date")
        .agg(
            orders_count=("TransactionsCount", "sum"),
            revenue=("TotalTurnover", "sum"),
            first_open=("FirstTransactionTime", "min"),
            last_close=("LastTransactionTime", "max"),
        )
        .reset_index()
    )
    return daily


def _pad_synthetic_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add zero-fill columns expected by build_features() / synthetic schema.
    External signals will be overwritten by the feature-engineering pipeline.
    """
    bool_cols = [
        "live_music_flag", "special_event_flag", "promo_flag",
        "stockout_flag", "menu_change_flag", "bank_holiday_flag",
        "school_holiday_flag", "payday_period_flag",
        "major_sports_event_flag", "city_event_flag", "st_patricks_week_flag",
        "cruise_ship_flag", "weather_severity_flag",
    ]
    zero_cols = [
        "bar_staff_count", "kitchen_staff_count",
        "temp_c", "rain_mm", "wind_speed_kmh",
        "airport_arrivals", "airport_arrivals_lag1",
        "ships_in_port_count", "cruise_passenger_estimate",
    ]
    ts = pd.to_datetime(df["timestamp_hour"])
    df["hour"]        = ts.dt.hour
    df["weekday"]     = ts.dt.weekday
    df["month"]       = ts.dt.month
    df["is_weekend"]  = ts.dt.weekday.isin([5, 6])
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["covers_count"] = df["food_tickets_count"]
    df["busy_label"]   = "unknown"

    for c in bool_cols:
        df[c] = False
    for c in zero_cols:
        df[c] = 0.0

    return df


def main() -> None:
    pos_path = Path("data/pubdata/resultsBTR.csv")
    out_path = Path("data/synthetic/odonoghues_hourly.csv")

    print(f"Loading POS data from {pos_path} ...")
    daily = load_pos_daily(pos_path)
    print(f"  {len(daily)} days  |  {daily['orders_count'].sum():,.0f} total transactions")
    print(f"  Date range: {daily['date'].min()} → {daily['date'].max()}")
    print(f"  Avg orders/day: {daily['orders_count'].mean():.0f}")

    print("\nDisaggregating to hourly ...")
    hourly = disaggregate_daily(daily)
    hourly = _pad_synthetic_columns(hourly)

    print(f"  {len(hourly):,} hourly rows generated")
    print(f"  orders_count:       {hourly['orders_count'].min()}–{hourly['orders_count'].max()}  (mean {hourly['orders_count'].mean():.1f})")
    print(f"  food_tickets_count: {hourly['food_tickets_count'].min()}–{hourly['food_tickets_count'].max()}  (mean {hourly['food_tickets_count'].mean():.1f})")

    hourly.to_csv(out_path, index=False)
    print(f"\n✓ Saved → {out_path}")
    print("Next: python src/features.py && python src/model.py")


if __name__ == "__main__":
    main()
