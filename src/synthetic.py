"""
Synthetic hourly demand generator for O'Donoghues Suffolk Street.
Calibrated to real venue schedule, Dublin tourism patterns, and Irish calendar.

Output schema matches config/schema.yaml so real POS data can replace it without
touching any downstream code.
"""

import numpy as np
import pandas as pd
import holidays
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Venue constants (from odonoghuesbars.ie, verified June 2026)
# ---------------------------------------------------------------------------
FOOD_OPEN = 9    # food service starts
FOOD_CLOSE = 21  # food service ends (9pm)
BREAKFAST_END = 15  # breakfast menu until 3pm

# Bar closing: weekends 2am, weekdays 12:30am — expressed as hour-of-day
# (hours >= BAR_CLOSE_WEEKDAY on Mon-Thu, >= BAR_CLOSE_WEEKEND on Fri-Sat)
BAR_CLOSE_WEEKDAY = 25  # 1am (hour 25 = 1am next day in 24h+ convention)
BAR_CLOSE_WEEKEND = 26  # 2am

# Live music windows (hour ranges, inclusive start)
MUSIC_START_WEEKDAY = 22   # 9:30pm rounds to 22 for hourly bins
MUSIC_END_WEEKDAY = 24     # 11:30pm
MUSIC_START_WEEKEND = 22   # 10pm
MUSIC_END_WEEKEND = 26     # 1am

# ---------------------------------------------------------------------------
# Base hourly demand curve (orders_count, average Tuesday in peak season)
# ---------------------------------------------------------------------------
BASE_ORDERS = {
    9: 8,   # opening, breakfast
    10: 12,
    11: 20,
    12: 42,  # lunch builds
    13: 55,  # lunch peak
    14: 48,
    15: 28,  # afternoon lull
    16: 22,
    17: 30,  # after-work begins
    18: 42,
    19: 52,  # dinner peak
    20: 46,
    21: 35,  # food closes, bar-only orders drop
    22: 40,  # live music starts — bar orders spike
    23: 38,
    0: 25,   # post-midnight (weekdays only open this late Thu+)
    1: 12,
}

# Food tickets are a fraction of orders (not all orders are food)
FOOD_TICKET_RATIO = {
    9: 0.70,   # breakfast-heavy
    10: 0.65,
    11: 0.60,
    12: 0.75,  # lunch — mostly food
    13: 0.78,
    14: 0.72,
    15: 0.60,
    16: 0.45,
    17: 0.50,
    18: 0.65,
    19: 0.72,
    20: 0.68,
    21: 0.0,   # food closes
    22: 0.0,
    23: 0.0,
    0: 0.0,
    1: 0.0,
}

# ---------------------------------------------------------------------------
# Multipliers
# ---------------------------------------------------------------------------
WEEKDAY_MULT = {0: 0.65, 1: 0.70, 2: 0.80, 3: 0.90, 4: 1.15, 5: 1.35, 6: 1.00}

MONTH_MULT = {
    1: 0.68, 2: 0.72, 3: 1.20,  # March: St. Patrick's boost
    4: 0.90, 5: 0.95, 6: 1.10,
    7: 1.15, 8: 1.10, 9: 0.95,
    10: 0.90, 11: 0.80, 12: 1.15,
}

def _weather_mult(temp_c, rain_mm):
    mult = 1.0
    if rain_mm > 5:
        mult *= 0.90   # heavy rain suppresses lunch walk-ins
    elif rain_mm > 2:
        mult *= 0.95
    if temp_c < 5:
        mult *= 1.05   # cold drives people inside
    elif temp_c > 18:
        mult *= 1.08   # nice weather boosts footfall in tourist zone
    return mult

def _event_mult(row):
    mult = 1.0
    if row.get("st_patricks_week_flag"):
        mult *= 2.0
    elif row.get("city_event_flag"):
        mult *= 1.20
    if row.get("major_sports_event_flag"):
        mult *= 1.35
    if row.get("bank_holiday_flag"):
        mult *= 1.30
    if row.get("cruise_ship_flag"):
        mult *= 1.10
    if row.get("special_event_flag"):
        mult *= 1.25
        
    # Extra multipliers from failte events
    failte_count = row.get("failte_event_count", 0)
    if failte_count > 10:
        mult *= 1.15
    elif failte_count > 3:
        mult *= 1.05
        
    return mult

def _footfall_mult(footfall_count):
    if pd.isna(footfall_count) or footfall_count == 0:
        return 1.0
    # Approximate: normal footfall ~ 1000, high > 3000
    if footfall_count > 3000:
        return 1.35
    if footfall_count > 2000:
        return 1.20
    if footfall_count > 1000:
        return 1.10
    return 1.0

def _live_music_mult(hour, weekday, live_music_flag):
    if not live_music_flag:
        return 1.0
    if hour < 21:
        return 1.0  # music not started yet
    is_weekend = weekday in (4, 5)
    start = MUSIC_START_WEEKEND if is_weekend else MUSIC_START_WEEKDAY
    end = MUSIC_END_WEEKEND if is_weekend else MUSIC_END_WEEKDAY
    if start <= hour <= end:
        return 1.40  # significant bar spike during live music
    return 1.0

def _is_live_music_hour(dt: datetime) -> bool:
    wd = dt.weekday()  # 0=Mon
    h = dt.hour
    is_weekend_night = wd in (4, 5)  # Fri, Sat
    if is_weekend_night:
        return h >= 22 or h <= 1
    else:  # Sun-Thu
        return 22 <= h <= 23

def _bar_is_open(dt: datetime) -> bool:
    wd = dt.weekday()
    h = dt.hour
    is_weekend = wd in (4, 5)
    if is_weekend:
        return not (2 < h < 9)   # closed 2am-9am
    else:
        return not (1 < h < 9)   # closed 1am-9am (approx 12:30am)

# ---------------------------------------------------------------------------
# Irish public holidays
# ---------------------------------------------------------------------------
def _build_irish_holidays(years):
    ie = holidays.Ireland(years=years)
    return set(ie.keys())

def _st_patricks_week(date):
    return date.month == 3 and 14 <= date.day <= 17

def _city_event(date):
    events = [
        (6, 16),   # Bloomsday
        (10, 27),  # Dublin City Marathon (approximate last Sunday October)
    ]
    for m, d in events:
        if date.month == m and abs(date.day - d) <= 1:
            return True
    return False

def _major_sports_event(date):
    # Six Nations: Feb-March (saturdays with home games roughly)
    if date.month in (2, 3) and date.weekday() == 5:
        return RNG.random() < 0.4
    # GAA: August-September weekends
    if date.month in (8, 9) and date.weekday() in (5, 6):
        return RNG.random() < 0.3
    return False

def _school_holiday(date):
    month, day = date.month, date.day
    # Christmas
    if month == 12 and day >= 21:
        return True
    if month == 1 and day <= 7:
        return True
    # Easter (approximate)
    if month == 4 and day <= 14:
        return True
    # Mid-term
    if month == 2 and 17 <= day <= 21:
        return True
    if month == 10 and 27 <= day <= 31:
        return True
    # Summer
    if month in (6, 7, 8):
        return True
    return False

def _payday_period(date):
    return date.day <= 3 or date.day >= 28

# ---------------------------------------------------------------------------
# Synthetic weather (correlated with Dublin climate norms)
# ---------------------------------------------------------------------------
def _gen_weather(dates):
    records = []
    for d in dates:
        m = d.month
        base_temp = [6, 6, 8, 10, 13, 15, 17, 17, 14, 11, 8, 6][m - 1]
        temp = float(RNG.normal(base_temp, 2.5))
        rain_prob = [0.60, 0.55, 0.50, 0.50, 0.45, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.65][m - 1]
        rain = float(RNG.exponential(2.5)) if RNG.random() < rain_prob else 0.0
        wind = float(RNG.gamma(2, 6))
        severe = bool(rain > 12 or wind > 50)
        records.append({"date": d, "temp_c": round(temp, 1), "rain_mm": round(rain, 1),
                        "wind_speed_kmh": round(wind, 1), "weather_severity_flag": severe})
    return pd.DataFrame(records).set_index("date")

def _gen_airport_arrivals(dates):
    records = []
    for d in dates:
        m = d.month
        base = [25000, 26000, 30000, 35000, 42000, 48000,
                52000, 51000, 45000, 38000, 28000, 26000][m - 1]
        wd_boost = 1.15 if d.weekday() in (4, 5, 6) else 1.0
        val = int(RNG.normal(base * wd_boost, base * 0.08))
        records.append({"date": d, "airport_arrivals": max(val, 0)})
    df = pd.DataFrame(records).set_index("date")
    df["airport_arrivals_lag1"] = df["airport_arrivals"].shift(1).bfill().astype(int)
    return df

def _gen_cruise_days(dates):
    records = []
    for d in dates:
        m = d.month
        cruise_prob = [0.0, 0.02, 0.05, 0.15, 0.20, 0.20,
                       0.20, 0.20, 0.20, 0.15, 0.05, 0.0][m - 1]
        has_cruise = RNG.random() < cruise_prob
        n_ships = int(RNG.integers(1, 3)) if has_cruise else 0
        pax = int(RNG.normal(2500, 600) * n_ships) if n_ships > 0 else 0
        records.append({"date": d, "cruise_ship_flag": has_cruise,
                        "ships_in_port_count": n_ships,
                        "cruise_passenger_estimate": max(pax, 0)})
    return pd.DataFrame(records).set_index("date")

# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
def generate(start: str = "2024-01-01", end: str = "2025-12-31",
             output_path: str = None) -> pd.DataFrame:
    """
    Generate synthetic hourly dataset for O'Donoghues Suffolk Street.
    Uses real public data (weather, footfall, events) if available to create realistic signal correlations.
    """
    from pathlib import Path
    
    dates = pd.date_range(start, end, freq="D").date
    years = list({d.year for d in dates})

    irish_holidays = _build_irish_holidays(years)
    weather_df = _gen_weather(dates)
    airport_df = _gen_airport_arrivals(dates)
    cruise_df = _gen_cruise_days(dates)

    # Load real data if available
    real_wx, real_enrich, real_ff, real_failte = None, None, None, None
    if Path("data/raw/weather_hourly.csv").exists():
        real_wx = pd.read_csv("data/raw/weather_hourly.csv")
        real_wx["timestamp_hour"] = pd.to_datetime(real_wx["timestamp_hour"])
        real_wx = real_wx.set_index("timestamp_hour")
        
    if Path("data/raw/enrichment.csv").exists():
        real_enrich = pd.read_csv("data/raw/enrichment.csv")
        real_enrich["date"] = pd.to_datetime(real_enrich["date"]).dt.date
        real_enrich = real_enrich.set_index("date")
        
    if Path("data/raw/footfall_hourly.csv").exists():
        real_ff = pd.read_csv("data/raw/footfall_hourly.csv")
        real_ff["timestamp_hour"] = pd.to_datetime(real_ff["timestamp_hour"])
        real_ff = real_ff.set_index("timestamp_hour")
        
    if Path("data/raw/failte_events.csv").exists():
        real_failte = pd.read_csv("data/raw/failte_events.csv")
        real_failte["date"] = pd.to_datetime(real_failte["date"]).dt.date
        real_failte = real_failte.set_index("date")

    rows = []
    for date in dates:
        wd = date.weekday()
        is_bank_hol = date in irish_holidays
        is_st_pat = _st_patricks_week(date)
        is_city_ev = _city_event(date)
        is_sports = _major_sports_event(date)
        is_school_hol = _school_holiday(date)
        is_payday = _payday_period(date)
        is_special = bool(RNG.random() < 0.04)  # ~4% days have special event

        w = weather_df.loc[date]
        a = airport_df.loc[date]
        c = cruise_df.loc[date]

        for hour in range(9, 26):  # 9am through 1am (hour 25 = 1am)
            actual_hour = hour % 24
            dt_full = datetime(date.year, date.month, date.day) + timedelta(hours=hour)
            dt = datetime(date.year, date.month, date.day) + timedelta(hours=actual_hour)
            # handle overnight crossing
            if hour >= 24:
                dt = datetime(date.year, date.month, date.day) + timedelta(days=1, hours=actual_hour)
            
            if not _bar_is_open(dt_full):
                continue

            is_music = _is_live_music_hour(dt_full)
            
            # Extract real weather or fallback
            if real_wx is not None and dt in real_wx.index:
                w_hourly = real_wx.loc[dt]
                temp_c = float(w_hourly["temp_c"]) if pd.notna(w_hourly["temp_c"]) else float(w["temp_c"])
                rain_mm = float(w_hourly["rain_mm"]) if pd.notna(w_hourly["rain_mm"]) else float(w["rain_mm"])
            else:
                temp_c = float(w["temp_c"])
                rain_mm = float(w["rain_mm"])
                
            # Extract real footfall
            ff_count = 0
            if real_ff is not None and dt in real_ff.index:
                ff_row = real_ff.loc[dt]
                if isinstance(ff_row, pd.DataFrame):
                    ff_row = ff_row.iloc[0]
                ff_count = ff_row.get("suffolk_nassau_footfall", 0)

            # Extract real enrichment or fallback
            if real_enrich is not None and date in real_enrich.index:
                e_daily = real_enrich.loc[date]
                is_st_pat = bool(e_daily.get("st_patricks_week_flag", is_st_pat))
                is_city_ev = bool(e_daily.get("city_event_flag", is_city_ev))
                is_sports = bool(e_daily.get("major_sports_event_flag", is_sports))
                is_bank_hol = bool(e_daily.get("bank_holiday_flag", is_bank_hol))
                c_ship_flag = bool(e_daily.get("cruise_ship_flag", c["cruise_ship_flag"]))
            else:
                c_ship_flag = bool(c["cruise_ship_flag"])
                
            failte_count = 0
            if real_failte is not None and date in real_failte.index:
                f_daily = real_failte.loc[date]
                failte_count = int(f_daily.get("failte_event_count", 0))

            ctx = {
                "st_patricks_week_flag": is_st_pat,
                "city_event_flag": is_city_ev,
                "major_sports_event_flag": is_sports,
                "bank_holiday_flag": is_bank_hol,
                "cruise_ship_flag": c_ship_flag,
                "special_event_flag": is_special,
                "failte_event_count": failte_count,
            }

            base = BASE_ORDERS.get(actual_hour, 10)
            mult = (
                WEEKDAY_MULT[wd]
                * MONTH_MULT[date.month]
                * _weather_mult(temp_c, rain_mm)
                * _event_mult(ctx)
                * _live_music_mult(actual_hour, wd, is_music)
                * _footfall_mult(ff_count)
            )

            mean_orders = base * mult
            orders = max(0, int(RNG.normal(mean_orders, mean_orders * 0.18)))

            food_ratio = FOOD_TICKET_RATIO.get(actual_hour, 0.0)
            food_tickets = max(0, int(RNG.normal(orders * food_ratio, orders * food_ratio * 0.15))) if food_ratio > 0 else 0

            sales = round(orders * RNG.normal(11.50, 2.0), 2)  # avg ~€11.50 per transaction

            # Busy label: thresholds relative to the base for that hour+weekday
            p = orders / max(base * WEEKDAY_MULT[wd], 1)
            if p < 0.5:
                label = "quiet"
            elif p < 0.9:
                label = "normal"
            elif p < 1.4:
                label = "busy"
            else:
                label = "slammed"

            rows.append({
                "timestamp_hour": dt.strftime("%Y-%m-%d %H:00:00"),
                "hour": actual_hour,
                "weekday": wd,
                "month": date.month,
                "is_weekend": wd in (5, 6),
                "week_of_year": dt.isocalendar()[1],
                "orders_count": orders,
                "food_tickets_count": food_tickets,
                "sales_total": sales,
                "covers_count": max(0, int(food_tickets * RNG.normal(1.4, 0.3))),
                "busy_label": label,
                "live_music_flag": is_music,
                "special_event_flag": is_special,
                "promo_flag": bool(RNG.random() < 0.02),
                "stockout_flag": bool(RNG.random() < 0.01),
                "menu_change_flag": False,
                "bar_staff_count": None,
                "kitchen_staff_count": None,
                "temp_c": float(w["temp_c"]),
                "rain_mm": float(w["rain_mm"]),
                "wind_speed_kmh": float(w["wind_speed_kmh"]),
                "weather_severity_flag": bool(w["weather_severity_flag"]),
                "airport_arrivals": int(a["airport_arrivals"]),
                "airport_arrivals_lag1": int(a["airport_arrivals_lag1"]),
                "cruise_ship_flag": bool(c["cruise_ship_flag"]),
                "ships_in_port_count": int(c["ships_in_port_count"]),
                "cruise_passenger_estimate": int(c["cruise_passenger_estimate"]),
                "bank_holiday_flag": is_bank_hol,
                "school_holiday_flag": is_school_hol,
                "payday_period_flag": is_payday,
                "major_sports_event_flag": is_sports,
                "city_event_flag": is_city_ev,
                "st_patricks_week_flag": is_st_pat,
            })

    df = pd.DataFrame(rows)
    df["timestamp_hour"] = pd.to_datetime(df["timestamp_hour"])
    df = df.sort_values("timestamp_hour").reset_index(drop=True)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df):,} rows to {output_path}")

    return df


if __name__ == "__main__":
    df = generate(
        start="2023-01-01",
        end="2026-08-31",
        output_path="data/synthetic/odonoghues_hourly.csv",
    )
    print(df.shape)
    print(df[["timestamp_hour", "orders_count", "food_tickets_count", "busy_label"]].head(20))
