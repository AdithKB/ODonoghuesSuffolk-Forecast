# O'Donoghues Suffolk Street — Demand Forecast System
*Handoff spec — stable architecture + current build state*

---

## Overview

Operational decision-support system for kitchen and bar management at O'Donoghues pub, 15 Suffolk St, Dublin D02 C671. Predicts hourly customer demand up to ~60 days ahead to support staffing, prep, and stock decisions. Not a generic ML demo — every design choice is oriented toward a shift manager reading it before service.

---

## Inputs

### Synthetic training data (`src/synthetic.py`)
Generated hourly demand Jan 2023 → Sep 2026, calibrated to actual venue operating rules:
- Kitchen 09:00–21:00, bar until 02:00 Fri/Sat, live music schedule
- Two targets: `orders_count` (bar + kitchen combined), `food_tickets_count` (kitchen only)
- Schema matches `config/schema.yaml` — real POS exports slot in with no pipeline change

### External signals

All free, no API keys required unless noted.

| Source | File | Cadence | Notes |
|---|---|---|---|
| Open-Meteo weather | `fetch_public_data.py` | Daily | Historical archive + 7-day forecast |
| Smart Dublin airport arrivals | `fetch_public_data.py` | Quarterly CSV → daily | Z-score vs quarterly baseline |
| Dublin Port cruise schedule | `fetch_public_data.py` | HTML scrape | Vessel type = "Cruise Liners" |
| Irish public holidays | `fetch_public_data.py` | Static + `holidays` lib | — |
| Static event calendar | `fetch_events.py` | Hardcoded + verified | Aviva, Croke Park, 3Arena, Bloomsday, etc. |
| School / TCD term / payday | `fetch_events.py` | Computed annually | — |
| DCC pedestrian counters | `fetch_footfall.py` | Hourly CSV download | Primary: Grafton/Nassau/Suffolk St (50m from venue); 5 backup counters for imputation |
| Fáilte Ireland events | `fetch_footfall.py` | Upcoming only | Filtered to 5km haversine radius; not historical |

**Critical distinction — historical vs forecast-available signals:**

| Signal type | Available for past dates | Available for future dates |
|---|---|---|
| Demand lags (24h, 168h, 336h) | Yes | Yes — look back into training history |
| Calendar features | Yes | Yes — always known |
| Weather (Open-Meteo) | Historical archive | Yes — 7-day forecast window |
| Static event calendar | Yes | Yes — covers months ahead |
| Fáilte Ireland events | No — upcoming only | Yes |
| DCC footfall (current hour) | Yes | No — unknown at forecast time |
| DCC footfall lags (24h, 168h) | Yes | Yes |
| Airport arrivals | Yes | Partial — quarterly lag |

---

## Feature Engineering (`src/features.py`)

**Current build state:** ~22,525 rows × 105 columns; 89 usable features after lag warm-up removal. These numbers shift when synthetic data is extended or retrained.

### Feature groups

- **Demand lags** — 1h, 24h, 168h, 336h for both targets; rolling means 3h and 24h
- **Calendar** — cyclical sin/cos encodings for hour/weekday/month, Friday/Saturday, weekend, lunch/dinner/music windows, hours since food close
- **Weather** — rain, temperature, wind, sunshine hours (real data replaces synthetic on overlap)
- **Airport** — daily arrivals, z-score vs quarterly average
- **Cruise** — ships in port, cruise flag
- **Events** — aviva/croke/nearby venue/city/special flags, event_impact_score (1–4), event_intensity (composite), St. Patrick's week, Bloomsday, Christmas/New Year/summer tourism
- **Footfall** — suffolk_nassau_footfall, nearby aggregates, city z-score, lag 1h/24h/168h, roll_24h, is_busy_flag (>70th pct for that hour-of-week), counter_is_live (imputation flag)
- **Fáilte** — event_count, free_event_count, festival_count within 5km
- **Academic/financial** — TCD term, school holiday, payday period, days_from_payday
- **Interactions** — weekend × music, tourism_pressure, composite event_intensity

### Prediction-time constraint (`SAFE_FOR_NEXT_DAY` feature set)

Prediction excludes any value that is not known before the forecast period begins. Specifically excluded at forecast time:
- `orders_count` and `food_tickets_count` for the target hour (the thing being predicted)
- `suffolk_nassau_footfall` current hour (unknown until people walk past)
- `suffolk_footfall_lag_1h` (same reason)

Included at forecast time: all demand lags ≥24h, all calendar flags, weather forecast, all event/calendar/footfall-lag features. This is what `SAFE_FOR_NEXT_DAY` enforces — the model cannot use a signal it wouldn't have in production.

---

## Model (`src/model.py`)

### Architecture
- XGBoost regressor (numeric forecast)
- Same-hour-last-week baseline (comparison + food-ticket shift classification fallback)

### Training
- 8-fold walk-forward (rolling-origin) CV — each fold trains on all history up to a cutoff, evaluates on the next 4 weeks, no future leakage
- `n_estimators=800`, `learning_rate=0.05`, `max_depth=6`, `early_stopping=50` (validated on last 20% of train split per fold)

### Shift labelling
Quiet / Normal / Busy / Slammed thresholds derived per shift from 25th/65th/88th historical percentiles of that shift's total orders.

### Current results (synthetic data — treat as upper bound, not production accuracy)

| Target | Baseline MAE | XGB MAE | Improvement | XGB shift accuracy |
|---|---|---|---|---|
| orders_count | 9.44 | 6.56 | 30.5% | 75.3% |
| food_tickets_count | 5.12 | 3.87 | 24.5% | 51.5% |

Top feature gain (orders_count): lag_168h (24%) → lag_336h (10%) → is_friday_saturday (9.9%) → lag_24h (5.7%) → roll_mean_24h (4%) → event_intensity (3%) → footfall lags (~0.5% each, expected to rise with real POS).

Food ticket shift accuracy at 51% is still near baseline — kitchen ticket patterns are harder to learn from synthetic data. Expected to improve substantially with real data.

### Artifacts
- `models/xgb_{target}.json`
- `models/baseline_{target}.pkl`
- `models/feature_importance_{target}.csv`

---

## Outputs / Dashboard (`dashboard/app.py`)

**Stack:** Streamlit, dark theme, JetBrains Mono for all numeric values, no emoji anywhere.

**Forecast horizon:** date picker covers Jan 2023 → Sep 2026. For dates beyond real POS history, the model uses demand lags from the synthetic training set + live enrichment signals.

### Panels

| Panel | Content |
|---|---|
| Controls bar | Expander at top: date picker, day flags, weather overrides, refresh button |
| Header | Venue name, forecast date, address, operating hours |
| Shift cards ×3 | Lunch (12–15:00) / Evening (17–21:00) / Late bar (21–close): status badge, orders, food tickets, peak hour, prep recommendation |
| Day banner | Single-line overall pressure summary with status-color left border |
| Hourly chart | Orders bars + food tickets line + last-week baseline dotted; shift shading; kitchen-close marker 21:00 |
| Footfall expander | Peak hourly count, yesterday same time, last week same time, city z-score, busy-hour count (DCC counter) |
| Signals panel | Grouped HIGH PRESSURE / MODERATE / CONTEXT rows with dot indicators |
| Feature importance | Horizontal bar, top 10 features by XGBoost gain |
| Hourly table | Time (index), Shift label, Orders, Food tickets, Baseline — all integers, CSV download |

### Controls (in-page expander at top)
- Date picker (defaults to today if in range, else last available)
- Day flags: live music, special event, sports event, cruise ship, St. Patrick's week
- Weather sliders: rain (mm), temperature (°C)
- "Refresh" button — runs full fetch pipeline + rebuilds feature table in-place

---

## Daily refresh pipeline

```bash
python refresh_data.py          # fetch weather + footfall + events + rebuild features
python refresh_data.py --retrain  # also retrains XGBoost models
```

Sequence: `fetch_public_data` → `fetch_footfall` → `build_features` → (optional) `model train`

---

## Constraints

- **No real POS data yet** — all training targets are synthetic; real accuracy unknown
- **Fáilte Ireland events** are upcoming-only (no historical feed) — static event calendar covers historical known events
- **Suffolk St footfall counter** goes offline periodically (especially 2026 XLSX); imputed from nearby counters with `suffolk_counter_is_live` flag
- **Food ticket shift accuracy** (~51%) is near baseline — do not use XGBoost for kitchen shift classification until real data is available; use same-hour-last-week baseline instead
- **60-day forward horizon** is limited by synthetic data extension; real forward capability requires the refresh pipeline running daily

---

## Next tasks (priority order)

1. **Connect real POS data** — schema in `config/schema.yaml`; replace `data/synthetic/odonoghues_hourly.csv` with real export; retrain
2. **Retrain on real data** — run `python src/model.py`; expect footfall feature importance to rise significantly
3. **Validate shift accuracy on real data** — decide whether to promote XGBoost or keep baseline for food ticket shift labels
4. **Schedule daily refresh** — `python refresh_data.py` as a cron job to keep weather forecast, footfall, and feature table current
5. **Optional: Ticketmaster API** — set `TICKETMASTER_API_KEY` env var; `fetch_ticketmaster_events()` in `fetch_events.py` already implemented; adds dynamic event discovery beyond the static calendar
