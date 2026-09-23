# O'Donoghues Suffolk Street — Demand Forecast System
*Handoff spec — read this first. It should be enough to pick the project up cold.*

**Last updated:** 2026-09-23. If this date is more than ~2 weeks old, treat every number in this
doc as approximate and re-derive from `models/retrain_history.csv` / `data/processed/features.parquet`
rather than trusting it blindly — see [Keeping this doc honest](#keeping-this-doc-honest).

---

## Overview

Operational decision-support system for kitchen and bar management at O'Donoghues pub,
15 Suffolk St, Dublin D02 C671. Predicts hourly customer demand to support staffing, prep,
and stock decisions. Not a generic ML demo — every design choice is oriented toward a shift
manager reading it before service. See `ARCHITECTURE.md` for *why* each major decision was
made and what alternatives were rejected; this doc is about current state and how to operate it.

Deployed dashboard: **https://odonoghuessuffolk-forecast.streamlit.app**
Repo: **https://github.com/AdithKB/ODonoghuesSuffolk-Forecast**

---

## Inputs

### Real POS data (as of 2026-07, replacing the earlier synthetic-only approach)

Training targets now come from **real Titan BI POS exports**, not synthetic data:

- `scripts/scrape_titanbi.py` — scrapes hourly item quantities from titanbi.net (Performance By Hour
  report) using a Chrome session cookie (`browser_cookie3`) for auth. Output: `data/raw/pos_titanbi/hourly_*.csv`.
- `scripts/scrape_titanbi_txn.py` — scrapes daily transaction counts (Transaction Report) for
  `basket_size`/`transactions`. Output: `data/raw/pos_titanbi/daily_txn_*.csv`. **Not currently run by
  any cron job** — `basket_size`/`transactions` aren't consumed by the model, so this is harmless, but
  run it manually if you start using those columns.
- `src/ingest_titan.py` — merges hourly + daily files into a model-ready frame. `orders_count` =
  Titan's hourly item quantity. `food_tickets_count` is *derived*, not directly measured: daily total
  items × a fixed kitchen-hour weight distribution (calibrated from a Jul 2026 PDF export showing
  ~81.2 food tickets/day) — see `_FOOD_HOUR_WEIGHTS` in `ingest_titan.py`. This derivation is a known
  simplification (see [Known issues](#known-issues)).
- **Real POS data is gitignored** (`data/raw/pos_*/`) for PII reasons and only exists on machines
  that run the scraper directly — never on GH Actions or Streamlit Cloud. This is the source of a
  major bug fixed 2026-09-23 (see [Known issues](#known-issues) and `ARCHITECTURE.md` §Data pipeline).
- `data/synthetic/odonoghues_hourly.csv` still exists and `src/synthetic.py` still works, but neither
  is in the live training path anymore. Keep it around as a fallback/demo dataset only.

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
| DCC pedestrian counters | `fetch_footfall.py` | Hourly CSV download | Primary: Grafton/Nassau/Suffolk St; 5 backup counters for imputation. **Public feed currently lags — see Known issues.** |
| Fáilte Ireland events | `fetch_footfall.py` | Upcoming only | Filtered to 5km haversine radius; not historical |
| Football fixtures | `fetch_sports.py` | football-data.org API | Needs `FOOTBALL_DATA_API_KEY` in `.env`; degrades gracefully (empty df) without it |
| Six Nations / horse racing / Ireland rugby | `fetch_sports.py`, `fetch_horse_racing.py` | Static + scrape w/ fallback | — |

**Critical distinction — historical vs forecast-available signals:**

| Signal type | Available for past dates | Available for future dates |
|---|---|---|
| Demand lags (24h, 168h, 336h) + trend features | Yes | Yes — look back into training history |
| Calendar features | Yes | Yes — always known |
| Weather (Open-Meteo) | Historical archive | Yes — 7-day forecast window |
| Static event calendar / TV sports fixtures | Yes | Yes — covers weeks/months ahead |
| Fáilte Ireland events | No — upcoming only | Yes |
| DCC footfall (current hour) | Yes | No — unknown at forecast time |
| DCC footfall lags (24h, 168h) | Yes | Yes |
| Airport arrivals | Yes | Partial — quarterly lag |

---

## Feature Engineering (`src/features.py`)

**Current build state:** 23,694 rows × 146 columns (2024-01-15 → 2026-09-29, including 7 days of
future forecast stub rows). 113 features in `SAFE_FOR_NEXT_DAY` (the set actually used for
next-day prediction — see `src/model.py`).

### Feature groups

- **Demand lags** — 24h/48h/168h/336h calendar-aligned, `same_slot_4w_avg` (4-week same-hour-same-weekday
  average), `same_wd_hour_roll8` (rolling avg of last 8 occurrences of that weekday+hour — tighter
  than the 4-week average), `roll_mean_{3,6,12,24,168}`, `roll_std_24`, `roll_max_24`, EWMA (spans
  6/24/168h — only `ewma_168` is in `SAFE_FOR_NEXT_DAY`)
- **Trend** — `trend_90d` (90-day rolling mean of daily totals), `trend_21d` (added 2026-09-23 — a
  faster-reacting complement that catches a 2-3 week swing `trend_90d` is too slow to see),
  `trend_ratio_21_90` (short-window ÷ long-window — an explicit "are we currently running above or
  below baseline" signal), `yoy_ratio` (28-day avg vs the same 28-day window 364 days prior)
- **Recency-context flags** (added 2026-09-23) — `event_impact_score_max_28d` (was there an outsized
  one-off event, e.g. a PPV boxing night, in the last 4 weeks that inflated the lag/rolling anchors
  above?), `data_gap_recent_28d` (was there a likely POS scraper outage — not Christmas — in the
  last 4 weeks that deflated them?). Both let the model discount an anchor it would otherwise trust
  at face value, without touching the anchor's own math.
- **Calendar** — cyclical sin/cos encodings for hour/weekday/month, Friday/Saturday, weekend,
  lunch/dinner/music windows, hours since food close, exam period, fresher/RAG week, budget day,
  Cheltenham Festival
- **Weather** — rain, temperature, wind, sunshine, UV, precip probability, plus derived
  `rainy_day_flag` / `afternoon_rain_mm` / `outdoor_weather_flag` (see code comments in
  `add_interaction_features` for why raw hourly rain is a weak signal on its own)
- **Airport / Cruise** — daily arrivals + z-score, ships in port, cruise passenger estimate
- **Events** — aviva/croke/nearby venue/city/special flags, `event_impact_score` (1–4),
  `event_intensity` (composite), St. Patrick's week (auto-detected, not manual), Bloomsday,
  Christmas/New Year/summer tourism
- **TV sports** — football-data.org fixtures + Six Nations + horse racing + Ireland rugby;
  `tv_sports_flag/intensity`, `match_kickoff_proximity`, `is_ireland_match_window`,
  `days_until_next_match` (pre-event buildup, known ahead of time — safe for next-day)
- **Footfall** — `suffolk_nassau_footfall`, nearby aggregates, city z-score, lag 24h/168h, roll_24h,
  `suffolk_is_busy_flag`, `suffolk_counter_is_live` (imputation flag — see Known issues)
- **Fáilte** — event_count, free_event_count, festival_count within 5km
- **Academic/financial** — TCD term, school holiday, payday period, days_from_payday
- **Interactions** — `weekend_x_music`, `rain_x_weekend`, `late_night_x_fri_sat`, `sunny_afternoon`,
  `tourism_pressure` (airport z-score + cruise flag)

### Prediction-time constraint (`SAFE_FOR_NEXT_DAY` in `src/model.py`)

Prediction excludes any value that is not known before the forecast period begins:
`orders_count`/`food_tickets_count` for the target hour, `suffolk_nassau_footfall` current hour,
`suffolk_footfall_lag_1h`, and the intra-day lags (`lag_1/2/3`, row-based — reserved for a
same-day-nowcasting mode that isn't built yet, see `FULL_FEATURES`). Everything else that's
backward-looking (all named-lag features, event/calendar/weather signals, the trend/recency flags
above) is included. This is the leakage boundary — **any new feature must be audited against it**
before being added to `SAFE_FOR_NEXT_DAY`.

---

## Model (`src/model.py`)

### Architecture

Two-level ensemble per target (`orders_count`, `food_tickets_count`):

1. **Global XGBoost** — one model trained on all hours, all shifts. Falls back predictor for any
   shift without its own model.
2. **Shift-specific models** — 5 shifts × 2 targets, each a blend of XGBoost (asymmetric loss) +
   LightGBM (quantile τ=0.70), blend weight chosen by grid search:

   | Shift | Hours |
   |---|---|
   | `lunch` | 12–15 |
   | `evening` | 17–20 |
   | `late_bar_weekend` | 21–02, Fri/Sat |
   | `late_bar_weekday` | 21–02, Sun–Thu |
   | `off_peak` | 09–11, 16 |

3. **Baseline** — same-hour-last-week (`lag_168h`) with rolling-mean fallback. Comparison benchmark,
   shown alongside XGBoost in the dashboard.
4. **Recent-trailing bias correction** (added 2026-09-23, `dashboard/app.py::compute_recent_bias_ratios`)
   — a post-hoc multiplier on the shift-routed prediction, based on how biased the model's own
   predictions have been over the trailing 10 days, capped at ±20%. **This is not a training-time
   change** — it's applied at prediction time in the dashboard, not baked into the saved model files.
   See `ARCHITECTURE.md` for why this was chosen over a full concept-drift ensemble.

Routing logic: `dashboard/app.py::_predict_with_shift_routing` — each hour uses its shift's blend if
one exists, else the global model.

### Training

- Dead hours (2–8am, avg 0.1 items/hr) excluded from training — inflate MAPE without adding signal.
- 8-fold walk-forward (rolling-origin) CV for the global model — each fold trains on all history up
  to a cutoff, evaluates on the next 4 weeks. **No random splits anywhere in this pipeline.**
- Shift models: Optuna (150 XGB + 50 LGBM trials per shift/target, TPE sampler, warm-started from
  `models/optuna_journal.log` / `optuna_lgbm_journal.log` on routine retrains — pass `--force-retune`
  to `src/model.py` to start fresh after a significant feature change).
- **Held-out evaluation** (fixed 2026-09-23): shift-model reporting, blend-weight selection, and the
  `--skip-optuna` fast-retrain path used to score against the *full* shift dataset — ~85% the same
  rows the model was just trained on, i.e. mostly in-sample. Now uses only the genuine last-15%
  chronological slice throughout. See `ARCHITECTURE.md` and [Known issues](#known-issues) — this fix
  revealed shift-specific models may not clearly beat the global model on `orders_count`, which the
  old in-sample numbers had been masking.
- Asymmetric loss (`_asymmetric_obj` in `src/model.py`): under-predictions penalised 2× more than
  over-predictions — understaffing/stockouts cost more than one extra staff member on the floor.
- MAPIE-style conformal intervals (`src/conformal.py`) — 80% coverage bands from CV residuals,
  segmented by hour-of-day. **Verified empirically calibrated** (82.6%/83.2% actual coverage on a
  genuinely held-out fold, checked 2026-09-23) — display-only, never modifies the point forecast.

### Current results (real POS data, retrained 2026-09-23)

| Target | Baseline MAE | XGB MAE | Improvement | Most-recent-fold MAE (canary) |
|---|---|---|---|---|
| orders_count | 24.57 | 17.29 | +29.6% | 14.22 |
| food_tickets_count | 1.31 | 0.96 | +26.4% | 0.45 |

These are walk-forward CV numbers on the **global** model, on real data. Treat them as the honest
floor — the shift-routed + bias-corrected numbers actually shown in the dashboard are better in
practice (backtested at +2.4% week-level bias on a hard test week vs the baseline's +4.6%, see
`models/retrain_history.csv` for the trend over time). Full history of every retrain's CV metrics is
appended to `models/retrain_history.csv` with a regression check against the previous run.

### Artifacts

- `models/xgb_{target}.json`, `models/baseline_{target}.pkl` — global models
- `models/xgb_{target}_{shift}.json`, `models/lgbm_{target}_{shift}.txt`, `models/blend_{target}_{shift}.json` — shift models + blend weights
- `models/best_params_{target}_{shift}.json` — Optuna best hyperparameters
- `models/feature_importance_{target}.csv`, `models/cv_out_of_sample_preds_{target}.csv` — diagnostics
- `models/mapie_{target}_residuals.json` — conformal interval residuals
- `models/retrain_history.csv` — one row per retrain, CV metrics + regression verdict (new 2026-09-23)
- `models/optuna_journal.log`, `models/optuna_lgbm_journal.log` — Optuna warm-start state (`optuna_journal.log` is gitignored; `optuna_lgbm_journal.log` currently isn't — inconsistent, low priority to fix)

---

## Outputs / Dashboard (`dashboard/app.py`)

**Stack:** Streamlit, dark theme, JetBrains Mono for numeric values, no emoji.

### Panels

| Panel | Content |
|---|---|
| Sidebar | Date nav (prev/next/today via `on_click` callbacks), day flags, weather sliders, "Refresh Live Data" button |
| Header | Venue name, forecast date, address, operating hours |
| Shift cards | Status badge, orders, food tickets, peak hour, prep recommendation, per shift |
| Day banner | Single-line overall pressure summary |
| Hourly chart | Orders bars + food tickets line + last-week baseline dotted; conformal interval band |
| Footfall expander | Peak hourly count, yesterday/last-week same time, city z-score |
| Signals panel | Grouped HIGH PRESSURE / MODERATE / CONTEXT rows |
| Feature importance | Top 10 features by XGBoost gain |
| Hourly table | Time, Shift, Orders, Food tickets, Baseline, CSV download |

### New since 2026-09-23

- **Data freshness warning** — shown if the most recent real POS data is >2 days old. This is exactly
  the failure mode that went unnoticed for 2 months (see Known issues) — now visible in the UI instead
  of silent.
- **"Refresh Live Data" button now surfaces errors** instead of discarding subprocess output silently.
  If it warns "no local Titan POS export found," that's expected on Streamlit Cloud — it still
  refreshes public signals, just not the POS-dependent feature rebuild (see pipeline section below).
- **Bias-ratio drift log** (`data/bias_ratio_log.csv`) — every live forecast appends its
  `{target}_bias_ratio` for that date. A ratio sitting near the ±20% cap for many consecutive days is
  the signal to investigate rather than trust the correction indefinitely — nothing currently alerts
  on this automatically; check it by hand.

---

## Data refresh & retrain pipeline

Three separate, differently-scoped automations — understand why there are three before changing any of them:

### 1. GH Actions — hourly, public signals only (`.github/workflows/refresh_data.yml`)

Runs `python refresh_data.py` every hour at :30 on a GitHub-hosted runner, commits `data/raw/*.csv`
(weather, events, footfall, fixtures — none of it PII) back to the repo. **Cannot touch POS data or
retrain models** — the runner has no access to `data/raw/pos_titanbi/` (gitignored, local-only).
`refresh_data.py` handles this gracefully (catches the missing-directory case and returns after
saving public signals) — this used to crash the whole script and silently fail every run for
~2 months before the 2026-09-23 fix (see Known issues).

### 2. Local daily cron — `daily_refresh.sh`, 06:00 every day

```
0 6 * * * /Users/adithkb/Ireland/PersonalProjects/CustomerFootfallPrediction/daily_refresh.sh
```

Runs `scripts/scrape_titanbi.py --resume` (real POS, needs a logged-in Chrome session on this
machine — see [Known issues](#known-issues) if it stops working after a browser update) then
`refresh_data.py` (rebuilds `features.parquet` with the fresh POS data). **Does not retrain or push**
— purely refreshes local files for the next `git add && commit && push` (manual) or for the weekly
retrain (automated) to pick up.

### 3. Local weekly cron — `weekly_retrain.sh`, Monday 05:00

```
0 5 * * 1 /Users/adithkb/Ireland/PersonalProjects/CustomerFootfallPrediction/weekly_retrain.sh
```

Scrapes, refreshes, retrains (`refresh_data.py --retrain`), and auto-commits + pushes to `main`
— **unless** `log_retrain_history()` flagged a regression against the previous retrain's
most-recent-fold MAE, in which case it leaves the working tree dirty for manual review instead of
shipping a worse model. Logs to `data/weekly_retrain.log`.

### Manual commands

```bash
python refresh_data.py            # public signals + feature rebuild (needs local POS data for the rebuild step)
python refresh_data.py --retrain  # also retrains (~20-25 min, warm-started)
python src/model.py --force-retune  # full retrain, fresh Optuna search (after a real feature-set change)
python scripts/scrape_titanbi.py --resume       # backfill missing hourly POS days (needs Chrome logged into titanbi.net)
python scripts/scrape_titanbi_txn.py --resume   # backfill daily transaction counts (not currently automated)
```

Both scraper `--resume` flags scan **all** `hourly_*.csv` / `daily_txn_*.csv` files in
`data/raw/pos_titanbi/`, not just one filename (fixed 2026-09-23 — the old version checked only the
file matching today's date-stamped name, which changes daily, so `--resume` silently re-scraped
the entire 2024-present history every run).

---

## Known issues

- **`food_tickets_count` is derived, not measured.** Titan's hourly export gives item quantity per
  hour but doesn't split food vs bar; `food_tickets_count` is `daily_intensity × fixed_hour_weights`
  calibrated from one PDF export. This is a real accuracy ceiling for that target specifically — a
  proper POS export that separates food tickets would be a meaningful upgrade if Des's system supports it.
- **Two confirmed POS data gaps** (scraper outages, not real closures): 2026-05-25→06-04 (11 days)
  and 2026-07-21→07-23 (3 days). Both post-date the crontab pointing at a stale project path — see
  `daily_refresh.sh` git history. `data_gap_recent_28d` now flags these to the model, but the raw gap
  itself isn't backfilled (can't be — the data was never captured) and the 168h/4-week-avg lag
  features for the weeks immediately after each gap are working off less real signal than usual.
- **DCC footfall public feed is stalled** — as of 2026-09-23 it only has real data through 2026-06-02;
  Dublin City Council hasn't published H2 2026 pedestrian counts yet. This is an external data source
  lag, not a bug in this pipeline — `suffolk_counter_is_live` correctly flags it, but footfall
  features have been running on imputed/nearby-counter data for 3+ months. Worth periodically
  checking if DCC has caught up (`src/fetch_footfall.py`).
- **`scripts/scrape_titanbi_txn.py` isn't run by any cron job** — `daily_refresh.sh` only calls the
  hourly scraper. Harmless today since `basket_size`/`transactions` aren't model inputs, but if
  brainstorm.md idea #4 (basket-size predictor) gets built, wire this in too.
- **Shift-specific models' advantage over the global model is now uncertain for `orders_count`** —
  see the held-out-eval fix above. The comparison isn't fully apples-to-apples yet (the global
  model's own baseline in that comparison still has a residual in-sample advantage, since it was
  trained on all data including the shift's held-out slice). Worth a dedicated proper ablation before
  trusting either verdict fully — `notebooks/04_ablation_and_temporal_diagnostics.py` is the right
  starting point.
- **The September 2026 demand decline's root cause is unknown.** Bias-corrected, the model now tracks
  it reasonably (+2.4% week bias vs +10.9% before correction), but *why* demand has been declining
  week-over-week since early September was investigated (ruled out: weather, YoY effects, the
  Sep 5 boxing-night contamination) and not resolved. If it continues, worth asking Des directly.
- **`weekly_retrain.sh` auto-push assumes non-interactive git push works from cron** — it should,
  since it uses the same credential store as interactive `git push`, but hasn't been observed running
  unattended yet as of this doc's last update. Check `data/weekly_retrain.log` after the first
  Monday run.
- **`.streamlit/config.toml`** hardcodes dark theme — fine for now (single-user internal tool), would
  need revisiting for a multi-tenant SaaS pivot.

---

## Next tasks (priority order)

1. **Confirm the weekly retrain cron actually runs unattended** — check `data/weekly_retrain.log`
   after the first scheduled Monday run; git push from cron is untested in practice.
2. **Investigate the September 2026 decline's cause directly with Des** — the model now compensates
   for it statistically, but understanding *why* would let a leading indicator be added instead of
   only reacting after the fact.
3. **Proper ablation on shift-specific vs global models** — resolve the uncertainty the held-out-eval
   fix surfaced. Use `notebooks/04_ablation_and_temporal_diagnostics.py`.
4. **Get a food-ticket-specific POS export from Des** if Titan supports splitting food vs bar sales —
   would remove the biggest known accuracy ceiling on `food_tickets_count`.
5. **Check DCC footfall feed periodically** for whether H2 2026 data has been published — `fetch_footfall.py` needs no code change, just re-run once the source updates.
6. **Optional: Ticketmaster API** — `fetch_ticketmaster_events()` in `fetch_events.py` is implemented;
   set `TICKETMASTER_API_KEY` to enable dynamic event discovery beyond the static calendar.
7. **Consider alerting on `data/bias_ratio_log.csv`** rather than requiring a manual check — e.g. a
   simple threshold check appended to `daily_refresh.sh`.

---

## Keeping this doc honest

This file went stale for ~2 months (last updated at initial commit, describing a synthetic-data
system the project had already moved past) before this rewrite. Concretely, before trusting a claim
in this doc:

- **Model numbers** → check `models/retrain_history.csv` (has a timestamp and git commit per row)
- **Data freshness** → check `data/processed/features.parquet`'s max real (non-NaN `orders_count`) date, or just look at the dashboard's freshness warning
- **Pipeline health** → `gh run list --workflow=refresh_data.yml` (GH Actions), `data/cron.log` / `data/weekly_retrain.log` (local cron)
- **What's actually in `SAFE_FOR_NEXT_DAY`** → `src/model.py`, not this doc's feature list, if the two disagree

See `ARCHITECTURE.md` for the reasoning behind each major decision, including ones this doc
mentions only briefly.
