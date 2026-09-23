# Architecture & Design Decisions

*Why this system is built the way it is, and what was rejected instead. Pair with `HANDOFF.md`
(current state, how to operate it) — this doc is about reasoning, not state, and goes stale much
more slowly.*

---

## Design principles, in priority order

1. **Leakage-safety over accuracy.** A model that's 5% less accurate but never uses information it
   wouldn't have at forecast time is worth more than one that's 5% better and quietly wrong every
   time the world doesn't cooperate with training-time assumptions. Every feature is audited against
   `SAFE_FOR_NEXT_DAY` before it ships.
2. **Cost-conscious for a solo project.** This runs on a laptop's crontab, GitHub Actions' free tier,
   and Streamlit Community Cloud's free tier. No managed feature store, no MLflow, no Kubernetes.
   Every infrastructure decision below is partly "what's the simplest thing that's still correct"
   for a single operator, not "what would a 10-person ML platform team build."
3. **Visible failure over silent failure.** The single most expensive mistake in this project's
   history (2 months of silently stale data, see §Data pipeline) came from a pipeline that failed
   quietly. Every fix made since has prioritized *surfacing* problems (warnings, logs, regression
   checks) over trying to prevent every possible failure mode outright.
4. **Real-world business logic over generic ML defaults.** Asymmetric loss, shift-specific models,
   the busyness-tier thresholds — all exist because a pub's demand curve and a manager's actual
   costs of being wrong don't match what a textbook regression setup assumes.

---

## Model family: gradient-boosted trees (XGBoost + LightGBM), not deep learning

**Chosen:** XGBoost (primary) + LightGBM (quantile blend partner) per shift.

**Rejected:**
- **Temporal Fusion Transformer / other deep sequence models.** TFT's advantages (attention-based
  interpretability, native multi-horizon quantiles, handling many correlated series) matter most
  above ~50k rows per series or with long lookback dependencies (>168h). This project has ~24k
  hourly rows total, well engineered lag/rolling features already do the "remembering" a
  sequence model would otherwise have to learn, and the team is one person who needs to debug
  production issues in an afternoon, not maintain a training loop with learning-rate schedules and
  attention-map introspection. Revisit if the row count grows an order of magnitude or if
  multi-venue support (a stated future direction) means genuinely correlated series across venues.
- **Prophet / ARIMA / classical statistical forecasting.** No natural way to ingest the rich mixed
  feature set here (weather, events, footfall, sports fixtures) without building a full regression
  layer on top anyway — at which point you've reinvented gradient boosting with extra steps and
  worse handling of nonlinear interactions (e.g. rain × weekend, event × time-of-day).
- **Foundation models (Chronos-2, TimesFM).** Promising for zero/few-shot forecasting with no
  feature engineering, but this project's edge *is* the feature engineering (footfall, events,
  sports fixtures fused with POS) — a foundation model would throw that away. Worth a benchmark
  someday, not a replacement.

Literature backs this for the current data regime: gradient boosting remains dominant for tabular
time series with rich mixed-type features at this scale (see `research.md` for citations gathered
during the 2026-09 accuracy investigation).

---

## Two models per shift, blended: XGBoost (asymmetric) + LightGBM (quantile)

**Why two, not one:** XGBoost is trained with a custom asymmetric loss (below) that shapes the point
forecast toward the business's actual cost asymmetry. LightGBM is trained as a quantile regressor
(τ=0.70) — deliberately biased toward slightly over-predicting, a different mechanism aimed at the
same goal. Blending them (weight chosen by grid search on a held-out slice, see below) captures cases
where one model's biases happen to cancel the other's errors better than either alone. This is a
cheap, standard ensembling technique — not novel, chosen because it reliably helps and costs one
extra model per shift to train.

**Why asymmetric loss at all, over standard MSE/MAE:** in a pub, under-predicting a busy Friday means
understaffing and stockouts — visible, embarrassing, costly. Over-predicting a quiet Tuesday means
one extra staff member on the floor — a soft cost. `_asymmetric_obj` in `src/model.py` penalizes
under-prediction 2× more than over-prediction, encoding this directly into what the model optimizes
for, rather than trying to correct for it after the fact with a symmetric-loss model and a manual
safety margin (which would be less principled and harder to tune).

---

## Shift-specific models + global fallback, not one model for all hours

**Chosen:** 5 shift-specific model pairs (lunch / evening / late_bar_weekend / late_bar_weekday /
off_peak) per target, falling back to one global model for any hour without a shift model.

**Reasoning at the time:** lunch demand cares about weather and food; late-bar weekend demand cares
about live music and whether it's a big match night. A single global model regresses toward the
mean across fundamentally different demand regimes. `late_bar` was originally one shift, later split
into weekday/weekend specifically because Friday/Saturday midnight (80-200 items) and Tuesday
midnight (0-10 items) are different enough problems that lumping them hurt both.

**Status as of 2026-09-23 — genuinely uncertain, not settled:** fixing a held-out-evaluation bug in
the shift-model training/reporting code (see §Overfitting below) revealed that on the honestly
re-measured numbers, shift-specific models for `orders_count` now score *worse* than the plain
global model in most shifts. The previous framing ("shift models clearly beat global") was more
optimistic than warranted — it was partly an artifact of evaluating on ~85% in-sample data. This
doesn't necessarily mean shift-specialization is wrong; smaller shifts (`late_bar_weekend`: 1,400
rows) may simply be too data-starved for a heavily Optuna-tuned specialist to beat a generalist
trained on 5-10× more data, especially now that the comparison is fairer. **This is flagged, not
resolved** — the fair comparison still has a residual asymmetry (the global model's own baseline
number in that comparison was itself trained on all data, including the shift's held-out rows). A
proper resolution needs the global model excluded from that data too, which needs a dedicated
retrain design, not a quick fix. See `HANDOFF.md` → Next tasks.

**Why not go further and split by day-of-week too, or fully personalize per (shift × day)?** Data
volume. `late_bar_weekend` at 1,400 rows already looks under-powered relative to its ~150 tunable
hyperparameters (150+50 Optuna trials). Splitting further would make the data-starvation problem
above strictly worse, not better, without first resolving whether shift-splitting is earning its
complexity at all.

---

## Walk-forward (rolling-origin) CV, never random splits

**Chosen exclusively.** Every fold trains on all history up to a cutoff and evaluates on the next
~4 weeks. 8 folds for the global model; a similar internal 3-fold walk-forward inside each Optuna
trial for shift models.

**Why not k-fold random CV:** random splits let the model "see the future" relative to any given
test row (a training row from next month sitting next to a test row from this month). For time
series with real autocorrelation and trend, this systematically overstates how well a model will
do on data it hasn't seen yet — silently, since the CV score still looks good. Walk-forward is the
only honest evaluation methodology for this problem shape, full stop. This has been true since the
project's synthetic-data days and was never revisited because there's no real argument for
revisiting it.

---

## Overfitting: the held-out-evaluation fix (2026-09-23)

**The bug:** shift-model reporting, blend-weight selection (`_compute_blend_weight`), and the
`--skip-optuna` fast-retrain path all evaluated on the *full* shift dataset after training — but
the model had just been fit on ~85% of those same rows (only the last chronological 15% was
genuinely held out, for early stopping). Reported MAEs and the chosen blend weight were therefore
optimistically biased, more so the more a model had memorized its training data. Separately, the
fast-retrain LightGBM path validated against its own training set (`valid_sets=[train_data]`),
making early stopping there nearly inert.

**The fix:** compute a single held-out chronological slice once per shift (last 15%, same split
already used internally for early stopping) and route *all* evaluation, blend-weight selection, and
reporting through it — training still uses the other 85%, nothing about what gets learned changed,
only what gets measured.

**Why this matters generally, not just for the numbers it changed:** any pipeline that reports a
metric computed on the same (or mostly-same) data a model trained on will look better than it is,
and the gap grows with model capacity and hyperparameter search intensity — exactly the situation
150+50 Optuna trials per shift creates. This is a standard trap, not specific to this codebase, but
worth stating explicitly here because it's exactly the kind of bug that's invisible until you go
looking for it (the numbers *looked* plausible before the fix).

**Deliberately not done:** blind feature pruning (removing correlated lag features) or per-shift
hyperparameter regularization tuning (tighter bounds for small shifts). Both are plausible next
steps, but doing them without a proper ablation study risks removing something that's actually
earning its keep, on a guess. `notebooks/04_ablation_and_temporal_diagnostics.py` exists for this
exact purpose and is the right place to validate either change before making it.

---

## Concept drift: post-hoc bias correction, not a full adaptive ensemble

**The problem:** lag/rolling-anchored features are, by construction, built from data *before* a
trend's most recent move. Any model leaning on them systematically overshoots during an
in-progress decline and undershoots during an upswing — discovered concretely in September 2026,
when the model over-predicted orders by 10-30% every day for a week during a real demand slide.

**What the research pointed to:** Error Contribution Weighting / Gradient Descent Weighting
(Grazzi et al., see `research.md`) — train two models (one on recent history only, one on full
history) and blend them, weighted by which one's been more accurate lately.

**What was actually built:** a much lighter post-hoc correction. New predictions are multiplied by
`actual ÷ predicted` summed over the trailing 10 days (capped at ±20%), computed at prediction time
in `dashboard/app.py`, not baked into training.

**Why the lighter version, deliberately:** the full ECW/GDW approach means training and maintaining
a second model pipeline, permanently, for a solo project. That's real ongoing complexity — more
code paths to keep correct, more to retrain, more that can silently drift out of sync with the
primary model. The post-hoc correction needs zero additional training infrastructure, is fully
interpretable (a single number you can print and reason about), and was validated before shipping
across three independent tests (2.5-year walk-forward CV backtest, the actual production
shift-routed model over Jun-Sep 2026, and a stable-period regression check) rather than assumed to
work from the paper alone. It solved the concrete problem it was built for (week bias +10.9% → +2.4%
on the hardest test week) without solving the general concept-drift problem in the abstract — that's
an honest trade, appropriate to the size of the team maintaining this.

**What it does NOT do:** correct a bias that hasn't shown up in the last 10 days yet, by
construction. It reacts to drift, it doesn't predict drift starting. If a full adaptive ensemble
becomes worth the maintenance cost later (e.g. if drift episodes become frequent enough that a
10-day reaction lag is itself the binding constraint), the ECW/GDW direction in `research.md` is
where to start.

---

## Recency-context flags, not modifying the anchor features directly

Two related problems surfaced the same underlying pattern: a one-off event (a PPV boxing night)
inflating `same_slot_4w_avg`/`lag_168h` for weeks after, and a POS scraper outage deflating them the
same way. The tempting fix for either is to winsorize/cap the raw values feeding those rolling
calculations. **This was deliberately rejected** in favor of adding new flag features
(`event_impact_score_max_28d`, `data_gap_recent_28d`) that tell the model "the window behind you
was contaminated," letting the model itself learn how much to discount the anchor.

**Why:** winsorizing the anchor's own math means computing a threshold from historical data, which
risks a subtle lookahead bug if that threshold is derived from anything beyond strictly-prior data —
and even done correctly, it silently changes what `lag_168h` *means* in a way that's hard to audit
later. A flag feature is additive, doesn't touch existing feature math at all, is trivially
auditable (grep for the column, see exactly when it's 1), and follows the standard "give the model
the information, let it learn the correction" principle that's what tree-based models are actually
good at. It requires a retrain to have effect (the model has to *learn* the flag matters — it's not
a guaranteed win the way the bias-correction layer is), which is a real cost, but it's a safer one.

---

## Real POS via browser-cookie scraping, not an API or manual export

**Chosen:** `scripts/scrape_titanbi.py` uses `browser_cookie3` to lift a live Chrome session cookie
for titanbi.net and scrapes the Performance-By-Hour report directly.

**Why not an official API:** Titan BI (the POS vendor here) doesn't expose one accessible to this
venue's account tier, as far as this project has determined. Scraping the authenticated web report
was the only path to hourly granularity without asking Des to manually export and hand over a file
on some cadence.

**Why not manual export + `sanitize_pos.py` as the primary path:** `sanitize_pos.py` still exists
and is the documented path for *whenever Des hands over a raw till export directly* — but it's a
one-off/occasional flow, not something that can run unattended daily. The scraper is what makes the
daily/weekly cron automation possible at all.

**Trade-off accepted knowingly:** this is fragile by nature — a Titan BI UI change, a cookie
expiring, or Chrome needing to be closed for `browser_cookie3` to read its cookie store cleanly
could break the scraper silently. This is exactly why the data freshness warning in the dashboard
(see "Visible failure over silent failure" above) and the retrain-history regression check exist —
the fragility is accepted, but made visible rather than designed away (which would likely mean
asking for an official API integration that may not exist, or moving to manual exports, which
defeats the automation).

---

## Data privacy: gitignore + `sanitize_pos.py`, not encryption or a separate private repo

Raw POS exports may contain transaction IDs, staff names, or payment fragments. The chosen design:
`data/raw/pos_*/` and pattern-matched filenames (`*pos*`, `*export*`, `*transactions*`, etc.) are
gitignored outright — never committed, not even encrypted. `sanitize_pos.py` exists for the (now
secondary) manual-export path to strip PII and aggregate before anything touches the repo.

**Why not a private repo or encrypted storage for the raw data:** the repo is already effectively
public-adjacent (Streamlit Community Cloud reads it, and personal GitHub repos have a way of ending
up public eventually). The simplest privacy guarantee is "the PII never enters git in the first
place," not "the PII enters git but is protected." This is also why the GH Actions hourly refresh
architecturally *cannot* touch POS data — it's not a permissions restriction that could be
misconfigured, it's a fact about what files exist on that runner at all.

**Direct consequence, learned the hard way (2026-09-23):** this design choice is exactly why the
GH Actions hourly workflow was broken for 2 months — `refresh_data.py` was rewritten to require real
POS data without anyone noticing that requirement can structurally never be satisfied on a runner
that (by design) never has it. The fix wasn't to relax the privacy boundary; it was to make
`refresh_data.py` degrade gracefully when POS data isn't available, preserving the boundary while
fixing the pipeline. See `HANDOFF.md` → Known issues for the full incident.

---

## Three-part pipeline (GH Actions + local daily cron + local weekly cron), not one automation

It would be simpler to have a single scheduled job that does everything. The reason it's split into
three is the privacy boundary above: only a machine with a live Chrome session and local POS access
can do POS-dependent work, so that work can never live in GH Actions. Splitting into:

- **GH Actions (hourly, cloud):** public signals only — weather, events, footfall, fixtures. Cheap,
  reliable, no local machine dependency, keeps forecast-relevant *external* context fresh even if
  the laptop is off.
- **Local daily cron (POS + features):** the only thing that can touch real POS data.
- **Local weekly cron (retrain):** deliberately less frequent than data refresh — retraining is
  expensive (~20-25 min) and the marginal value of daily retraining is low relative to the
  bias-correction layer already handling short-term drift between retrains.

...lets each piece run at the cadence and location that actually makes sense for what it does,
rather than forcing one location/cadence to serve three different constraints.

---

## Streamlit for the dashboard, not Dash / Flask+React / a BI tool

**Chosen:** Streamlit, deployed on Streamlit Community Cloud (free, git-push-to-deploy).

**Why:** single-developer project, Python-only stack throughout (no separate frontend
language/build step), and Streamlit Community Cloud's free tier removes hosting entirely — commit
to `main`, the app redeploys. `st.cache_data`/`st.cache_resource` handle the model/feature-loading
performance concerns cheaply. A BI tool (PowerBI, Tableau) would fight the custom prediction logic,
conformal intervals, and interactive weather/event overrides this needs; Dash or Flask+React would
require maintaining a second language/build toolchain for a solo project with no other reason to.

**Accepted limitation:** Streamlit Community Cloud has no server-side cron, which is why the GH
Actions hourly workflow pushes data commits to trigger a redeploy rather than the dashboard fetching
live on a timer — a workaround for the hosting choice, not a first-choice design.

---

## File-based model artifacts + git history, not a model registry

Models are plain files (`models/*.json`, `*.pkl`, `*.txt`) committed to git, with `retrain_history.csv`
(added 2026-09-23) as a lightweight version/metrics log. No MLflow, no DVC, no dedicated model
registry.

**Why:** at solo-project scale, git already gives versioning, diffing, and rollback (`git checkout
-- models/`) for free. A model registry's value proposition — multi-user coordination, complex
promotion workflows, serving infrastructure — doesn't apply here. `retrain_history.csv` was added
specifically to close the one real gap this leaves (no automated "is the new model actually better"
check) without adopting a much heavier tool to get it.

**Known gap, accepted for now:** there's no automated rollback if a bad model gets promoted — the
regression check in `log_retrain_history` only *warns and skips the auto-commit*; a human still has
to look. Fine for a weekly cadence a single person is watching; would need real gating before this
runs unattended at higher frequency or for a multi-venue product.

---

## What would change first for the SaaS pivot

This system is explicitly a single-venue pilot with productization as a stated future direction
(see `context.md`). The decisions above that are most likely to need revisiting first, in rough
order:

1. **File-based models + git** → would need a real model registry once there's more than one venue's
   models to track and promote independently.
2. **Shift-specific models per venue** → multi-venue would multiply the data-starvation problem in
   small shifts; likely needs either pooled cross-venue training with a venue embedding, or resolving
   the shift-specialization question (above) before replicating an unresolved architecture N times.
3. **Local-machine-dependent POS scraping** → doesn't scale past one operator's laptop; would need
   either official POS integrations per vendor or a hosted scraping service, changing the privacy
   architecture's assumptions.
4. **Streamlit Community Cloud** → free tier is fine for one pilot venue; would need real hosting
   (with server-side cron, removing the GH-Actions-commit workaround) at multi-tenant scale.
