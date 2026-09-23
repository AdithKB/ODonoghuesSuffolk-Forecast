#!/usr/bin/env bash
# Weekly model retrain — keeps the model current so recency features (trend_21d,
# event_impact_score_max_28d, bias-correction) don't have to compensate for a
# model that's months stale, and so newer regime-change examples (like the
# Sep 2026 decline) accumulate over time instead of staying a one-off.
# Cron: 0 5 * * 1 /Users/adithkb/Ireland/PersonalProjects/CustomerFootfallPrediction/weekly_retrain.sh
#       (Monday 05:00, before the daily_refresh.sh 06:00 run)

set -euo pipefail

PYTHON=/opt/anaconda3/bin/python3
ROOT=/Users/adithkb/Ireland/PersonalProjects/CustomerFootfallPrediction
LOG="$ROOT/data/weekly_retrain.log"

cd "$ROOT"

echo "--- $(date '+%Y-%m-%d %H:%M:%S') weekly_retrain START ---" >> "$LOG"

# 1. Make sure POS data + features are current before retraining
"$PYTHON" scripts/scrape_titanbi.py --resume >> "$LOG" 2>&1
"$PYTHON" refresh_data.py --retrain >> "$LOG" 2>&1

# 2. Only commit+push if src/model.py's retrain-history check (see
#    log_retrain_history()) didn't flag a regression on the most recent
#    walk-forward fold. A REGRESSED verdict leaves the working tree dirty for
#    manual review instead of auto-shipping a worse model to the live dashboard.
if tail -200 "$LOG" | grep -q "REGRESSED"; then
    echo "--- $(date '+%Y-%m-%d %H:%M:%S') REGRESSION DETECTED — not auto-committing, leaving for manual review ---" >> "$LOG"
else
    git add data/ models/ >> "$LOG" 2>&1
    if ! git diff --staged --quiet; then
        git commit -m "chore: weekly model retrain [skip ci]" >> "$LOG" 2>&1
        git push >> "$LOG" 2>&1
    else
        echo "--- $(date '+%Y-%m-%d %H:%M:%S') No changes to commit ---" >> "$LOG"
    fi
fi

echo "--- $(date '+%Y-%m-%d %H:%M:%S') weekly_retrain END ---" >> "$LOG"
