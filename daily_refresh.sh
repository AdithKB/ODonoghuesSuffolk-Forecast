#!/usr/bin/env bash
# Daily data refresh — scrape yesterday's POS data then rebuild features.
# Cron: 0 6 * * * /Users/adithkb/Ireland/PersonalProjects/CustomerFootfallPrediction/daily_refresh.sh

set -euo pipefail

PYTHON=/opt/anaconda3/bin/python3
ROOT=/Users/adithkb/Ireland/PersonalProjects/CustomerFootfallPrediction
LOG="$ROOT/data/cron.log"

cd "$ROOT"

echo "--- $(date '+%Y-%m-%d %H:%M:%S') daily_refresh START ---" >> "$LOG"

# 1. Scrape yesterday's Titan POS data (safe_lag guard handles the 2am close)
"$PYTHON" scripts/scrape_titanbi.py --resume >> "$LOG" 2>&1

# 2. Rebuild weather, enrichment, footfall, features.parquet
"$PYTHON" refresh_data.py >> "$LOG" 2>&1

echo "--- $(date '+%Y-%m-%d %H:%M:%S') daily_refresh END ---" >> "$LOG"
