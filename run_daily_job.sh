#!/bin/bash
set -euo pipefail

# Allow the cron job to open more file descriptors
ulimit -n 4096

# Determine repository path from script location or override with environment variables
SCRIPT_DIRECTORY="$(cd "$(dirname "$0")" && pwd)"
REPOSITORY_ROOT="${REPO:-$SCRIPT_DIRECTORY}"
SOURCE_DIRECTORY="${SRC:-$REPOSITORY_ROOT/src}"
VIRTUAL_ENVIRONMENT_DIRECTORY="${VENV:-$REPOSITORY_ROOT/venv}"

# Production fish_head config: Top500 Pick5, NO SL (sl=1.0, never triggers).
# Matches no-SL backtest baseline: CAGR 17.14%, MDD 38.01%, Calmar 0.45.
# strategy_id renamed buy3 -> fish_head_vacuum_turn in commit 9f9660d0 (2026-05-06).
ARG_LINE_1='dollar_volume>0.02%,Top500,Pick5 1.0 strategy=fish_head_vacuum_turn tp=0.078 max_pos=6 min_hold=5'

# Set up logging directories
LOG_DIRECTORY="$REPOSITORY_ROOT/cron_logs"
DATE_LOG_DIRECTORY="$REPOSITORY_ROOT/logs"
mkdir -p "$LOG_DIRECTORY" "$DATE_LOG_DIRECTORY"

# Ensure the module can be resolved
cd "$SOURCE_DIRECTORY"

# Compute the latest trading date and six-month start date
LATEST_DATE="$("$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -c 'from stock_indicator.daily_job import determine_latest_trading_date as determine_date;print(determine_date())')"
START_DATE="$("$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -c 'from datetime import datetime, timedelta; import sys; print((datetime.fromisoformat(sys.argv[1]) - timedelta(days=183)).date().isoformat())' "$LATEST_DATE")"

# Update historical data and record signals
"$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -m stock_indicator.manage update_all_data_from_yf "$START_DATE" "$LATEST_DATE" >> "$LOG_DIRECTORY/cron_stdout.log" 2>&1

{
  "$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -m stock_indicator.manage find_history_signal "$LATEST_DATE" "$ARG_LINE_1"
  echo ""
  "$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -m stock_indicator.manage compute_adaptive_tp_sl
  "$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -m stock_indicator.manage show_positions
} | tee -a "$LOG_DIRECTORY/cron_stdout.log" >> "$DATE_LOG_DIRECTORY/$LATEST_DATE.log"

# Stage 2.1 shadow mode: run the new multi-bucket today-slice generator
# alongside the live single-strategy block above. Writes to *_shadow.json
# files only — System B does NOT consume these. Failures here MUST NOT
# break the live cron run, hence `|| true` and a separate log file.
SHADOW_LOG="$LOG_DIRECTORY/shadow.log"
{
  echo ""
  echo "=== shadow run $LATEST_DATE ==="
  "$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -m stock_indicator.manage \
      multi_bucket_daily_signal data/multi_bucket_production.json \
      "$LATEST_DATE" --shadow
} >> "$SHADOW_LOG" 2>&1 || true
