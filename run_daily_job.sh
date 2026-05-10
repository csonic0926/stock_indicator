#!/bin/bash
set -euo pipefail

# Allow the cron job to open more file descriptors
ulimit -n 4096

# Determine repository path from script location or override with environment variables
SCRIPT_DIRECTORY="$(cd "$(dirname "$0")" && pwd)"
REPOSITORY_ROOT="${REPO:-$SCRIPT_DIRECTORY}"
SOURCE_DIRECTORY="${SRC:-$REPOSITORY_ROOT/src}"
VIRTUAL_ENVIRONMENT_DIRECTORY="${VENV:-$REPOSITORY_ROOT/venv}"

# Production multi-bucket config (live).  fish_head_vacuum_turn +
# fish_tail_blow_off_top sharing max_position_count=9 / starting 90k /
# margin 1.5.  See data/multi_bucket_production.json for per-bucket
# narrative-correct flags (pre_cross_signal_lookback / tp_regime_adjust /
# tp_slope_amplify).
PRODUCTION_CONFIG="$REPOSITORY_ROOT/data/multi_bucket_production.json"

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
  "$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -m stock_indicator.manage \
      multi_bucket_daily_signal "$PRODUCTION_CONFIG" "$LATEST_DATE"
  echo ""
  "$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -m stock_indicator.manage compute_adaptive_tp_sl
  "$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -m stock_indicator.manage show_positions
} | tee -a "$LOG_DIRECTORY/cron_stdout.log" >> "$DATE_LOG_DIRECTORY/$LATEST_DATE.log"
