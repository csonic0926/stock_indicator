#!/bin/bash
set -euo pipefail

# Allow the cron job to open more file descriptors
ulimit -n 4096

# Determine repository path from script location or override with environment variables
SCRIPT_DIRECTORY="$(cd "$(dirname "$0")" && pwd)"
REPOSITORY_ROOT="${REPO:-$SCRIPT_DIRECTORY}"
SOURCE_DIRECTORY="${SRC:-$REPOSITORY_ROOT/src}"
VIRTUAL_ENVIRONMENT_DIRECTORY="${VENV:-$REPOSITORY_ROOT/venv}"

# Production multi-bucket config (live).  Mirrors the promoted
# triple-bucket old-universe risk-priority setup, with data_source kept
# as "daily" so cron reads data/stock_data/ (yfinance daily cache).
PRODUCTION_CONFIG="$REPOSITORY_ROOT/data/multi_bucket_production.json"

# Set up logging directories
LOG_DIRECTORY="$REPOSITORY_ROOT/cron_logs"
DATE_LOG_DIRECTORY="$REPOSITORY_ROOT/logs"
mkdir -p "$LOG_DIRECTORY" "$DATE_LOG_DIRECTORY"

# Ensure the module can be resolved
cd "$SOURCE_DIRECTORY"

# Compute the cache refresh window. The signal date is anchored to the
# refreshed S&P 500 cache below so exchange holidays do not produce holiday
# signal logs from sparse single-symbol Yahoo rows.
REFRESH_END_DATE="$("$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -c 'from stock_indicator.daily_job import determine_latest_trading_date as determine_date;print(determine_date())')"
START_DATE="$("$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -c 'from datetime import datetime, timedelta; import sys; from stock_indicator.daily_job import YAHOO_CACHE_REFRESH_LOOKBACK_DAYS; print((datetime.fromisoformat(sys.argv[1]) - timedelta(days=YAHOO_CACHE_REFRESH_LOOKBACK_DAYS)).date().isoformat())' "$REFRESH_END_DATE")"

# Update the production daily price cache, retry missing target-date Yahoo
# rows, then record live signals.
CRON_START_EPOCH=$(date +%s)
"$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -m stock_indicator.manage update_all_data_from_yf "$START_DATE" "$REFRESH_END_DATE" >> "$LOG_DIRECTORY/cron_stdout.log" 2>&1
FIRST_DOWNLOAD_END_EPOCH=$(date +%s)
"$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -m stock_indicator.manage retry_missing_date_from_yf "$REFRESH_END_DATE" >> "$LOG_DIRECTORY/cron_stdout.log" 2>&1
UPDATE_END_EPOCH=$(date +%s)
LATEST_DATE="$("$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -c 'from stock_indicator.daily_job import determine_latest_cached_market_date, STOCK_DATA_DIRECTORY;print(determine_latest_cached_market_date(STOCK_DATA_DIRECTORY))')"

# TODO: review
# Multi-bucket signal generation. The cron publishes every tradable signal and
# starts its ADAPTIVE TP/SL virtual-history observation independently of Futu
# execution. Each [FROZEN_TP_SL] line carries the bucket-specific tp_pct /
# sl_pct used by the separate dashboard allocation layer.
# compute_adaptive_tp_sl + show_positions were the legacy single-bucket
# display path; per-bucket frozen values made them misleading (the
# global TP/SL ignored per-bucket sigma / fixed_sl), so they are no
# longer invoked.
set +e
"$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -m stock_indicator.manage \
    multi_bucket_daily_signal "$PRODUCTION_CONFIG" "$LATEST_DATE" \
    2>&1 | tee -a "$LOG_DIRECTORY/cron_stdout.log" \
    >> "$DATE_LOG_DIRECTORY/$LATEST_DATE.log"
SIGNAL_EXIT_CODE=$?
set -e
CRON_END_EPOCH=$(date +%s)
TOTAL_SECONDS=$((CRON_END_EPOCH - CRON_START_EPOCH))
UPDATE_SECONDS=$((UPDATE_END_EPOCH - CRON_START_EPOCH))
SIGNAL_SECONDS=$((CRON_END_EPOCH - UPDATE_END_EPOCH))
FIRST_DOWNLOAD_SECONDS=$((FIRST_DOWNLOAD_END_EPOCH - CRON_START_EPOCH))
RETRY_DOWNLOAD_SECONDS=$((UPDATE_END_EPOCH - FIRST_DOWNLOAD_END_EPOCH))
"$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" - \
    "$LOG_DIRECTORY/cron_runtime.csv" \
    "$LATEST_DATE" \
    "$CRON_START_EPOCH" \
    "$FIRST_DOWNLOAD_END_EPOCH" \
    "$UPDATE_END_EPOCH" \
    "$CRON_END_EPOCH" <<'PY' || true
from pathlib import Path
import sys

from stock_indicator.daily_job import record_cron_runtime

record_cron_runtime(
    Path(sys.argv[1]),
    signal_date=sys.argv[2],
    start_epoch=float(sys.argv[3]),
    first_download_end_epoch=float(sys.argv[4]),
    update_end_epoch=float(sys.argv[5]),
    end_epoch=float(sys.argv[6]),
)
PY
echo "[CRON_TIMING] date=$LATEST_DATE total=${TOTAL_SECONDS}s first_download=${FIRST_DOWNLOAD_SECONDS}s retry_download=${RETRY_DOWNLOAD_SECONDS}s total_download=${UPDATE_SECONDS}s process=${SIGNAL_SECONDS}s" >> "$LOG_DIRECTORY/cron_stdout.log" || true
exit "$SIGNAL_EXIT_CODE"
