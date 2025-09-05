#!/bin/bash
set -euo pipefail

# Determine repository path from script location or override with environment variables
SCRIPT_DIRECTORY="$(cd "$(dirname "$0")" && pwd)"
REPOSITORY_ROOT="${REPO:-$SCRIPT_DIRECTORY}"
SOURCE_DIRECTORY="${SRC:-$REPOSITORY_ROOT/src}"
VIRTUAL_ENVIRONMENT_DIRECTORY="${VENV:-$REPOSITORY_ROOT/venv}"

# Your daily_job argument line:
ARG_LINE='dollar_volume>0.05%,Top30,Pick10 strategy=s4 1.0'

# Set up logging directory
LOG_DIRECTORY="$REPOSITORY_ROOT/cron_logs"
mkdir -p "$LOG_DIRECTORY"

# Determine date range for Yahoo Finance refresh
STOCK_DATA_DIRECTORY="$REPOSITORY_ROOT/data/stock_data"
TODAY="$(date +%F)"
LAST_CACHED_DATE="$("$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" <<PY
import pandas
from pathlib import Path
from datetime import date

stock_data_directory = Path("$STOCK_DATA_DIRECTORY")
latest_date = None
if stock_data_directory.exists():
    for csv_path in stock_data_directory.glob("*.csv"):
        try:
            frame = pandas.read_csv(csv_path, usecols=[0], parse_dates=[0])
        except Exception:
            continue
        if frame.empty:
            continue
        value = frame.iloc[-1, 0]
        if hasattr(value, "date"):
            current_date = value.date()
            if latest_date is None or current_date > latest_date:
                latest_date = current_date
if latest_date is None:
    latest_date = date.fromisoformat("2019-01-01")
print(latest_date.isoformat())
PY
)"

# Ensure the module can be resolved
cd "$SOURCE_DIRECTORY"

# Refresh local data cache before running the daily job
"$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -m stock_indicator.manage update_all_data_from_yf "$LAST_CACHED_DATE" "$TODAY" >> "$LOG_DIRECTORY/cron_stdout.log" 2>&1

# Run as a module so `from . import cron` works
# Stdout/stderr go to a rolling cron log for debugging
# Disable automatic breakpoints during the daily job run
PYTHONBREAKPOINT=0 "$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -m stock_indicator.daily_job "$ARG_LINE" >> "$LOG_DIRECTORY/cron_stdout.log" 2>&1
