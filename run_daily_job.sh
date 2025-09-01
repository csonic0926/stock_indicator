#!/bin/bash
set -euo pipefail

# Determine repository path from script location or override with environment variables
SCRIPT_DIRECTORY="$(cd "$(dirname "$0")" && pwd)"
REPOSITORY_ROOT="${REPO:-$SCRIPT_DIRECTORY}"
SOURCE_DIRECTORY="${SRC:-$REPOSITORY_ROOT/src}"
VIRTUAL_ENVIRONMENT_DIRECTORY="${VENV:-$REPOSITORY_ROOT/venv}"

# Your daily_job argument line:
# TODO: review
ARG_LINE='dollar_volume>2.14%,Top3 ema_sma_cross ema_sma_cross 0.1'

# Set up logging directory
LOG_DIRECTORY="$REPOSITORY_ROOT/cron_logs"
mkdir -p "$LOG_DIRECTORY"

# Ensure the module can be resolved
cd "$SOURCE_DIRECTORY"

# Run as a module so `from . import cron` works
# Stdout/stderr go to a rolling cron log for debugging
"$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -m stock_indicator.daily_job "$ARG_LINE" >> "$LOG_DIRECTORY/cron_stdout.log" 2>&1
