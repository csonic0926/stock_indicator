#!/bin/bash
set -euo pipefail

# TODO: review
# launchd calls this once per minute. Python decides whether a New York market
# checkpoint is due, including automatic daylight-saving conversion.

SCRIPT_DIRECTORY="$(cd "$(dirname "$0")" && pwd)"
REPOSITORY_ROOT="${REPO:-$SCRIPT_DIRECTORY}"
SOURCE_DIRECTORY="${SRC:-$REPOSITORY_ROOT/src}"
VIRTUAL_ENVIRONMENT_DIRECTORY="${VENV:-$REPOSITORY_ROOT/venv}"

cd "$SOURCE_DIRECTORY"
exec "$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" \
    -m stock_indicator.take_profit_scheduler
