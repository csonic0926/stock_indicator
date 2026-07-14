"""Run live take-profit reconciliation at US-session checkpoints.

The scheduler is launched once per minute by launchd.  It converts the current
time to ``America/New_York`` so daylight-saving changes require no HKT cron
edits.  Durable checkpoint state makes each reconciliation idempotent while
still allowing a missed checkpoint to run immediately after sleep or restart.
"""

from __future__ import annotations

# TODO: review

import json
import logging
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from stock_indicator import place_tp_sl

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCHEDULER_STATE_PATH = (
    PROJECT_ROOT / "data" / "live_state" / "take_profit_scheduler_state.json"
)
NEW_YORK_TIME_ZONE = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class TakeProfitCheckpoint:
    """One idempotent reconciliation checkpoint in New York local time."""

    name: str
    due_time: time


WEEKDAY_CHECKPOINTS = (
    TakeProfitCheckpoint("pre_open", time(9, 20)),
    TakeProfitCheckpoint("open_plus_1", time(9, 31)),
    TakeProfitCheckpoint("open_plus_3", time(9, 33)),
    TakeProfitCheckpoint("open_plus_5", time(9, 35)),
    TakeProfitCheckpoint("open_plus_10", time(9, 40)),
    TakeProfitCheckpoint("open_follow_up", time(10, 0)),
    TakeProfitCheckpoint("midday_reconciliation", time(12, 0)),
    TakeProfitCheckpoint("late_session_reconciliation", time(15, 30)),
    TakeProfitCheckpoint("post_close_reconciliation", time(16, 10)),
)


def _latest_due_checkpoint(
    current_time: datetime,
) -> TakeProfitCheckpoint | None:
    """Return the latest due weekday checkpoint in New York time."""
    new_york_time = current_time.astimezone(NEW_YORK_TIME_ZONE)
    if new_york_time.weekday() >= 5:
        return None

    due_checkpoints = [
        checkpoint
        for checkpoint in WEEKDAY_CHECKPOINTS
        if checkpoint.due_time <= new_york_time.time().replace(tzinfo=None)
    ]
    if not due_checkpoints:
        return None
    return due_checkpoints[-1]


def _checkpoint_key(
    current_time: datetime,
    checkpoint: TakeProfitCheckpoint,
) -> str:
    """Build the durable key for one New York market-date checkpoint."""
    market_date_text = current_time.astimezone(NEW_YORK_TIME_ZONE).date().isoformat()
    return f"{market_date_text}:{checkpoint.name}"


def _load_scheduler_state(state_path: Path) -> dict[str, Any]:
    """Load scheduler state, treating a missing or corrupt file as empty."""
    try:
        loaded_state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded_state if isinstance(loaded_state, dict) else {}


def _save_scheduler_state(state_path: Path, state: dict[str, Any]) -> None:
    """Atomically persist the last completed scheduler checkpoint."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(state, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(state_path)


def run_due_take_profit_reconciliation(
    *,
    current_time: datetime | None = None,
    state_path: Path = SCHEDULER_STATE_PATH,
    placement_function: Callable[[], None] | None = None,
) -> bool:
    """Run a due checkpoint once and return whether placement was invoked.

    A launch after sleep or shutdown sees the latest missed checkpoint and
    executes it immediately.  Later checkpoints deliberately re-verify broker
    coverage, while ``place_tp_sl`` itself prevents duplicate live orders.
    """
    evaluation_time = current_time or datetime.now(tz=NEW_YORK_TIME_ZONE)
    if evaluation_time.tzinfo is None:
        raise ValueError("current_time must be timezone-aware")

    due_checkpoint = _latest_due_checkpoint(evaluation_time)
    if due_checkpoint is None:
        return False

    due_checkpoint_key = _checkpoint_key(evaluation_time, due_checkpoint)
    scheduler_state = _load_scheduler_state(state_path)
    if scheduler_state.get("last_completed_checkpoint") == due_checkpoint_key:
        return False

    new_york_time = evaluation_time.astimezone(NEW_YORK_TIME_ZONE)
    LOGGER.info(
        "Running take-profit checkpoint %s at %s",
        due_checkpoint_key,
        new_york_time.isoformat(),
    )
    if placement_function is None:
        place_tp_sl.main(require_complete_take_profit_coverage=True)
    else:
        placement_function()
    _save_scheduler_state(
        state_path,
        {
            "last_completed_checkpoint": due_checkpoint_key,
            "completed_at_new_york": new_york_time.isoformat(),
        },
    )
    return True


def main() -> None:
    """Run one launchd scheduler tick."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        run_due_take_profit_reconciliation()
    except Exception:
        LOGGER.exception("Take-profit scheduler checkpoint failed")
        raise


if __name__ == "__main__":
    main()
