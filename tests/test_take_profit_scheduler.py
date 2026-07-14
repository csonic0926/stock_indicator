"""Tests for restart-safe New York take-profit scheduling."""

from __future__ import annotations

# TODO: review

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from stock_indicator import take_profit_scheduler

HONG_KONG_TIME_ZONE = ZoneInfo("Asia/Hong_Kong")


def test_summer_time_runs_one_minute_after_new_york_open(
    tmp_path: Path,
) -> None:
    """HKT 21:31 maps to 09:31 New York while daylight saving is active."""
    placement_times: list[str] = []

    was_invoked = take_profit_scheduler.run_due_take_profit_reconciliation(
        current_time=datetime(
            2026,
            7,
            13,
            21,
            31,
            tzinfo=HONG_KONG_TIME_ZONE,
        ),
        state_path=tmp_path / "scheduler_state.json",
        placement_function=lambda: placement_times.append("placed"),
    )

    assert was_invoked is True
    assert placement_times == ["placed"]
    assert "open_plus_1" in (tmp_path / "scheduler_state.json").read_text()


def test_winter_time_runs_one_minute_after_new_york_open(
    tmp_path: Path,
) -> None:
    """HKT 22:31 maps to 09:31 New York during standard time."""
    placement_times: list[str] = []

    was_invoked = take_profit_scheduler.run_due_take_profit_reconciliation(
        current_time=datetime(
            2026,
            12,
            14,
            22,
            31,
            tzinfo=HONG_KONG_TIME_ZONE,
        ),
        state_path=tmp_path / "scheduler_state.json",
        placement_function=lambda: placement_times.append("placed"),
    )

    assert was_invoked is True
    assert placement_times == ["placed"]
    assert "open_plus_1" in (tmp_path / "scheduler_state.json").read_text()


def test_restart_catches_up_latest_missed_checkpoint(tmp_path: Path) -> None:
    """A late restart runs the most recent checkpoint immediately."""
    placement_times: list[str] = []

    was_invoked = take_profit_scheduler.run_due_take_profit_reconciliation(
        current_time=datetime(
            2026,
            7,
            13,
            11,
            17,
            tzinfo=take_profit_scheduler.NEW_YORK_TIME_ZONE,
        ),
        state_path=tmp_path / "scheduler_state.json",
        placement_function=lambda: placement_times.append("placed"),
    )

    assert was_invoked is True
    assert placement_times == ["placed"]
    assert "open_follow_up" in (tmp_path / "scheduler_state.json").read_text()


def test_same_checkpoint_is_not_executed_twice(tmp_path: Path) -> None:
    """Per-checkpoint state prevents launchd's minute ticks duplicating work."""
    placement_times: list[str] = []
    current_time = datetime(
        2026,
        7,
        13,
        9,
        31,
        tzinfo=take_profit_scheduler.NEW_YORK_TIME_ZONE,
    )
    scheduler_state_path = tmp_path / "scheduler_state.json"

    first_invocation = take_profit_scheduler.run_due_take_profit_reconciliation(
        current_time=current_time,
        state_path=scheduler_state_path,
        placement_function=lambda: placement_times.append("placed"),
    )
    second_invocation = take_profit_scheduler.run_due_take_profit_reconciliation(
        current_time=current_time,
        state_path=scheduler_state_path,
        placement_function=lambda: placement_times.append("placed"),
    )

    assert first_invocation is True
    assert second_invocation is False
    assert placement_times == ["placed"]


def test_weekend_does_not_run_regular_checkpoint(tmp_path: Path) -> None:
    """Regular reconciliation remains a Monday-to-Friday process."""
    placement_times: list[str] = []

    was_invoked = take_profit_scheduler.run_due_take_profit_reconciliation(
        current_time=datetime(
            2026,
            7,
            11,
            9,
            40,
            tzinfo=take_profit_scheduler.NEW_YORK_TIME_ZONE,
        ),
        state_path=tmp_path / "scheduler_state.json",
        placement_function=lambda: placement_times.append("placed"),
    )

    assert was_invoked is False
    assert placement_times == []


def test_failed_checkpoint_is_not_recorded_and_can_retry(tmp_path: Path) -> None:
    """OpenD or placement failure must leave the checkpoint eligible."""
    scheduler_state_path = tmp_path / "scheduler_state.json"
    current_time = datetime(
        2026,
        7,
        13,
        9,
        31,
        tzinfo=take_profit_scheduler.NEW_YORK_TIME_ZONE,
    )

    def fail_placement() -> None:
        raise RuntimeError("OpenD unavailable")

    with pytest.raises(RuntimeError, match="OpenD unavailable"):
        take_profit_scheduler.run_due_take_profit_reconciliation(
            current_time=current_time,
            state_path=scheduler_state_path,
            placement_function=fail_placement,
        )

    assert scheduler_state_path.exists() is False
