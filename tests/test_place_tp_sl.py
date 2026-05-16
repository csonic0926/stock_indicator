"""Tests for live TP/SL placement safety gates."""

from __future__ import annotations

from typing import Any

from stock_indicator import place_tp_sl


def test_entry_disables_stop_loss_trigger_when_flag_is_true() -> None:
    """SL order placement must follow the accepted entry disable flag."""

    entry_record: dict[str, Any] = {"disable_sl_trigger": True}

    assert place_tp_sl._entry_disables_stop_loss_trigger(entry_record) is True


def test_entry_keeps_stop_loss_enabled_when_flag_is_absent() -> None:
    """Legacy entries keep previous behavior unless cron backfills the flag."""

    entry_record: dict[str, Any] = {}

    assert place_tp_sl._entry_disables_stop_loss_trigger(entry_record) is False
