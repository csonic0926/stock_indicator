"""Tests for the explicit ADAPTIVE TP/SL history state boundary."""

from __future__ import annotations

import json
from pathlib import Path

from stock_indicator import adaptive_tp_sl_virtual_trade_history
from stock_indicator import multi_bucket_today


# TODO: review
def test_legacy_fields_migrate_into_explicit_history_namespace() -> None:
    """Legacy generic fields retain their values under purposeful names."""

    state_document = {
        "schema_version": 2,
        "accepted_entries": [{"symbol": "AAA"}],
        "closed_trades": [{"symbol": "BBB", "raw_pct": 0.05}],
        "pending_rolling": [{"closed_date": "2026-07-10", "pct": 0.05}],
        "winners": [0.05],
        "losers": [-0.03],
        "raw_trade_profits": [0.05, -0.03],
    }

    history_state = (
        adaptive_tp_sl_virtual_trade_history.get_adaptive_tp_sl_virtual_trade_history(
            state_document
        )
    )

    assert history_state[
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_OPEN_TRADES_KEY
    ] == [{"symbol": "AAA"}]
    assert history_state[
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_CLOSED_TRADES_KEY
    ] == [{"symbol": "BBB", "raw_pct": 0.05}]
    assert history_state[
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_PENDING_RETURNS_KEY
    ] == [{"closed_date": "2026-07-10", "pct": 0.05}]
    assert history_state[
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_WINNER_RETURNS_KEY
    ] == [0.05]
    assert history_state[
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_LOSER_RETURNS_KEY
    ] == [-0.03]
    assert history_state[
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_RAW_RETURNS_KEY
    ] == [0.05, -0.03]
    for legacy_key in (
        "accepted_entries",
        "closed_trades",
        "pending_rolling",
        "winners",
        "losers",
        "raw_trade_profits",
    ):
        assert legacy_key not in state_document


def test_namespaced_history_is_authoritative_over_stale_legacy_fields() -> None:
    """A stale legacy field cannot overwrite a namespaced history value."""

    explicit_history_state = (
        adaptive_tp_sl_virtual_trade_history.empty_adaptive_tp_sl_virtual_trade_history()
    )
    explicit_history_state[
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_WINNER_RETURNS_KEY
    ] = [0.11]
    state_document = {
        "schema_version": 2,
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_TRADE_HISTORY_KEY: (
            explicit_history_state
        ),
        "winners": [9.9],
    }

    history_state = (
        adaptive_tp_sl_virtual_trade_history.get_adaptive_tp_sl_virtual_trade_history(
            state_document
        )
    )

    assert history_state[
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_WINNER_RETURNS_KEY
    ] == [0.11]
    assert "winners" not in state_document


def test_atomic_save_persists_only_explicit_history_namespace(
    tmp_path: Path,
) -> None:
    """Normal persistence completes the in-memory legacy migration."""

    state_path = tmp_path / "adaptive_state.json"
    state_document = {
        "schema_version": multi_bucket_today.SCHEMA_VERSION,
        "winners": [0.07],
    }

    multi_bucket_today.save_adaptive_tp_sl_virtual_trade_history_state_atomically(
        state_path,
        state_document,
    )

    saved_document = json.loads(state_path.read_text(encoding="utf-8"))
    saved_history_state = saved_document[
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_TRADE_HISTORY_KEY
    ]
    assert saved_history_state[
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_WINNER_RETURNS_KEY
    ] == [0.07]
    assert "winners" not in saved_document
    assert not state_path.with_suffix(".json.tmp").exists()
