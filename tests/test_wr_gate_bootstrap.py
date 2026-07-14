"""Tests for the WR-gate cold-start install command (merge_wr_gate_bootstrap).

The command must be provably non-destructive: it copies the universal
sensor from the sim export, derives the pending list from the namespaced
ADAPTIVE TP/SL virtual trade history, and leaves every other key byte-identical.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from stock_indicator import adaptive_tp_sl_virtual_trade_history
from stock_indicator import manage as manage_module


def _live_state() -> dict:
    """A representative live adaptive_state.json with a rolling pool, closed
    trades, and open positions across buckets."""
    adaptive_history = {
        "schema_version": 1,
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_WINNER_RETURNS_KEY: [
            0.058,
            0.084,
        ],
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_LOSER_RETURNS_KEY: [
            -0.011,
            -0.037,
        ],
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_PENDING_RETURNS_KEY: [],
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_RAW_RETURNS_KEY: [
            0.058,
            -0.011,
            0.084,
        ],
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_CLOSED_TRADES_KEY: [
            {
                "symbol": "CVX",
                "bucket": "fish_head_production",
                "raw_pct": 0.058,
            },
        ],
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_OPEN_TRADES_KEY: [
            {"symbol": "NFLX", "bucket": "fish_head_production",
             "entry_date": "2026-06-04", "tp_pct": 0.07, "max_hold": None},
            {"symbol": "MU", "bucket": "fish_tail_production",
             "entry_date": "2026-06-08", "tp_pct": 0.21, "max_hold": 7},
            {"symbol": "PG", "bucket": "fish_tail_production",
             "entry_date": "2026-06-10", "tp_pct": 0.05, "max_hold": 7},
        ],
    }
    return {
        "schema_version": 2,
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_TRADE_HISTORY_KEY: (
            adaptive_history
        ),
    }


def _export() -> dict:
    """A sim export carrying the warm sensor plus keys that must NOT leak
    (winner returns here differ from the live sample; pending here is the sim's,
    not this machine's)."""
    return {
        "schema_version": 2,
        "captured_at_date": "2026-05-01",
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_TRADE_HISTORY_KEY: {
            "schema_version": 1,
            adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_WINNER_RETURNS_KEY: [
                9.9,
                9.9,
            ],
        },
        "wr_gate_sensor": {
            "cross_ema": 0.561,
            "cross_window": [1.0] * 12,
            "winner_pcts": [0.05] * 7,
            "loser_pcts": [0.04] * 5,
        },
        "wr_gate_pending_ft": [{"symbol": "STALE", "signal_date": "2026-04-20"}],
    }


def _run(state_path: Path, export_path: Path) -> str:
    buffer = io.StringIO()
    shell = manage_module.StockShell(stdout=buffer)
    shell.onecmd(f"merge_wr_gate_bootstrap {export_path} --state {state_path}")
    return buffer.getvalue()


def test_merge_is_non_destructive_and_derives_pending(tmp_path: Path) -> None:
    state_path = tmp_path / "adaptive_state.json"
    export_path = tmp_path / "export.json"
    original = _live_state()
    state_path.write_text(json.dumps(original), encoding="utf-8")
    export_path.write_text(json.dumps(_export()), encoding="utf-8")

    _run(state_path, export_path)
    merged = json.loads(state_path.read_text(encoding="utf-8"))

    bootstrap_keys = {"wr_gate_sensor", "wr_gate_pending_ft"}
    preserved_before = {k: v for k, v in original.items() if k not in bootstrap_keys}
    preserved_after = {k: v for k, v in merged.items() if k not in bootstrap_keys}
    assert preserved_after == preserved_before  # byte-identical preservation
    # The export's winner returns must NOT leak over the live sample.
    merged_adaptive_history = merged[
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_TRADE_HISTORY_KEY
    ]
    assert merged_adaptive_history[
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_WINNER_RETURNS_KEY
    ] == [0.058, 0.084]
    # Sensor copied verbatim.
    assert merged["wr_gate_sensor"]["cross_ema"] == 0.561
    assert len(merged["wr_gate_sensor"]["cross_window"]) == 12
    # Pending DERIVED from this machine's open fish_tail_production entries,
    # NOT the export's stale 'STALE' position.
    assert [p["symbol"] for p in merged["wr_gate_pending_ft"]] == ["MU", "PG"]
    assert merged["wr_gate_pending_ft"][0] == {
        "symbol": "MU", "signal_date": "2026-06-08",
        "tp_pct": 0.21, "min_hold_tp": 1, "max_hold": 7,
    }


def test_merge_is_idempotent(tmp_path: Path) -> None:
    state_path = tmp_path / "adaptive_state.json"
    export_path = tmp_path / "export.json"
    state_path.write_text(json.dumps(_live_state()), encoding="utf-8")
    export_path.write_text(json.dumps(_export()), encoding="utf-8")

    _run(state_path, export_path)
    first = state_path.read_text(encoding="utf-8")
    _run(state_path, export_path)
    second = state_path.read_text(encoding="utf-8")
    assert first == second


def test_merge_aborts_when_export_lacks_sensor(tmp_path: Path) -> None:
    state_path = tmp_path / "adaptive_state.json"
    export_path = tmp_path / "export.json"
    original = _live_state()
    state_path.write_text(json.dumps(original), encoding="utf-8")
    export_path.write_text(
        json.dumps({
            adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_TRADE_HISTORY_KEY: {
                "schema_version": 1,
                adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_WINNER_RETURNS_KEY: [1],
            },
        }),
        encoding="utf-8",
    )

    output = _run(state_path, export_path)
    assert "no wr_gate_sensor" in output.lower()
    # State unchanged.
    assert json.loads(state_path.read_text(encoding="utf-8")) == original


def test_merge_aborts_when_state_missing(tmp_path: Path) -> None:
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(_export()), encoding="utf-8")
    missing_state = tmp_path / "nope.json"

    output = _run(missing_state, export_path)
    assert "not found" in output.lower()
    assert not missing_state.exists()
