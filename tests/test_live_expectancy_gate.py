"""Production integration tests for the rolling-expectancy gate."""

from __future__ import annotations

# TODO: review

import json
from pathlib import Path

import pandas
import pytest

from stock_indicator import dashboard, live_expectancy_gate, strategy


def _live_config_document(*, cold_start: str = "open") -> dict:
    """Return a small valid live global expectancy gate config."""

    return {
        "expectancy_gate": {
            "enabled": True,
            "window": 2,
            "baseline_mean": 0.0,
            "baseline_sigma": 0.1,
            "sigma_multiplier": 1.0,
            "cold_start": cold_start,
        },
        "buckets": [
            {"label": "first"},
            {"label": "second"},
        ],
    }


def _accepted_trade(
    *,
    symbol: str,
    bucket: str,
    signal_date: str,
    wr_gate_phantom: bool = False,
    expectancy_gated: bool = False,
) -> dict:
    """Return one complete dashboard-confirmed sensor observation."""

    return {
        "signal_date": signal_date,
        "bucket": bucket,
        "strategy_id": f"{bucket}_strategy",
        "symbol": symbol,
        "tp_pct": 0.50,
        "sl_pct": 0.0,
        "min_hold_tp": 1,
        "min_hold_sl": 0,
        "disable_sl_trigger": True,
        "max_hold": 20,
        "wr_gate_phantom": wr_gate_phantom,
        "expectancy_gated": expectancy_gated,
    }


def _write_price_rows(
    data_directory: Path,
    symbol: str,
    rows: list[tuple[str, float, float, float]],
) -> None:
    """Write the production daily-price columns used by adaptive replay."""

    data_directory.mkdir(parents=True, exist_ok=True)
    lines = ["Date,open,high,low"]
    lines.extend(
        f"{date_text},{open_price},{high_price},{low_price}"
        for date_text, open_price, high_price, low_price in rows
    )
    (data_directory / f"{symbol}.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("cold_start", "expected_gate_closed"),
    [("open", False), ("closed", True)],
)
def test_live_cold_start_controls_global_stop(
    cold_start: str,
    expected_gate_closed: bool,
) -> None:
    """The global stop follows cold_start before its deque is full."""

    config_document = _live_config_document(cold_start=cold_start)
    gate_config = live_expectancy_gate.parse_live_expectancy_gate_config(
        config_document
    )
    assert gate_config is not None
    state_document = live_expectancy_gate.empty_expectancy_gate_state_document(
        gate_config
    )

    sensor_state = live_expectancy_gate.build_expectancy_gate_sensor_state(
        state_document, gate_config
    )

    assert sensor_state["gate_closed"] is expected_gate_closed
    assert sensor_state["window_full"] is False


def test_live_sensor_feeds_funded_and_both_phantom_sources_strictly_before(
    tmp_path: Path,
) -> None:
    """All accepted sources share one deterministic final-outcome deque."""

    config_document = _live_config_document()
    gate_config = live_expectancy_gate.parse_live_expectancy_gate_config(
        config_document
    )
    assert gate_config is not None
    state_path = tmp_path / "expectancy_gate_state.json"
    initial_state_document = (
        live_expectancy_gate.empty_expectancy_gate_state_document(
            gate_config
        )
    )
    initial_state_document["last_evaluation_date"] = "2026-01-02"
    live_expectancy_gate.save_expectancy_gate_state_atomically(
        state_path, initial_state_document, gate_config
    )
    first_day_accepted_trades = [
        _accepted_trade(
            symbol="FUNDED",
            bucket="first",
            signal_date="2026-01-02",
        ),
        _accepted_trade(
            symbol="WRPH",
            bucket="second",
            signal_date="2026-01-02",
            wr_gate_phantom=True,
        ),
    ]
    assert live_expectancy_gate.record_accepted_trades_at_path(
        state_path=state_path,
        config_document=config_document,
        accepted_trades=first_day_accepted_trades,
    ) == 2
    # Re-confirmation is idempotent.
    assert live_expectancy_gate.record_accepted_trades_at_path(
        state_path=state_path,
        config_document=config_document,
        accepted_trades=first_day_accepted_trades,
    ) == 0
    second_day_state_document = (
        live_expectancy_gate.load_expectancy_gate_state(
            state_path, gate_config
        )
    )
    second_day_state_document["last_evaluation_date"] = "2026-01-05"
    live_expectancy_gate.save_expectancy_gate_state_atomically(
        state_path, second_day_state_document, gate_config
    )
    assert live_expectancy_gate.record_accepted_trades_at_path(
        state_path=state_path,
        config_document=config_document,
        accepted_trades=[
            _accepted_trade(
                symbol="EXPPH",
                bucket="first",
                signal_date="2026-01-05",
                expectancy_gated=True,
            )
        ],
    ) == 1

    data_directory = tmp_path / "stock_data"
    for symbol in ("FUNDED", "WRPH"):
        _write_price_rows(
            data_directory,
            symbol,
            [
                ("2026-01-05", 10.0, 10.1, 9.9),
                ("2026-01-06", 8.0, 8.2, 7.9),
            ],
        )
    _write_price_rows(
        data_directory,
        "EXPPH",
        [
            ("2026-01-06", 10.0, 10.1, 9.9),
            ("2026-01-07", 12.0, 12.1, 11.9),
        ],
    )
    adaptive_history_state = {
        "closed_trades": [
            {
                "symbol": "FUNDED",
                "bucket": "first",
                "strategy_id": "first_strategy",
                "entry_date": "2026-01-02",
                "exit_date": "2026-01-06",
                "raw_pct": -0.20,
            },
            {
                "symbol": "WRPH",
                "bucket": "second",
                "strategy_id": "second_strategy",
                "entry_date": "2026-01-02",
                "exit_date": "2026-01-06",
                "raw_pct": -0.20,
            },
            {
                "symbol": "EXPPH",
                "bucket": "first",
                "strategy_id": "first_strategy",
                "entry_date": "2026-01-05",
                "exit_date": "2026-01-07",
                "raw_pct": 0.20,
            },
        ]
    }

    state_document = live_expectancy_gate.load_expectancy_gate_state(
        state_path, gate_config
    )
    assert live_expectancy_gate.advance_expectancy_gate_state(
        state_document=state_document,
        gate_config=gate_config,
        adaptive_history_state=adaptive_history_state,
        evaluation_date=pandas.Timestamp("2026-01-06"),
        data_directory=data_directory,
    ) == 0

    assert live_expectancy_gate.advance_expectancy_gate_state(
        state_document=state_document,
        gate_config=gate_config,
        adaptive_history_state=adaptive_history_state,
        evaluation_date=pandas.Timestamp("2026-01-07"),
        data_directory=data_directory,
    ) == 2
    sensor_state = live_expectancy_gate.build_expectancy_gate_sensor_state(
        state_document, gate_config
    )
    assert sensor_state["rolling_mean"] == pytest.approx(-0.20)
    assert sensor_state["gate_closed"] is True

    assert live_expectancy_gate.advance_expectancy_gate_state(
        state_document=state_document,
        gate_config=gate_config,
        adaptive_history_state=adaptive_history_state,
        evaluation_date=pandas.Timestamp("2026-01-08"),
        data_directory=data_directory,
    ) == 1
    # The deque is global FIFO: the later expectancy phantom replaces the
    # first same-day close and reopens the strict stop threshold at equality.
    assert [
        outcome["symbol"] for outcome in state_document["rolling_outcomes"]
    ] == ["WRPH", "EXPPH"]
    sensor_state = live_expectancy_gate.build_expectancy_gate_sensor_state(
        state_document, gate_config
    )
    assert sensor_state["rolling_mean"] == pytest.approx(0.0)
    assert sensor_state["gate_closed"] is False


def test_live_outcome_replay_resets_max_hold_on_cross_bucket_signal(
    tmp_path: Path,
) -> None:
    """A raw signal from any bucket re-anchors a configured live hold."""

    data_directory = tmp_path / "stock_data"
    _write_price_rows(
        data_directory,
        "RESET",
        [
            ("2026-01-05", 10.0, 10.1, 9.9),
            ("2026-01-06", 10.0, 10.1, 9.9),
            ("2026-01-07", 10.0, 10.1, 9.9),
            ("2026-01-08", 10.0, 10.1, 9.9),
            ("2026-01-09", 10.0, 10.1, 9.9),
        ],
    )
    pending_trade = _accepted_trade(
        symbol="RESET",
        bucket="first",
        signal_date="2026-01-02",
    )
    pending_trade.update({
        "entry_price": 10.0,
        "fill_date": "2026-01-05",
        "max_hold": 2,
        "reset_hold_on_reentry_signal": True,
    })

    outcome_without_reset = (
        live_expectancy_gate.resolve_trade_outcome_before_date(
            pending_trade=dict(pending_trade),
            evaluation_date=pandas.Timestamp("2026-01-09"),
            data_directory=data_directory,
            closed_trade_by_identity={},
            reentry_signal_dates_by_symbol={},
        )
    )
    outcome_with_reset = (
        live_expectancy_gate.resolve_trade_outcome_before_date(
            pending_trade=dict(pending_trade),
            evaluation_date=pandas.Timestamp("2026-01-09"),
            data_directory=data_directory,
            closed_trade_by_identity={},
            reentry_signal_dates_by_symbol={
                "RESET": {pandas.Timestamp("2026-01-07")}
            },
        )
    )

    assert outcome_without_reset is not None
    assert outcome_without_reset["exit_reason"] == "max_hold"
    assert outcome_with_reset is None


def test_dashboard_fails_closed_without_enabled_sensor_heartbeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An enabled production gate cannot silently become an open gate."""

    config_path = tmp_path / "multi_bucket_production.json"
    config_path.write_text(
        json.dumps(_live_config_document()), encoding="utf-8"
    )
    monkeypatch.setattr(dashboard, "PRODUCTION_CONFIG_PATH", config_path)
    state_path = tmp_path / "expectancy_gate_state.json"
    gate_config = live_expectancy_gate.parse_live_expectancy_gate_config(
        _live_config_document()
    )
    assert gate_config is not None
    state_document = (
        live_expectancy_gate.empty_expectancy_gate_state_document(
            gate_config
        )
    )
    state_document["last_evaluation_date"] = "2026-01-07"
    live_expectancy_gate.save_expectancy_gate_state_atomically(
        state_path, state_document, gate_config
    )
    monkeypatch.setattr(
        dashboard, "EXPECTANCY_GATE_STATE_PATH", state_path
    )

    decision = dashboard._load_expectancy_gate_decision({
        "date": "2026-01-07"
    })

    assert decision["status"] == "error"
    assert "no sensor heartbeat" in decision["reason"]


def test_dashboard_accepts_matching_once_per_day_sensor_heartbeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typed matching cron output controls the production stop."""

    config_path = tmp_path / "multi_bucket_production.json"
    config_path.write_text(
        json.dumps(_live_config_document()), encoding="utf-8"
    )
    monkeypatch.setattr(dashboard, "PRODUCTION_CONFIG_PATH", config_path)
    state_path = tmp_path / "expectancy_gate_state.json"
    gate_config = live_expectancy_gate.parse_live_expectancy_gate_config(
        _live_config_document()
    )
    assert gate_config is not None
    state_document = (
        live_expectancy_gate.empty_expectancy_gate_state_document(
            gate_config
        )
    )
    state_document["rolling_outcomes"] = [
        {"percentage_change": -0.2},
        {"percentage_change": -0.2},
    ]
    state_document["last_evaluation_date"] = "2026-01-07"
    live_expectancy_gate.save_expectancy_gate_state_atomically(
        state_path, state_document, gate_config
    )
    monkeypatch.setattr(
        dashboard, "EXPECTANCY_GATE_STATE_PATH", state_path
    )
    log_path = tmp_path / "2026-01-07.log"
    log_path.write_text(
        "[EXPECTANCY_GATE_SENSOR] status=ready mean=-0.200000 "
        "stop_threshold=-0.100000 gate_closed=True window=2/2 "
        "window_full=True open_pending=1 fed_this_run=2 "
        "closed_episodes=0 expectancy_gated_trades=0\n",
        encoding="utf-8",
    )

    parsed_log = dashboard._parse_log(log_path)
    decision = dashboard._load_expectancy_gate_decision(parsed_log)

    assert decision["status"] == "closed"
    assert decision["gate_closed"] is True


def test_dashboard_labels_matching_cold_start_as_warming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An incomplete deque is open by policy, not by a threshold reading."""

    config_document = _live_config_document(cold_start="open")
    config_path = tmp_path / "multi_bucket_production.json"
    config_path.write_text(json.dumps(config_document), encoding="utf-8")
    gate_config = live_expectancy_gate.parse_live_expectancy_gate_config(
        config_document
    )
    assert gate_config is not None
    state_document = (
        live_expectancy_gate.empty_expectancy_gate_state_document(
            gate_config
        )
    )
    state_document["last_evaluation_date"] = "2026-01-07"
    state_path = tmp_path / "expectancy_gate_state.json"
    live_expectancy_gate.save_expectancy_gate_state_atomically(
        state_path,
        state_document,
        gate_config,
    )
    monkeypatch.setattr(dashboard, "PRODUCTION_CONFIG_PATH", config_path)
    monkeypatch.setattr(dashboard, "EXPECTANCY_GATE_STATE_PATH", state_path)
    log_path = tmp_path / "2026-01-07.log"
    log_path.write_text(
        "[EXPECTANCY_GATE_SENSOR] status=ready mean=None "
        "stop_threshold=-0.100000 gate_closed=False window=0/2 "
        "window_full=False open_pending=0 fed_this_run=0 "
        "closed_episodes=0 expectancy_gated_trades=0\n",
        encoding="utf-8",
    )

    decision = dashboard._load_expectancy_gate_decision(
        dashboard._parse_log(log_path)
    )

    assert decision["status"] == "open"
    assert decision["cold_start"] == "open"
    assert decision["window"] == "0/2"
    assert decision["reason"] == "warming up; cold_start=open"


def test_dashboard_shows_acceptances_added_after_daily_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-cron acceptances update counters without changing the regime."""

    config_document = _live_config_document(cold_start="open")
    config_path = tmp_path / "multi_bucket_production.json"
    config_path.write_text(json.dumps(config_document), encoding="utf-8")
    gate_config = live_expectancy_gate.parse_live_expectancy_gate_config(
        config_document
    )
    assert gate_config is not None
    state_document = (
        live_expectancy_gate.empty_expectancy_gate_state_document(
            gate_config
        )
    )
    state_document["last_evaluation_date"] = "2026-01-07"
    state_path = tmp_path / "expectancy_gate_state.json"
    live_expectancy_gate.save_expectancy_gate_state_atomically(
        state_path,
        state_document,
        gate_config,
    )
    assert live_expectancy_gate.record_accepted_trades_at_path(
        state_path=state_path,
        config_document=config_document,
        accepted_trades=[
            _accepted_trade(
                symbol="AAA",
                bucket="first",
                signal_date="2026-01-07",
            )
        ],
    ) == 1
    monkeypatch.setattr(dashboard, "PRODUCTION_CONFIG_PATH", config_path)
    monkeypatch.setattr(dashboard, "EXPECTANCY_GATE_STATE_PATH", state_path)
    log_path = tmp_path / "2026-01-07.log"
    log_path.write_text(
        "[EXPECTANCY_GATE_SENSOR] status=ready mean=None "
        "stop_threshold=-0.100000 gate_closed=False window=0/2 "
        "window_full=False open_pending=0 fed_this_run=0 "
        "closed_episodes=0 expectancy_gated_trades=0\n",
        encoding="utf-8",
    )

    decision = dashboard._load_expectancy_gate_decision(
        dashboard._parse_log(log_path)
    )

    assert decision["status"] == "open"
    assert decision["window"] == "0/2"
    assert decision["open_pending_at_evaluation"] == 0
    assert decision["open_pending"] == 1


def test_dashboard_rejects_stale_log_after_sensor_state_advances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cron/dashboard race cannot apply yesterday's regime to today."""

    config_document = _live_config_document()
    config_path = tmp_path / "multi_bucket_production.json"
    config_path.write_text(json.dumps(config_document), encoding="utf-8")
    gate_config = live_expectancy_gate.parse_live_expectancy_gate_config(
        config_document
    )
    assert gate_config is not None
    state_document = (
        live_expectancy_gate.empty_expectancy_gate_state_document(
            gate_config
        )
    )
    state_document["last_evaluation_date"] = "2026-01-08"
    state_path = tmp_path / "expectancy_gate_state.json"
    live_expectancy_gate.save_expectancy_gate_state_atomically(
        state_path, state_document, gate_config
    )
    monkeypatch.setattr(dashboard, "PRODUCTION_CONFIG_PATH", config_path)
    monkeypatch.setattr(dashboard, "EXPECTANCY_GATE_STATE_PATH", state_path)

    decision = dashboard._load_expectancy_gate_decision({
        "date": "2026-01-07",
        "expectancy_gate_sensor": {
            "status": "ready",
        },
    })

    assert decision["status"] == "error"
    assert "state date does not match log date" in decision["reason"]


def test_non_finite_live_threshold_inputs_are_rejected() -> None:
    """JSON Infinity cannot bypass numeric fail-closed validation."""

    for field_name in ("baseline_sigma", "sigma_multiplier"):
        raw_config = {
            "enabled": True,
            "window": 2,
            "baseline_mean": 0.0,
            "baseline_sigma": 0.1,
            "sigma_multiplier": 1.0,
            field_name: float("inf"),
        }
        with pytest.raises(ValueError):
            strategy.parse_expectancy_gate_config(raw_config)
