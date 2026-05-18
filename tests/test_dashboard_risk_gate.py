"""Tests for dashboard-owned risk-score order gating."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pandas

from stock_indicator import dashboard


class FakeTradeContext:
    """Minimal Futu trade context for dashboard preview tests."""

    def __init__(self, *, total_assets: float = 78_000.0) -> None:
        self.total_assets = total_assets

    def accinfo_query(self, trd_env: Any = None) -> tuple[int, pandas.DataFrame]:
        """Return a deterministic account response."""
        return 0, pandas.DataFrame([{"total_assets": self.total_assets}])

    def position_list_query(self, trd_env: Any = None) -> tuple[int, pandas.DataFrame]:
        """Return no current positions."""
        return 0, pandas.DataFrame(columns=["code", "qty"])

    def close(self) -> None:
        """Match the Futu context close contract."""
        return None


def _write_dashboard_fixture(
    tmp_path: Path,
    *,
    signal_date: str,
    risk_score: int,
) -> tuple[Path, Path]:
    """Write a production config, risk-score CSV, and daily log."""
    config_path = tmp_path / "multi_bucket_production.json"
    risk_score_path = tmp_path / "historical_risk_scores.csv"
    log_directory = tmp_path / "logs"
    log_directory.mkdir()

    config_path.write_text(
        json.dumps(
            {
                "max_position_count": 6,
                "risk_score_gate": {
                    "csv_path": "historical_risk_scores.csv",
                    "stop_threshold": 75,
                },
            }
        ),
        encoding="utf-8",
    )
    risk_score_path.write_text(
        "year_month,duration_score,breadth_score,risk_score,recommendation,key_event,confidence\n"
        f"{signal_date[:7]},50,50,{risk_score},stop,test,H\n",
        encoding="utf-8",
    )
    (log_directory / f"{signal_date}.log").write_text(
        "[ENTRY_SIGNAL] bucket=fish_head_production "
        "strategy_id=fish_head_vacuum_turn symbol=AAA\n"
        "[ENTRY_SIGNAL] bucket=fish_tail_explore "
        "strategy_id=fish_tail_blow_off_top symbol=BBB\n"
        "accepted: [('AAA', 'fish_head_production')]\n"
        "rejected: [('BBB', 'fish_tail_explore', 'bucket_cap')]\n"
        "max_position_count=6 held_before_today=2 same_day_closes=0\n"
        "[FROZEN_TP_SL] "
        f"entry_date={signal_date} bucket=fish_head_production "
        "strategy_id=fish_head_vacuum_turn symbol=AAA "
        "dollar_volume_rank=0 tp_pct=0.050000 sl_pct=0.030000 "
        "rolling_mp=0.040000 slope_60=0.1000 near_delta=0.2000 "
        "min_hold_tp=1 disable_sl_trigger=True\n",
        encoding="utf-8",
    )
    return config_path, log_directory


def test_parse_log_separates_raw_entries_from_accepted_buys(
    tmp_path: Path,
) -> None:
    """New cron logs should show raw entries without changing order inputs."""
    _config_path, log_directory = _write_dashboard_fixture(
        tmp_path,
        signal_date="2026-05-15",
        risk_score=50,
    )

    parsed_log = dashboard._parse_log(log_directory / "2026-05-15.log")

    assert parsed_log["entry_signals"] == ["AAA", "BBB"]
    assert parsed_log["buy_actions"] == ["AAA"]
    assert parsed_log["accepted_buy_actions"] == ["AAA"]
    assert parsed_log["slot_allocation"]["accepted"] == [
        {"symbol": "AAA", "bucket": "fish_head_production"}
    ]
    assert parsed_log["slot_allocation"]["rejected"] == [
        {
            "symbol": "BBB",
            "bucket": "fish_tail_explore",
            "reason": "bucket_cap",
        }
    ]
    assert parsed_log["position_count"] == 3


def _patch_dashboard_paths(
    monkeypatch,
    *,
    config_path: Path,
    log_directory: Path,
    repository_root: Path,
) -> None:
    """Point dashboard helpers at temporary fixtures."""
    futu_module = types.SimpleNamespace(TrdEnv=types.SimpleNamespace(REAL="REAL"))
    monkeypatch.setitem(sys.modules, "futu", futu_module)
    monkeypatch.setattr(dashboard, "PRODUCTION_CONFIG_PATH", config_path)
    monkeypatch.setattr(dashboard, "REPOSITORY_ROOT", repository_root)
    monkeypatch.setattr(dashboard, "LOGS_DIRECTORY", log_directory)
    monkeypatch.setattr(dashboard, "_get_futu_trd_ctx", lambda: FakeTradeContext())
    monkeypatch.setattr(dashboard, "_get_trd_env", lambda: object())
    monkeypatch.setattr(dashboard, "_get_last_price", lambda symbol: 10.0)


def test_risk_score_gate_state_stops_when_score_reaches_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Dashboard gate should classify threshold months as stop."""
    config_path, log_directory = _write_dashboard_fixture(
        tmp_path,
        signal_date="2008-04-01",
        risk_score=100,
    )
    _patch_dashboard_paths(
        monkeypatch,
        config_path=config_path,
        log_directory=log_directory,
        repository_root=tmp_path,
    )

    gate_state = dashboard._load_risk_score_gate_state("2008-04-01")

    assert gate_state["status"] == "stop"
    assert gate_state["year_month"] == "2008-04"
    assert gate_state["risk_score"] == 100
    assert gate_state["stop_threshold"] == 75


def test_preview_orders_blocks_buy_orders_during_risk_score_stop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Risk stop should suppress dashboard BUY orders, not cron signals."""
    config_path, log_directory = _write_dashboard_fixture(
        tmp_path,
        signal_date="2008-04-01",
        risk_score=100,
    )
    _patch_dashboard_paths(
        monkeypatch,
        config_path=config_path,
        log_directory=log_directory,
        repository_root=tmp_path,
    )

    preview = dashboard.api_preview_orders()

    assert preview["risk_score_gate"]["status"] == "stop"
    assert preview["orders"] == [
        {
            "side": "BUY",
            "symbol": "AAA",
            "qty": 0,
            "ref_price": 10.0,
            "order_type": "MARKET",
            "bucket": "fish_head_production",
            "strategy_id": "fish_head_vacuum_turn",
            "tp_pct": 0.05,
            "sl_pct": 0.03,
            "dollar_volume_rank": 0,
            "status": "risk_score_stop",
            "skip_reason": "risk_score=100 >= stop_threshold=75 for 2008-04",
        }
    ]


def test_preview_orders_allows_buy_orders_when_risk_score_is_below_stop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Non-stop risk months should keep current dashboard order behavior."""
    config_path, log_directory = _write_dashboard_fixture(
        tmp_path,
        signal_date="2026-05-15",
        risk_score=50,
    )
    _patch_dashboard_paths(
        monkeypatch,
        config_path=config_path,
        log_directory=log_directory,
        repository_root=tmp_path,
    )

    preview = dashboard.api_preview_orders()

    assert preview["risk_score_gate"]["status"] == "open"
    order = preview["orders"][0]
    assert order["symbol"] == "AAA"
    assert order["qty"] == 250
    assert "status" not in order


def test_risk_score_gate_reports_missing_csv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Configured but missing risk CSV should fail closed at preview time."""
    config_path = tmp_path / "multi_bucket_production.json"
    config_path.write_text(
        json.dumps(
            {
                "risk_score_gate": {
                    "csv_path": "missing.csv",
                    "stop_threshold": 75,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "PRODUCTION_CONFIG_PATH", config_path)
    monkeypatch.setattr(dashboard, "REPOSITORY_ROOT", tmp_path)

    gate_state = dashboard._load_risk_score_gate_state("2026-05-15")

    assert gate_state["status"] == "error"
    assert "risk score CSV not found" in gate_state["reason"]



def test_current_bucket_tp_sl_uses_config_sigma_not_stale_frozen_log(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Dashboard summary should recompute bucket base TP from live config."""
    from stock_indicator import multi_bucket_today

    config_path = tmp_path / "multi_bucket_production.json"
    config_path.write_text(
        json.dumps(
            {
                "max_position_count": 6,
                "starting_cash": 60_000,
                "margin": 1.5,
                "withdraw": 0,
                "adaptive_tp_sl": {"window": 20, "sigma": 0.5, "min_samples": 5},
                "buckets": [
                    {
                        "label": "fish_head_production",
                        "strategy_id": "fish_head_vacuum_turn",
                        "dollar_volume_filter": "dollar_volume>0.02%,Top500,Pick5",
                        "sigma": 0.75,
                    },
                    {
                        "label": "fish_tail_explore",
                        "strategy_id": "fish_tail_blow_off_top",
                        "dollar_volume_filter": "dollar_volume>0.02%,Top500,Pick5",
                        "sigma": 0.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "PRODUCTION_CONFIG_PATH", config_path)
    monkeypatch.setattr(
        multi_bucket_today,
        "load_strategy_set_mapping",
        lambda: {
            "fish_head_vacuum_turn": ("fish_head_buy", "fish_head_sell"),
            "fish_tail_blow_off_top": ("fish_tail_buy", "fish_tail_sell"),
        },
    )
    monkeypatch.setattr(
        multi_bucket_today,
        "load_strategy_entry_filters",
        lambda: {},
    )
    adaptive_state = {
        "winners": [0.03, 0.05, 0.07, 0.09, 0.11],
        "losers": [-0.02, -0.03, -0.04],
    }

    bucket_states = dashboard._load_current_bucket_tp_sl(adaptive_state)
    tp_by_bucket = {state["bucket"]: state["tp_pct"] for state in bucket_states}

    assert tp_by_bucket["fish_head_production"] > tp_by_bucket["fish_tail_explore"]
    assert tp_by_bucket["fish_tail_explore"] == 0.07
