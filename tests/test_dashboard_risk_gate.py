"""Tests for dashboard-owned risk-score order gating."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pandas

from stock_indicator import dashboard
from stock_indicator.futu_trade_metadata import parse_futu_order_remark


class FakeTradeContext:
    """Minimal Futu trade context for dashboard preview tests."""

    def __init__(
        self,
        *,
        total_assets: float = 78_000.0,
        positions: list[dict[str, Any]] | None = None,
        deals: list[dict[str, Any]] | None = None,
        orders: list[dict[str, Any]] | None = None,
    ) -> None:
        self.total_assets = total_assets
        self.positions = positions or []
        self.deals = deals or []
        self.orders = orders or []
        self.is_closed = False

    def _raise_if_closed(self) -> None:
        """Fail tests if dashboard queries history after closing Futu context."""
        if self.is_closed:
            raise RuntimeError("Futu context used after close")

    def accinfo_query(self, trd_env: Any = None) -> tuple[int, pandas.DataFrame]:
        """Return a deterministic account response."""
        self._raise_if_closed()
        return 0, pandas.DataFrame([{"total_assets": self.total_assets}])

    def position_list_query(self, trd_env: Any = None) -> tuple[int, pandas.DataFrame]:
        """Return deterministic current positions."""
        self._raise_if_closed()
        return 0, pandas.DataFrame(self.positions, columns=["code", "qty"])

    def history_deal_list_query(
        self,
        code: str = "",
        start: str = "",
        end: str = "",
        trd_env: Any = None,
    ) -> tuple[int, pandas.DataFrame]:
        """Return deterministic historical Futu deals."""
        self._raise_if_closed()
        return 0, pandas.DataFrame(
            self.deals,
            columns=["code", "trd_side", "qty", "create_time", "order_id"],
        )

    def history_order_list_query(
        self,
        status_filter_list: list[Any] | None = None,
        code: str = "",
        start: str = "",
        end: str = "",
        trd_env: Any = None,
    ) -> tuple[int, pandas.DataFrame]:
        """Return deterministic historical Futu orders."""
        self._raise_if_closed()
        return 0, pandas.DataFrame(
            self.orders,
            columns=["order_id", "remark"],
        )

    def close(self) -> None:
        """Match the Futu context close contract."""
        self.is_closed = True



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
        "[ROLLING_TP_SL_STATE] winners=20 losers=20 pending_rolling=0 closed_trades=40\n"
        "[BUCKET_TP_SL] date=2026-05-15 bucket=fish_head_production "
        "strategy_id=fish_head_vacuum_turn tp_pct=0.074200 sl_pct=0.043400 "
        "rolling_mp=0.055000 rolling_ml=-0.030000 min_hold_tp=1 "
        "min_hold_sl=1 disable_sl_trigger=True max_hold=None "
        "reset_hold_on_reentry_signal=False\n"
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
    assert parsed_log["rolling_tp_sl_state"] == {
        "winners": 20,
        "losers": 20,
        "pending_rolling": 0,
        "closed_trades": 40,
    }
    assert parsed_log["bucket_tp_sl"] == [
        {
            "date": "2026-05-15",
            "bucket": "fish_head_production",
            "strategy_id": "fish_head_vacuum_turn",
            "tp_pct": 0.0742,
            "sl_pct": 0.0434,
            "rolling_mp": 0.055,
            "rolling_ml": -0.03,
            "min_hold_tp": 1,
            "min_hold_sl": 1,
            "disable_sl_trigger": True,
            "max_hold": None,
            "reset_hold_on_reentry_signal": False,
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
    monkeypatch.setattr(dashboard, "DATA_DIRECTORY", repository_root)
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
            "disable_sl_trigger": True,
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

def test_dashboard_buy_remark_uses_compact_v2_schema() -> None:
    """BUY remarks should fit Futu limits and round-trip live metadata."""
    order = {
        "side": "BUY",
        "strategy_id": "fish_head_b30_35",
        "tp_pct": 0.0658,
        "sl_pct": 0.0417,
        "min_hold_sl": 1,
        "disable_sl_trigger": True,
        "max_hold": 14,
        "reset_hold_on_reentry_signal": True,
    }

    remark_text = dashboard._format_dashboard_order_remark(order)
    metadata = parse_futu_order_remark(remark_text)

    assert remark_text == "si2|s=b|tp=658|sl=417|ms=1|ds=1|mh=14|rr=1"
    assert len(remark_text.encode("utf-8")) <= 64
    assert metadata["strategy_id"] == "fish_head_b30_35"
    assert metadata["tp_pct"] == 0.0658
    assert metadata["sl_pct"] == 0.0417
    assert metadata["min_hold_sl"] == 1
    assert metadata["disable_sl_trigger"] is True
    assert metadata["max_hold"] == 14
    assert metadata["reset_hold_on_reentry_signal"] is True
    assert metadata["supports_tp_sl"] is True


def test_legacy_remark_is_max_hold_only() -> None:
    """Legacy remarks must not be treated as TP/SL metadata."""
    metadata = parse_futu_order_remark(
        "si|sid=fish_head_b30_35|b=fish_head_b30_35|mh=10|rr=0"
    )

    assert metadata["strategy_id"] == "fish_head_b30_35"
    assert metadata["max_hold"] == 10
    assert metadata["supports_tp_sl"] is False
    assert "tp_pct" not in metadata


def test_preview_orders_adds_max_hold_sell_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Dashboard should create SELL orders when a live position reaches max_hold."""
    from stock_indicator import multi_bucket_today

    config_path, log_directory = _write_dashboard_fixture(
        tmp_path,
        signal_date="2026-05-15",
        risk_score=50,
    )
    config_path.write_text(
        json.dumps(
            {
                "max_position_count": 6,
                "starting_cash": 60_000,
                "margin": 1.5,
                "withdraw": 0,
                "risk_score_gate": {
                    "csv_path": "historical_risk_scores.csv",
                    "stop_threshold": 75,
                },
                "buckets": [
                    {
                        "label": "fish_head_b30_35",
                        "strategy_id": "fish_head_b30_35",
                        "dollar_volume_filter": "dollar_volume>0.02%,Top500,Pick5",
                        "max_hold": 10,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    _patch_dashboard_paths(
        monkeypatch,
        config_path=config_path,
        log_directory=log_directory,
        repository_root=tmp_path,
    )
    monkeypatch.setattr(
        dashboard,
        "_get_futu_trd_ctx",
        lambda: FakeTradeContext(
            positions=[{"code": "US.HELD", "qty": 12}],
            deals=[
                {
                    "code": "US.HELD",
                    "trd_side": "BUY",
                    "qty": 12,
                    "create_time": "2026-05-01 09:30:00",
                    "order_id": "101",
                }
            ],
            orders=[
                {
                    "order_id": "101",
                    "remark": "si|sid=fish_head_b30_35|b=fish_head_b30_35|mh=10|rr=0",
                }
            ],
        ),
    )
    monkeypatch.setattr(
        multi_bucket_today,
        "load_strategy_set_mapping",
        lambda: {"fish_head_b30_35": ("fish_head_buy", "fish_head_sell")},
    )
    monkeypatch.setattr(
        multi_bucket_today,
        "load_strategy_entry_filters",
        lambda: {},
    )

    preview = dashboard.api_preview_orders()

    max_hold_orders = [
        order for order in preview["orders"]
        if order.get("exit_reason") == "max_hold"
    ]
    assert max_hold_orders == [
        {
            "side": "SELL",
            "symbol": "HELD",
            "qty": 12,
            "price": 10.0,
            "order_type": "MARKET",
            "bucket": "fish_head_b30_35",
            "strategy_id": "fish_head_b30_35",
            "exit_reason": "max_hold",
            "bars_held": 11,
            "max_hold": 10,
            "entry_source": "futu_history_deals",
        }
    ]


def test_preview_orders_skips_max_hold_when_reentry_resets_window(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A same-day raw entry should suppress max-hold for reset-enabled buckets."""
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
    monkeypatch.setattr(
        dashboard,
        "_get_futu_trd_ctx",
        lambda: FakeTradeContext(
            positions=[{"code": "US.AAA", "qty": 12}],
            deals=[
                {
                    "code": "US.AAA",
                    "trd_side": "BUY",
                    "qty": 12,
                    "create_time": "2026-05-01 09:30:00",
                    "order_id": "201",
                }
            ],
            orders=[
                {
                    "order_id": "201",
                    "remark": "si|sid=fish_head_vacuum_turn|b=fish_head_production|mh=10|rr=1",
                }
            ],
        ),
    )

    preview = dashboard.api_preview_orders()

    assert not [
        order for order in preview["orders"]
        if order.get("exit_reason") == "max_hold"
    ]


def test_preview_orders_does_not_use_local_state_for_max_hold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Local accepted_entries should not create live max-hold SELL orders."""
    config_path, log_directory = _write_dashboard_fixture(
        tmp_path,
        signal_date="2026-05-15",
        risk_score=50,
    )
    (tmp_path / "adaptive_state.json").write_text(
        json.dumps(
            {
                "accepted_entries": [
                    {
                        "entry_date": "2026-05-01",
                        "bucket": "fish_head_production",
                        "strategy_id": "fish_head_vacuum_turn",
                        "symbol": "LOCAL_ONLY",
                        "max_hold": 10,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _patch_dashboard_paths(
        monkeypatch,
        config_path=config_path,
        log_directory=log_directory,
        repository_root=tmp_path,
    )
    monkeypatch.setattr(
        dashboard,
        "_get_futu_trd_ctx",
        lambda: FakeTradeContext(
            positions=[{"code": "US.LOCAL_ONLY", "qty": 12}],
            deals=[],
        ),
    )

    preview = dashboard.api_preview_orders()

    assert not [
        order for order in preview["orders"]
        if order.get("symbol") == "LOCAL_ONLY"
    ]


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


def test_cron_dashboard_contract_names_layer_ownership() -> None:
    """Dashboard should expose a plain-language source-of-truth contract."""
    communication_contract = dashboard._build_cron_dashboard_contract()

    step_owners = [
        contract_step["owner"]
        for contract_step in communication_contract["steps"]
    ]
    note_text = " ".join(communication_contract["notes"])

    assert step_owners == ["Cron", "Dashboard", "Futu"]
    assert "Raw entries are strategy signals" in note_text
    assert "diagnostic" in note_text
