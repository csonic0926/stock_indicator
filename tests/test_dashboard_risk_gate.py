"""Tests for dashboard-owned risk-score order gating."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pandas

from stock_indicator import (
    adaptive_tp_sl_virtual_trade_history,
    dashboard,
    live_expectancy_gate,
)
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
        return 0, pandas.DataFrame(
            self.positions,
            columns=[
                "code",
                "qty",
                "cost_price",
                "nominal_price",
                "market_val",
                "unrealized_pl",
                "pl_ratio",
            ],
        )

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

    def order_list_query(
        self,
        trd_env: Any = None,
    ) -> tuple[int, pandas.DataFrame]:
        """Return non-terminal current broker orders only."""
        current_orders = [
            order
            for order in self.orders
            if order.get("code") and order.get("trd_side")
        ]
        return 0, pandas.DataFrame(
            current_orders,
            columns=[
                "code",
                "trd_side",
                "order_status",
                "order_type",
                "order_id",
                "remark",
            ],
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
        "tradable_candidates: [('AAA', 'fish_head_production')]\n"
        "filtered_out: [('BBB', 'fish_tail_explore', 'symbol_seasoning')]\n"
        "adaptive_tp_sl_virtual_open_trades_before_today=2 "
        "adaptive_tp_sl_virtual_trades_closed_today=0\n"
        "[ADAPTIVE_TP_SL_VIRTUAL_TRADE_HISTORY_STATE] "
        "winner_returns=20 loser_returns=20 pending_returns=0 "
        "closed_trades=40\n"
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


def test_parse_log_separates_raw_entries_from_tradable_candidates(
    tmp_path: Path,
) -> None:
    """Cron should preserve bucket-aware tradable records for allocation."""
    _config_path, log_directory = _write_dashboard_fixture(
        tmp_path,
        signal_date="2026-05-15",
        risk_score=50,
    )

    parsed_log = dashboard._parse_log(log_directory / "2026-05-15.log")

    assert parsed_log["entry_signals"] == ["AAA", "BBB"]
    assert parsed_log["tradable_entry_candidates"][0]["symbol"] == "AAA"
    assert parsed_log["signal_candidate_summary"]["tradable"] == [
        {"symbol": "AAA", "bucket": "fish_head_production"}
    ]
    assert parsed_log["signal_candidate_summary"]["filtered_out"] == [
        {
            "symbol": "BBB",
            "bucket": "fish_tail_explore",
            "reason": "symbol_seasoning",
        }
    ]
    assert parsed_log["adaptive_tp_sl_virtual_trade_history_state"] == {
        "winner_returns": 20,
        "loser_returns": 20,
        "pending_returns": 0,
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
    assert parsed_log["adaptive_tp_sl_virtual_open_trade_count"] == 2


def test_parse_slot_allocation_pairs_preserves_detailed_rejection_reason() -> None:
    """Dashboard should show cron's measured rejection reason unchanged."""

    parsed_records = dashboard._parse_slot_allocation_pairs(
        "[('OKLO', 'fish_head_production', "
        "'free_fall: slope_60=-0.3079 < -0.2000 and "
        "near_delta=-0.1218 < -0.0500')]"
    )

    assert parsed_records == [
        {
            "symbol": "OKLO",
            "bucket": "fish_head_production",
            "reason": (
                "free_fall: slope_60=-0.3079 < -0.2000 and "
                "near_delta=-0.1218 < -0.0500"
            ),
        }
    ]


def test_parse_log_uses_last_cron_run_when_log_was_appended(
    tmp_path: Path,
) -> None:
    """Dashboard should not duplicate TP/SL rows after a manual cron rerun."""

    log_path = tmp_path / "2026-06-19.log"
    log_path.write_text(
        "[multi_bucket_daily_signal mode=live state=adaptive_state.json]\n"
        "[multi_bucket_daily_signal] eval_date=2026-06-19\n"
        "tradable_candidates: [('OLD', 'fish_head_production')]\n"
        "adaptive_tp_sl_virtual_open_trades_before_today=1 "
        "adaptive_tp_sl_virtual_trades_closed_today=0\n"
        "[BUCKET_TP_SL] date=2026-06-19 bucket=fish_head_production "
        "strategy_id=fish_head_vacuum_turn tp_pct=0.010000 sl_pct=0.020000 "
        "rolling_mp=0.010000 rolling_ml=-0.020000 min_hold_tp=1 "
        "min_hold_sl=1 disable_sl_trigger=True max_hold=None "
        "reset_hold_on_reentry_signal=False\n"
        "[multi_bucket_daily_signal mode=live state=adaptive_state.json]\n"
        "[multi_bucket_daily_signal] eval_date=2026-06-19\n"
        "tradable_candidates: []\n"
        "adaptive_tp_sl_virtual_open_trades_before_today=8 "
        "adaptive_tp_sl_virtual_trades_closed_today=0\n"
        "[BUCKET_TP_SL] date=2026-06-19 bucket=fish_head_production "
        "strategy_id=fish_head_vacuum_turn tp_pct=0.060502 sl_pct=0.037015 "
        "rolling_mp=0.034519 rolling_ml=0.037015 min_hold_tp=1 "
        "min_hold_sl=1 disable_sl_trigger=True max_hold=None "
        "reset_hold_on_reentry_signal=False\n",
        encoding="utf-8",
    )

    parsed_log = dashboard._parse_log(log_path)

    assert parsed_log["signal_candidate_summary"]["tradable"] == []
    assert parsed_log["adaptive_tp_sl_virtual_open_trade_count"] == 8
    assert parsed_log["bucket_tp_sl"] == [
        {
            "date": "2026-06-19",
            "bucket": "fish_head_production",
            "strategy_id": "fish_head_vacuum_turn",
            "tp_pct": 0.060502,
            "sl_pct": 0.037015,
            "rolling_mp": 0.034519,
            "rolling_ml": 0.037015,
            "min_hold_tp": 1,
            "min_hold_sl": 1,
            "disable_sl_trigger": True,
            "max_hold": None,
            "reset_hold_on_reentry_signal": False,
        }
    ]


def test_get_log_dates_skips_logs_after_latest_sp500_cache_date(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Dashboard latest log should follow the latest real market session."""

    log_directory = tmp_path / "logs"
    stock_data_directory = tmp_path / "stock_data"
    log_directory.mkdir()
    stock_data_directory.mkdir()
    (log_directory / "2026-06-18.log").write_text("", encoding="utf-8")
    (log_directory / "2026-06-19.log").write_text("", encoding="utf-8")
    (stock_data_directory / f"{dashboard.daily_job.SP500_SYMBOL}.csv").write_text(
        "Date,close\n2026-06-17,1\n2026-06-18,1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "LOGS_DIRECTORY", log_directory)
    monkeypatch.setattr(dashboard, "STOCK_DATA_DIRECTORY", stock_data_directory)

    assert dashboard._get_log_dates() == ["2026-06-18"]


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
    live_state_directory = repository_root / "live_state"
    live_state_directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(dashboard, "LIVE_STATE_DIRECTORY", live_state_directory)
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


def test_preview_orders_ignores_closed_zero_quantity_futu_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Same-day closed rows must not occupy slots or count as holdings."""
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
            positions=[{"code": "US.CLOSED", "qty": 0}],
        ),
    )

    preview = dashboard.api_preview_orders()

    assert preview["held_count"] == 0
    assert preview["orders"][0]["symbol"] == "AAA"
    assert "status" not in preview["orders"][0]


def test_preview_does_not_duplicate_pending_broker_buy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A submitted BUY must consume a slot and disappear from fresh preview."""
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
            orders=[
                {
                    "code": "US.AAA",
                    "trd_side": "BUY",
                    "order_status": "SUBMITTED",
                    "order_type": "MARKET",
                    "order_id": "pending-buy",
                    "remark": "",
                }
            ]
        ),
    )

    preview = dashboard.api_preview_orders()

    assert preview["orders"] == []
    assert preview["pending_buy_count"] == 1


def test_submit_failed_buy_is_not_treated_as_pending() -> None:
    """A rejected Futu BUY must remain eligible for pre-open retry."""
    order_data = pandas.DataFrame(
        [
            {
                "code": "US.OKLO",
                "trd_side": "BUY",
                "order_status": "SUBMIT_FAILED",
                "order_type": "MARKET",
                "remark": "si2|s=h|tp=784|sl=335|ms=1|ds=1|rr=0",
            },
            {
                "code": "US.BE",
                "trd_side": "BUY",
                "order_status": "SUBMITTED",
                "order_type": "MARKET",
                "remark": "si2|s=h|tp=784|sl=335|ms=1|ds=1|rr=0",
            },
        ]
    )

    pending_symbols = dashboard._load_pending_buy_order_symbols(order_data)

    assert pending_symbols == {"BE"}


def test_futu_positions_include_bucket_metadata(monkeypatch) -> None:
    """Live positions should expose bucket metadata resolved from Futu history."""
    fake_trade_context = FakeTradeContext(
        positions=[
            {
                "code": "US.AAA",
                "qty": 12,
                "cost_price": 10.0,
                "nominal_price": 11.0,
                "market_val": 132.0,
                "unrealized_pl": 12.0,
                "pl_ratio": 0.10,
            }
        ],
    )
    monkeypatch.setattr(
        dashboard,
        "_get_futu_trd_ctx",
        lambda: fake_trade_context,
    )
    monkeypatch.setattr(dashboard, "_get_trd_env", lambda: object())
    monkeypatch.setattr(
        dashboard,
        "_load_futu_open_trade_entries",
        lambda trade_context, trading_environment, *, signal_date_text: {
            "AAA": {
                "bucket": "fish_tail_squeeze",
                "strategy_id": "fish_tail_blow_off_top",
                "entry_date": "2026-06-01",
            }
        },
    )

    response = dashboard.api_futu_positions()

    assert fake_trade_context.is_closed is True
    assert response["connected"] is True
    assert response["positions"] == [
        {
            "symbol": "AAA",
            "bucket": "fish_tail_squeeze",
            "strategy_id": "fish_tail_blow_off_top",
            "entry_date": "2026-06-01",
            "qty": 12.0,
            "cost_price": 10.0,
            "market_price": 11.0,
            "market_val": 132.0,
            "unrealized_pl": 12.0,
            "pl_ratio": 0.10,
        }
    ]


def test_preview_orders_cold_start_keeps_old_positions_and_sizes_new_buy_by_seven(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Cold start should let old Futu positions unwind while new buys use 1/7 sizing."""
    signal_date = "2026-06-02"
    config_path = tmp_path / "multi_bucket_production.json"
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "max_position_count": 7,
                "margin": 1.5,
                "buckets": [
                    {
                        "label": "fish_head_production",
                        "strategy_id": "fish_head_vacuum_turn",
                        "dollar_volume_filter": (
                            "dollar_volume>0.02%,Top500,Pick5"
                        ),
                    },
                    {
                        "label": "fish_head_b30_35",
                        "strategy_id": "fish_head_b30_35",
                        "dollar_volume_filter": (
                            "dollar_volume>0.02%,Top500,Pick5"
                        ),
                        "max_hold": 14,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (log_directory / f"{signal_date}.log").write_text(
        "[FROZEN_TP_SL] entry_date=2026-06-02 "
        "bucket=fish_head_b30_35 strategy_id=fish_head_b30_35 "
        "symbol=AAA dollar_volume_rank=1 tp_pct=0.070000 sl_pct=0.025000 "
        "min_hold_sl=1 disable_sl_trigger=True max_hold=14 "
        "reset_hold_on_reentry_signal=True\n"
        "[FROZEN_TP_SL] entry_date=2026-06-02 "
        "bucket=fish_head_b30_35 strategy_id=fish_head_b30_35 "
        "symbol=BBB dollar_volume_rank=2 tp_pct=0.070000 sl_pct=0.025000 "
        "min_hold_sl=1 disable_sl_trigger=True max_hold=14 "
        "reset_hold_on_reentry_signal=True\n"
        "[EXIT_SIGNAL] symbol=EXITOLD "
        "buckets=fish_head_production strategies=fish_head_vacuum_turn\n",
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
            total_assets=54_600.0,
            positions=[
                {"code": "US.OLD1", "qty": 1},
                {"code": "US.OLD2", "qty": 1},
                {"code": "US.OLD3", "qty": 1},
                {"code": "US.OLD4", "qty": 1},
                {"code": "US.OLD5", "qty": 1},
                {"code": "US.EXITOLD", "qty": 9},
            ],
            deals=[],
        ),
    )

    preview = dashboard.api_preview_orders()

    assert preview["max_positions"] == 7
    assert preview["held_count"] == 6
    buy_orders = [
        order for order in preview["orders"] if order["side"] == "BUY"
    ]
    sell_orders = [
        order for order in preview["orders"] if order["side"] == "SELL"
    ]

    assert buy_orders[0]["symbol"] == "AAA"
    assert buy_orders[0]["qty"] == 150
    assert "status" not in buy_orders[0]
    assert buy_orders[1]["symbol"] == "BBB"
    assert buy_orders[1]["status"] == "slot_full"
    assert "max_positions=7 already filled" in buy_orders[1]["skip_reason"]
    assert sell_orders == [
        {
            "side": "SELL",
            "symbol": "EXITOLD",
            "qty": 9,
            "price": 10.0,
            "order_type": "MARKET",
            "exit_reason": "signal",
        }
    ]


# TODO: review
def test_shared_position_group_counts_members_without_counting_other_buckets(
) -> None:
    """Live group occupancy should sum only buckets in the named group."""

    entry_limits = {
        "head": {
            "bucket": "head",
            "maximum_positions": 7,
            "shared_position_group": None,
        },
        "high_dispersion": {
            "bucket": "high_dispersion",
            "maximum_positions": 4,
            "shared_position_group": "ordinary_tail",
        },
        "standard": {
            "bucket": "standard",
            "maximum_positions": 4,
            "shared_position_group": "ordinary_tail",
        },
    }

    group_counts = dashboard._count_live_positions_by_shared_group(
        bucket_position_counts={
            "head": 3,
            "high_dispersion": 1,
            "standard": 2,
        },
        current_bucket_entry_limits=entry_limits,
    )

    assert group_counts == {"ordinary_tail": 3}


# TODO: review
def test_shared_position_group_cap_uses_any_global_slots() -> None:
    """A grouped candidate may use a late slot until its group cap is full."""

    entry_limit = {
        "bucket": "standard",
        "maximum_positions": 4,
        "fill_remaining": False,
        "shared_position_group": "ordinary_tail",
    }

    assert dashboard._entry_limit_skip_reason(
        entry_limit=entry_limit,
        bucket_position_counts={"standard": 2},
        shared_position_group_counts={"ordinary_tail": 3},
        occupied_position_count=6,
    ) is None

    skip_reason = dashboard._entry_limit_skip_reason(
        entry_limit=entry_limit,
        bucket_position_counts={"standard": 2},
        shared_position_group_counts={"ordinary_tail": 4},
        occupied_position_count=6,
    )

    assert skip_reason is not None
    assert "shared position group ordinary_tail" in skip_reason
    assert "current: 4" in skip_reason


def test_preview_orders_blocks_buy_when_live_bucket_is_at_cap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Live Futu bucket metadata should enforce bucket max_positions."""
    signal_date = "2026-06-26"
    config_path = tmp_path / "multi_bucket_production.json"
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "max_position_count": 7,
                "margin": 1.5,
                "buckets": [
                    {
                        "label": "fish_head_production",
                        "strategy_id": "fish_head_vacuum_turn",
                        "dollar_volume_filter": (
                            "dollar_volume>0.02%,Top500,Pick5"
                        ),
                        "max_positions": 6,
                        "max_hold": 14,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (log_directory / f"{signal_date}.log").write_text(
        "[FROZEN_TP_SL] entry_date=2026-06-26 "
        "bucket=fish_head_production strategy_id=fish_head_vacuum_turn "
        "symbol=GGG dollar_volume_rank=1 tp_pct=0.070000 sl_pct=0.025000 "
        "min_hold_sl=1 disable_sl_trigger=True max_hold=14 "
        "reset_hold_on_reentry_signal=False\n",
        encoding="utf-8",
    )
    _patch_dashboard_paths(
        monkeypatch,
        config_path=config_path,
        log_directory=log_directory,
        repository_root=tmp_path,
    )
    held_symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    fish_head_remark = "si2|s=h|tp=700|sl=300|ms=1|ds=1|mh=14|rr=0"
    monkeypatch.setattr(
        dashboard,
        "_get_futu_trd_ctx",
        lambda: FakeTradeContext(
            total_assets=54_600.0,
            positions=[
                {"code": f"US.{symbol}", "qty": 1}
                for symbol in held_symbols
            ],
            deals=[
                {
                    "code": f"US.{symbol}",
                    "trd_side": "BUY",
                    "qty": 1,
                    "create_time": "2026-06-25 09:30:00",
                    "order_id": str(symbol_position),
                }
                for symbol_position, symbol in enumerate(held_symbols, start=1)
            ],
            orders=[
                {"order_id": str(symbol_position), "remark": fish_head_remark}
                for symbol_position, _symbol in enumerate(held_symbols, start=1)
            ],
        ),
    )

    preview = dashboard.api_preview_orders()

    assert preview["held_count"] == 6
    buy_orders = [
        order for order in preview["orders"] if order["side"] == "BUY"
    ]
    assert buy_orders == [
        {
            "side": "BUY",
            "symbol": "GGG",
            "qty": 0,
            "ref_price": 10.0,
            "order_type": "MARKET",
            "bucket": "fish_head_production",
            "strategy_id": "fish_head_vacuum_turn",
            "tp_pct": 0.07,
            "sl_pct": 0.025,
            "dollar_volume_rank": 1,
            "min_hold_sl": 1,
            "disable_sl_trigger": True,
            "max_hold": 14,
            "reset_hold_on_reentry_signal": False,
            "status": "bucket_cap",
            "skip_reason": (
                "bucket fish_head_production max_positions=6 already filled "
                "(current: 6)"
            ),
        }
    ]


def test_preview_orders_backfills_lower_priority_candidate_after_bucket_cap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A broker-blocked high candidate must not erase the next candidate."""
    signal_date = "2026-07-10"
    config_path = tmp_path / "multi_bucket_production.json"
    risk_score_path = tmp_path / "historical_risk_scores.csv"
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    config_path.write_text(
        json.dumps({
            "max_position_count": 2,
            "risk_score_gate": {
                "csv_path": "historical_risk_scores.csv",
                "stop_threshold": 75,
            },
            "buckets": [
                {
                    "label": "fish_head_production",
                    "strategy_id": "fish_head_vacuum_turn",
                    "dollar_volume_filter": "dollar_volume>0.02%,Top500,Pick5",
                    "priority": 1,
                    "max_positions": 1,
                },
                {
                    "label": "fish_tail_production",
                    "strategy_id": "fish_tail_blow_off_top",
                    "dollar_volume_filter": "dollar_volume>0.02%,Top500,Pick5",
                    "priority": 2,
                    "max_positions": 2,
                },
            ],
        }),
        encoding="utf-8",
    )
    risk_score_path.write_text(
        "year_month,risk_score\n2026-07,0\n",
        encoding="utf-8",
    )
    (log_directory / f"{signal_date}.log").write_text(
        "[FROZEN_TP_SL] entry_date=2026-07-10 "
        "bucket=fish_head_production strategy_id=fish_head_vacuum_turn "
        "symbol=HIGH dollar_volume_rank=0 tp_pct=0.050000 sl_pct=0.030000 "
        "min_hold_sl=1 disable_sl_trigger=True "
        "reset_hold_on_reentry_signal=False\n"
        "[FROZEN_TP_SL] entry_date=2026-07-10 "
        "bucket=fish_tail_production strategy_id=fish_tail_blow_off_top "
        "symbol=LOW dollar_volume_rank=0 tp_pct=0.060000 sl_pct=0.030000 "
        "min_hold_sl=1 disable_sl_trigger=True max_hold=7 "
        "reset_hold_on_reentry_signal=False\n",
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
            positions=[{"code": "US.HELD", "qty": 1}],
            deals=[{
                "code": "US.HELD",
                "trd_side": "BUY",
                "qty": 1,
                "create_time": "2026-07-01 09:30:00",
                "order_id": "held-buy",
            }],
            orders=[{
                "order_id": "held-buy",
                "remark": "si2|s=h|tp=500|sl=300|ms=1|ds=1|rr=0",
            }],
        ),
    )

    preview = dashboard.api_preview_orders()
    buy_orders = [
        order for order in preview["orders"] if order["side"] == "BUY"
    ]

    assert buy_orders[0]["symbol"] == "HIGH"
    assert buy_orders[0]["status"] == "bucket_cap"
    assert buy_orders[1]["symbol"] == "LOW"
    assert "status" not in buy_orders[1]


def test_pending_buy_counts_toward_its_bucket_and_allows_backfill(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A pending BUY must reserve both its global slot and bucket slot."""
    signal_date = "2026-07-10"
    config_path = tmp_path / "multi_bucket_production.json"
    risk_score_path = tmp_path / "historical_risk_scores.csv"
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    config_path.write_text(
        json.dumps({
            "max_position_count": 2,
            "risk_score_gate": {
                "csv_path": "historical_risk_scores.csv",
                "stop_threshold": 75,
            },
            "buckets": [
                {
                    "label": "fish_head_production",
                    "strategy_id": "fish_head_vacuum_turn",
                    "dollar_volume_filter": "dollar_volume>0.02%,Top500,Pick5",
                    "priority": 1,
                    "max_positions": 1,
                },
                {
                    "label": "fish_tail_production",
                    "strategy_id": "fish_tail_blow_off_top",
                    "dollar_volume_filter": "dollar_volume>0.02%,Top500,Pick5",
                    "priority": 2,
                    "max_positions": 2,
                },
            ],
        }),
        encoding="utf-8",
    )
    risk_score_path.write_text(
        "year_month,risk_score\n2026-07,0\n",
        encoding="utf-8",
    )
    (log_directory / f"{signal_date}.log").write_text(
        "[FROZEN_TP_SL] entry_date=2026-07-10 "
        "bucket=fish_head_production strategy_id=fish_head_vacuum_turn "
        "symbol=HIGH dollar_volume_rank=0 tp_pct=0.050000 sl_pct=0.030000 "
        "min_hold_sl=1 disable_sl_trigger=True "
        "reset_hold_on_reentry_signal=False\n"
        "[FROZEN_TP_SL] entry_date=2026-07-10 "
        "bucket=fish_tail_production strategy_id=fish_tail_blow_off_top "
        "symbol=LOW dollar_volume_rank=0 tp_pct=0.060000 sl_pct=0.030000 "
        "min_hold_sl=1 disable_sl_trigger=True max_hold=7 "
        "reset_hold_on_reentry_signal=False\n",
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
        lambda: FakeTradeContext(orders=[{
            "code": "US.PENDING",
            "trd_side": "BUY",
            "order_status": "SUBMITTING",
            "order_type": "MARKET",
            "order_id": "pending-buy",
            "remark": "si2|s=h|tp=500|sl=300|ms=1|ds=1|rr=0",
        }]),
    )

    preview = dashboard.api_preview_orders()
    buy_orders = [
        order for order in preview["orders"] if order["side"] == "BUY"
    ]

    assert preview["pending_buy_count"] == 1
    assert buy_orders[0]["symbol"] == "HIGH"
    assert buy_orders[0]["status"] == "bucket_cap"
    assert buy_orders[1]["symbol"] == "LOW"
    assert "status" not in buy_orders[1]


def test_preview_orders_resolves_same_symbol_competition_by_bucket_priority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Two bucket signals for one symbol should produce one winning BUY."""
    signal_date = "2026-07-10"
    config_path = tmp_path / "multi_bucket_production.json"
    risk_score_path = tmp_path / "historical_risk_scores.csv"
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    config_path.write_text(
        json.dumps({
            "max_position_count": 2,
            "max_same_symbol": 1,
            "risk_score_gate": {
                "csv_path": "historical_risk_scores.csv",
                "stop_threshold": 75,
            },
            "buckets": [
                {
                    "label": "fish_tail_squeeze",
                    "strategy_id": "fish_tail_blow_off_top",
                    "dollar_volume_filter": "dollar_volume>0.02%,Top500,Pick5",
                    "priority": 1,
                    "max_positions": 2,
                    "max_hold": 7,
                },
                {
                    "label": "fish_tail_production",
                    "strategy_id": "fish_tail_blow_off_top",
                    "dollar_volume_filter": "dollar_volume>0.02%,Top500,Pick5",
                    "priority": 2,
                    "max_positions": 2,
                    "max_hold": 7,
                },
            ],
        }),
        encoding="utf-8",
    )
    risk_score_path.write_text(
        "year_month,risk_score\n2026-07,0\n",
        encoding="utf-8",
    )
    (log_directory / f"{signal_date}.log").write_text(
        "[FROZEN_TP_SL] entry_date=2026-07-10 "
        "bucket=fish_tail_production strategy_id=fish_tail_blow_off_top "
        "symbol=DUAL dollar_volume_rank=0 tp_pct=0.040000 sl_pct=0.030000 "
        "min_hold_sl=1 disable_sl_trigger=True max_hold=7 "
        "reset_hold_on_reentry_signal=False\n"
        "[FROZEN_TP_SL] entry_date=2026-07-10 "
        "bucket=fish_tail_squeeze strategy_id=fish_tail_blow_off_top "
        "symbol=DUAL dollar_volume_rank=0 tp_pct=0.070000 sl_pct=0.030000 "
        "min_hold_sl=1 disable_sl_trigger=True max_hold=7 "
        "reset_hold_on_reentry_signal=False\n",
        encoding="utf-8",
    )
    _patch_dashboard_paths(
        monkeypatch,
        config_path=config_path,
        log_directory=log_directory,
        repository_root=tmp_path,
    )

    preview = dashboard.api_preview_orders()
    buy_orders = [
        order for order in preview["orders"] if order["side"] == "BUY"
    ]

    assert len(buy_orders) == 1
    assert buy_orders[0]["symbol"] == "DUAL"
    assert buy_orders[0]["bucket"] == "fish_tail_squeeze"
    assert buy_orders[0]["tp_pct"] == 0.07
    assert parse_futu_order_remark(
        dashboard._format_dashboard_order_remark(buy_orders[0])
    )["bucket"] == "fish_tail_squeeze"


def test_preview_orders_applies_risk_score_priority_override_in_dashboard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Current risk-score settings, not log order, should choose the slot."""
    signal_date = "2026-07-10"
    config_path = tmp_path / "multi_bucket_production.json"
    risk_score_path = tmp_path / "historical_risk_scores.csv"
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    config_path.write_text(
        json.dumps({
            "max_position_count": 1,
            "risk_score_gate": {
                "csv_path": "historical_risk_scores.csv",
                "stop_threshold": 75,
            },
            "risk_score_priority_overrides": {
                "scores": [25],
                "priorities": {
                    "fish_head_production": 2,
                    "fish_tail_production": 1,
                },
            },
            "buckets": [
                {
                    "label": "fish_head_production",
                    "strategy_id": "fish_head_vacuum_turn",
                    "dollar_volume_filter": "dollar_volume>0.02%,Top500,Pick5",
                    "priority": 1,
                },
                {
                    "label": "fish_tail_production",
                    "strategy_id": "fish_tail_blow_off_top",
                    "dollar_volume_filter": "dollar_volume>0.02%,Top500,Pick5",
                    "priority": 2,
                },
            ],
        }),
        encoding="utf-8",
    )
    risk_score_path.write_text(
        "year_month,risk_score\n2026-07,25\n",
        encoding="utf-8",
    )
    (log_directory / f"{signal_date}.log").write_text(
        "[FROZEN_TP_SL] entry_date=2026-07-10 "
        "bucket=fish_head_production strategy_id=fish_head_vacuum_turn "
        "symbol=HEAD dollar_volume_rank=0 tp_pct=0.050000 sl_pct=0.030000 "
        "min_hold_sl=1 disable_sl_trigger=True "
        "reset_hold_on_reentry_signal=False\n"
        "[FROZEN_TP_SL] entry_date=2026-07-10 "
        "bucket=fish_tail_production strategy_id=fish_tail_blow_off_top "
        "symbol=TAIL dollar_volume_rank=9 tp_pct=0.060000 sl_pct=0.030000 "
        "min_hold_sl=1 disable_sl_trigger=True max_hold=7 "
        "reset_hold_on_reentry_signal=False\n",
        encoding="utf-8",
    )
    _patch_dashboard_paths(
        monkeypatch,
        config_path=config_path,
        log_directory=log_directory,
        repository_root=tmp_path,
    )

    preview = dashboard.api_preview_orders()
    buy_orders = [
        order for order in preview["orders"] if order["side"] == "BUY"
    ]

    assert buy_orders[0]["symbol"] == "TAIL"
    assert "status" not in buy_orders[0]
    assert buy_orders[1]["symbol"] == "HEAD"
    assert buy_orders[1]["status"] == "slot_full"


def test_dashboard_contract_names_virtual_history_as_statistical_state() -> None:
    """The UI contract must not describe adaptive history as a portfolio."""

    contract = dashboard._build_cron_dashboard_contract()
    serialized_contract = json.dumps(contract)

    assert "ADAPTIVE TP/SL virtual trade history" in serialized_contract
    assert "statistical" in serialized_contract
    assert "virtual portfolio" not in serialized_contract.lower()


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


def test_dashboard_buy_remark_uses_live_fish_tail_production_bucket() -> None:
    """Fish-tail BUY remarks should parse to the promoted production bucket."""
    order = {
        "side": "BUY",
        "strategy_id": "fish_tail_blow_off_top",
        "tp_pct": 0.0586,
        "sl_pct": 0.0420,
        "min_hold_sl": 1,
        "disable_sl_trigger": True,
        "max_hold": 7,
        "reset_hold_on_reentry_signal": False,
    }

    remark_text = dashboard._format_dashboard_order_remark(order)
    metadata = parse_futu_order_remark(remark_text)

    assert metadata["strategy_id"] == "fish_tail_blow_off_top"
    assert metadata["bucket"] == "fish_tail_production"
    assert metadata["max_hold"] == 7
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
            # Entry bar is zero, matching the simulator's
            # bars_since_anchor counter.
            "bars_held": 10,
            "max_hold": 10,
            "entry_source": "futu_history_deals",
        }
    ]


def test_preview_orders_fills_missing_fish_head_max_hold_from_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Current bucket config should repair old fish-head remarks missing mh."""
    config_path, log_directory = _write_dashboard_fixture(
        tmp_path,
        signal_date="2026-07-01",
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
                        "label": "fish_head_production",
                        "strategy_id": "fish_head_vacuum_turn",
                        "dollar_volume_filter": (
                            "dollar_volume>0.02%,Top500,Pick5"
                        ),
                        "max_hold": 14,
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
        "_load_us_trading_days",
        lambda start_date_text, end_date_text: [
            "2026-06-10",
            "2026-06-11",
            "2026-06-12",
            "2026-06-15",
            "2026-06-16",
            "2026-06-17",
            "2026-06-18",
            "2026-06-22",
            "2026-06-23",
            "2026-06-24",
            "2026-06-25",
            "2026-06-26",
            "2026-06-29",
            "2026-06-30",
            "2026-07-01",
        ],
    )
    monkeypatch.setattr(
        dashboard,
        "_get_futu_trd_ctx",
        lambda: FakeTradeContext(
            positions=[{"code": "US.LYB", "qty": 75}],
            deals=[
                {
                    "code": "US.LYB",
                    "trd_side": "BUY",
                    "qty": 75,
                    "create_time": "2026-06-10 09:30:00",
                    "order_id": "201",
                }
            ],
            orders=[
                {
                    "order_id": "201",
                    "remark": "si2|s=h|tp=628|sl=248|ms=1|ds=1|rr=0",
                }
            ],
        ),
    )

    preview = dashboard.api_preview_orders()

    max_hold_orders = [
        order for order in preview["orders"]
        if order.get("exit_reason") == "max_hold"
    ]
    assert max_hold_orders == [
        {
            "side": "SELL",
            "symbol": "LYB",
            "qty": 75,
            "price": 10.0,
            "order_type": "MARKET",
            "bucket": "fish_head_production",
            "strategy_id": "fish_head_vacuum_turn",
            "exit_reason": "max_hold",
            "bars_held": 14,
            "max_hold": 14,
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
    (tmp_path / "live_state").mkdir(exist_ok=True)
    (tmp_path / "live_state" / "adaptive_state.json").write_text(
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
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_TRADE_HISTORY_KEY: {
            "schema_version": 1,
            adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_WINNER_RETURNS_KEY: [
                0.03,
                0.05,
                0.07,
                0.09,
                0.11,
            ],
            adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_LOSER_RETURNS_KEY: [
                -0.02,
                -0.03,
                -0.04,
            ],
        }
    }

    bucket_states = (
        dashboard._load_current_bucket_tp_sl_from_adaptive_tp_sl_virtual_trade_history(
            adaptive_state
        )
    )
    tp_by_bucket = {state["bucket"]: state["tp_pct"] for state in bucket_states}

    assert tp_by_bucket["fish_head_production"] > tp_by_bucket["fish_tail_explore"]
    assert tp_by_bucket["fish_tail_explore"] == 0.07


def _write_min_hold_fixture(
    tmp_path: Path,
    *,
    signal_date: str,
    held_symbol: str,
    min_hold: int,
) -> tuple[Path, Path]:
    """Write a production config (with min_hold) and a SELL-signal log."""
    config_path = tmp_path / "multi_bucket_production.json"
    log_directory = tmp_path / "logs"
    log_directory.mkdir()

    config_path.write_text(
        json.dumps(
            {
                "max_position_count": 6,
                "starting_cash": 60_000,
                "margin": 1.5,
                "withdraw": 0,
                "min_hold": min_hold,
                "buckets": [
                    {
                        "label": "fish_head_production",
                        "strategy_id": "fish_head_vacuum_turn",
                        "dollar_volume_filter": (
                            "dollar_volume>0.02%,Top500,Pick5"
                        ),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (log_directory / f"{signal_date}.log").write_text(
        f"[EXIT_SIGNAL] symbol={held_symbol} "
        "buckets=fish_head_production strategies=fish_head_vacuum_turn\n",
        encoding="utf-8",
    )
    return config_path, log_directory


def test_preview_orders_blocks_sell_when_min_hold_not_satisfied(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A SELL signal on the same trading day as a Futu entry must be blocked."""
    config_path, log_directory = _write_min_hold_fixture(
        tmp_path,
        signal_date="2026-05-26",
        held_symbol="UNH",
        min_hold=5,
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
            positions=[{"code": "US.UNH", "qty": 14}],
            deals=[
                {
                    "code": "US.UNH",
                    "trd_side": "BUY",
                    "qty": 14,
                    "create_time": "2026-05-26 09:30:00",
                    "order_id": "11",
                }
            ],
            orders=[],
        ),
    )

    preview = dashboard.api_preview_orders()
    sell_orders = [
        order for order in preview["orders"] if order["side"] == "SELL"
    ]

    assert len(sell_orders) == 1
    blocked_order = sell_orders[0]
    assert blocked_order["symbol"] == "UNH"
    assert blocked_order["status"] == "min_hold_block"
    assert blocked_order["qty"] == 0
    # The entry bar is zero; the first subsequent trading bar is one.
    assert blocked_order["bars_held"] == 0
    assert blocked_order["min_hold"] == 5
    assert blocked_order["entry_date"] == "2026-05-26"
    assert "min_hold=5" in blocked_order["skip_reason"]


def test_preview_orders_allows_sell_when_min_hold_satisfied(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A SELL signal after the min_hold window must pass through unblocked."""
    config_path, log_directory = _write_min_hold_fixture(
        tmp_path,
        signal_date="2026-05-26",
        held_symbol="HD",
        min_hold=5,
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
            positions=[{"code": "US.HD", "qty": 17}],
            deals=[
                {
                    "code": "US.HD",
                    "trd_side": "BUY",
                    "qty": 17,
                    "create_time": "2026-05-06 09:30:00",
                    "order_id": "22",
                }
            ],
            orders=[],
        ),
    )

    preview = dashboard.api_preview_orders()
    sell_orders = [
        order for order in preview["orders"] if order["side"] == "SELL"
    ]

    assert len(sell_orders) == 1
    passed_order = sell_orders[0]
    assert passed_order["symbol"] == "HD"
    assert "status" not in passed_order
    assert passed_order["qty"] == 17
    assert passed_order["exit_reason"] == "signal"


def test_preview_orders_sells_broker_symbol_after_strategy_symbol_rename(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A SATS signal exit should sell the renamed ECHO broker position."""
    config_path, log_directory = _write_min_hold_fixture(
        tmp_path,
        signal_date="2026-06-26",
        held_symbol="SATS",
        min_hold=1,
    )
    (tmp_path / "symbol_rename_map.csv").write_text(
        "strategy_symbol,broker_symbol\nSATS,ECHO\n",
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
        "_load_us_trading_days",
        lambda start_date_text, end_date_text: [
            "2026-06-22",
            "2026-06-23",
            "2026-06-24",
            "2026-06-25",
            "2026-06-26",
        ],
    )
    monkeypatch.setattr(
        dashboard,
        "_get_futu_trd_ctx",
        lambda: FakeTradeContext(
            positions=[{"code": "US.ECHO", "qty": 9}],
            deals=[
                {
                    "code": "US.SATS",
                    "trd_side": "BUY",
                    "qty": 9,
                    "create_time": "2026-06-22 09:30:00",
                    "order_id": "31",
                }
            ],
            orders=[],
        ),
    )

    preview = dashboard.api_preview_orders()
    sell_orders = [
        order for order in preview["orders"] if order["side"] == "SELL"
    ]

    assert sell_orders == [
        {
            "side": "SELL",
            "symbol": "ECHO",
            "qty": 9,
            "price": 10.0,
            "order_type": "MARKET",
            "exit_reason": "signal",
            "signal_symbol": "SATS",
        }
    ]


def test_preview_orders_max_hold_uses_renamed_broker_symbol(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Max-hold should match a SATS buy history to the live ECHO position."""
    config_path, log_directory = _write_dashboard_fixture(
        tmp_path,
        signal_date="2026-06-26",
        risk_score=50,
    )
    config_path.write_text(
        json.dumps(
            {
                "max_position_count": 6,
                "risk_score_gate": {
                    "csv_path": "historical_risk_scores.csv",
                    "stop_threshold": 75,
                },
                "buckets": [
                    {
                        "label": "fish_head_production",
                        "strategy_id": "fish_head_vacuum_turn",
                        "dollar_volume_filter": (
                            "dollar_volume>0.02%,Top500,Pick5"
                        ),
                        "max_hold": 2,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (log_directory / "2026-06-26.log").write_text(
        "[multi_bucket_daily_signal mode=live state=adaptive_state.json]\n"
        "[multi_bucket_daily_signal] eval_date=2026-06-26\n",
        encoding="utf-8",
    )
    (tmp_path / "symbol_rename_map.csv").write_text(
        "strategy_symbol,broker_symbol\nSATS,ECHO\n",
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
        "_load_us_trading_days",
        lambda start_date_text, end_date_text: [
            "2026-06-22",
            "2026-06-23",
            "2026-06-24",
            "2026-06-25",
            "2026-06-26",
        ],
    )
    monkeypatch.setattr(
        dashboard,
        "_get_futu_trd_ctx",
        lambda: FakeTradeContext(
            positions=[{"code": "US.ECHO", "qty": 5}],
            deals=[
                {
                    "code": "US.SATS",
                    "trd_side": "BUY",
                    "qty": 5,
                    "create_time": "2026-06-22 09:30:00",
                    "order_id": "41",
                }
            ],
            orders=[
                {
                    "order_id": "41",
                    "remark": (
                        "si|sid=fish_head_vacuum_turn|"
                        "b=fish_head_production|mh=2|rr=0"
                    ),
                }
            ],
        ),
    )

    preview = dashboard.api_preview_orders()
    max_hold_orders = [
        order for order in preview["orders"]
        if order.get("exit_reason") == "max_hold"
    ]

    assert max_hold_orders == [
        {
            "side": "SELL",
            "symbol": "ECHO",
            "qty": 5,
            "price": 10.0,
            "order_type": "MARKET",
            "bucket": "fish_head_production",
            "strategy_id": "fish_head_vacuum_turn",
            "exit_reason": "max_hold",
            "bars_held": 4,
            "max_hold": 2,
            "entry_source": "futu_history_deals",
        }
    ]


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


# ----------------------------------------------------------------------
# WR-gate phantom (Step E): the order layer ANDs the cron's wr_degrading
# flag with the month's risk score, holds a slot with zero capital, and
# releases it when the position would have adaptively closed.
# ----------------------------------------------------------------------


def _write_phantom_csv(
    data_directory: Path,
    symbol: str,
    rows: list[tuple[str, float, float, float]],
) -> None:
    """Write a per-symbol daily CSV (Date,open,high,low) compatible with
    both _read_open_price and compute_adaptive_tp_sl_virtual_trade_close_for_wr_gate."""
    data_directory.mkdir(parents=True, exist_ok=True)
    lines = ["Date,open,high,low"]
    for date_text, open_price, high_price, low_price in rows:
        lines.append(f"{date_text},{open_price},{high_price},{low_price}")
    (data_directory / f"{symbol}.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_phantom_fixture(
    tmp_path: Path,
    *,
    signal_date: str,
    risk_score: int,
    wr_degrading: bool,
    symbol: str = "PHX",
    max_position_count: int = 6,
) -> tuple[Path, Path]:
    """Write a config WITH ft_family_wr_gate (activation 25) and a log
    whose single gated-bucket FROZEN entry carries the wr_degrading flag."""
    config_path = tmp_path / "multi_bucket_production.json"
    risk_score_path = tmp_path / "historical_risk_scores.csv"
    log_directory = tmp_path / "logs"
    log_directory.mkdir(exist_ok=True)

    config_path.write_text(
        json.dumps(
            {
                "max_position_count": max_position_count,
                "risk_score_gate": {
                    "csv_path": "historical_risk_scores.csv",
                    "stop_threshold": 75,
                },
                "ft_family_wr_gate": {
                    "sensor_bucket": "fish_tail_production",
                    "gated_buckets": [
                        "fish_tail_production",
                        "fish_tail_squeeze",
                    ],
                    "window": 12,
                    "curve": "wr_cross",
                    "risk_score_activation_threshold": 25,
                },
            }
        ),
        encoding="utf-8",
    )
    risk_score_path.write_text(
        "year_month,duration_score,breadth_score,risk_score,recommendation,key_event,confidence\n"
        f"{signal_date[:7]},50,50,{risk_score},hold,test,H\n",
        encoding="utf-8",
    )
    (log_directory / f"{signal_date}.log").write_text(
        f"[ENTRY_SIGNAL] bucket=fish_tail_production "
        f"strategy_id=fish_tail_blow_off_top symbol={symbol}\n"
        f"tradable_candidates: [('{symbol}', 'fish_tail_production')]\n"
        "filtered_out: []\n"
        "adaptive_tp_sl_virtual_open_trades_before_today=0 "
        "adaptive_tp_sl_virtual_trades_closed_today=0\n"
        "[ADAPTIVE_TP_SL_VIRTUAL_TRADE_HISTORY_STATE] "
        "winner_returns=20 loser_returns=20 pending_returns=0 "
        "closed_trades=40\n"
        "[WR_GATE_SENSOR] ema=0.4800 sma=0.6000 breakeven=0.4500 "
        f"degrading={wr_degrading} window=12/12 window_full=True "
        "open_pending=1 fed_this_run=0\n"
        "[FROZEN_TP_SL] "
        f"entry_date={signal_date} bucket=fish_tail_production "
        f"strategy_id=fish_tail_blow_off_top symbol={symbol} "
        "dollar_volume_rank=0 tp_pct=0.050000 sl_pct=0.030000 "
        "rolling_mp=0.040000 slope_60=0.1000 near_delta=0.2000 "
        "min_hold_tp=1 disable_sl_trigger=True max_hold=7 "
        f"wr_degrading={wr_degrading}\n",
        encoding="utf-8",
    )
    return config_path, log_directory


def _patch_phantom_paths(monkeypatch, repository_root: Path) -> Path:
    """Point the module-level phantom constants at the temp tree and
    return the daily data directory."""
    live_state_directory = repository_root / "live_state"
    live_state_directory.mkdir(parents=True, exist_ok=True)
    data_directory = repository_root / "stock_data"
    data_directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        dashboard,
        "PHANTOM_POSITIONS_PATH",
        live_state_directory / "phantom_positions.json",
    )
    monkeypatch.setattr(
        dashboard, "PRODUCTION_DATA_DIRECTORY", data_directory
    )
    return data_directory


def test_preview_marks_gated_entry_phantom_when_rs_active(
    tmp_path: Path, monkeypatch
) -> None:
    """wr_degrading + risk_score >= activation -> phantom: zero qty, slot
    held, phantom_record carried for execute persistence."""
    config_path, log_directory = _write_phantom_fixture(
        tmp_path, signal_date="2026-05-15", risk_score=30, wr_degrading=True
    )
    _patch_dashboard_paths(
        monkeypatch,
        config_path=config_path,
        log_directory=log_directory,
        repository_root=tmp_path,
    )
    _patch_phantom_paths(monkeypatch, tmp_path)

    preview = dashboard.api_preview_orders()

    assert preview["risk_score_gate"]["status"] == "open"
    order = preview["orders"][0]
    assert order["symbol"] == "PHX"
    assert order["status"] == "wr_gate_phantom"
    assert order["qty"] == 0
    assert order["phantom_record"] == {
        "symbol": "PHX",
        "signal_date": "2026-05-15",
        "bucket": "fish_tail_production",
        "tp_pct": 0.05,
        "min_hold_tp": 1,
        "max_hold": 7,
    }

    # Observability: the wr_gate summary surfaces gate state + the cron's
    # sensor heartbeat + today's phantom decision.
    wr_gate = preview["wr_gate"]
    assert wr_gate["configured"] is True
    assert wr_gate["active"] is True
    assert wr_gate["risk_score"] == 30
    assert wr_gate["activation_threshold"] == 25
    assert wr_gate["sensor"]["degrading"] is True
    assert wr_gate["sensor"]["ema"] == 0.48
    assert wr_gate["phantomed_today"] == ["PHX"]


def test_parse_log_extracts_wr_gate_sensor_heartbeat(tmp_path: Path) -> None:
    """_parse_log surfaces the last WR_GATE_SENSOR line with typed fields."""
    log_path = tmp_path / "2026-05-15.log"
    log_path.write_text(
        "[WR_GATE_SENSOR] ema=0.4800 sma=0.6000 breakeven=0.4500 "
        "degrading=True window=12/12 window_full=True open_pending=2 "
        "fed_this_run=1\n",
        encoding="utf-8",
    )
    sensor = dashboard._parse_log(log_path)["wr_gate_sensor"]
    assert sensor["ema"] == 0.48
    assert sensor["degrading"] is True
    assert sensor["window_full"] is True
    assert sensor["open_pending"] == 2
    assert sensor["window"] == "12/12"


def test_preview_no_phantom_when_rs_below_activation(
    tmp_path: Path, monkeypatch
) -> None:
    """Below the activation threshold the gate is OFF; a degrading entry
    funds normally."""
    config_path, log_directory = _write_phantom_fixture(
        tmp_path, signal_date="2026-05-15", risk_score=20, wr_degrading=True
    )
    _patch_dashboard_paths(
        monkeypatch,
        config_path=config_path,
        log_directory=log_directory,
        repository_root=tmp_path,
    )
    _patch_phantom_paths(monkeypatch, tmp_path)

    order = dashboard.api_preview_orders()["orders"][0]
    assert order["symbol"] == "PHX"
    assert order["qty"] > 0
    assert order.get("status") != "wr_gate_phantom"


def test_preview_no_phantom_when_not_degrading(
    tmp_path: Path, monkeypatch
) -> None:
    """Active month but a non-degrading entry funds normally."""
    config_path, log_directory = _write_phantom_fixture(
        tmp_path, signal_date="2026-05-15", risk_score=30, wr_degrading=False
    )
    _patch_dashboard_paths(
        monkeypatch,
        config_path=config_path,
        log_directory=log_directory,
        repository_root=tmp_path,
    )
    _patch_phantom_paths(monkeypatch, tmp_path)

    order = dashboard.api_preview_orders()["orders"][0]
    assert order["qty"] > 0
    assert order.get("status") != "wr_gate_phantom"


def test_preview_wr_gate_uses_sensor_without_risk_score(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Production-style WR config must phantom directly from its sensor."""

    config_path, log_directory = _write_phantom_fixture(
        tmp_path,
        signal_date="2026-05-15",
        risk_score=30,
        wr_degrading=True,
    )
    config_document = json.loads(config_path.read_text(encoding="utf-8"))
    config_document.pop("risk_score_gate")
    config_document["ft_family_wr_gate"].pop(
        "risk_score_activation_threshold"
    )
    config_path.write_text(
        json.dumps(config_document),
        encoding="utf-8",
    )
    (tmp_path / "historical_risk_scores.csv").unlink()
    _patch_dashboard_paths(
        monkeypatch,
        config_path=config_path,
        log_directory=log_directory,
        repository_root=tmp_path,
    )
    _patch_phantom_paths(monkeypatch, tmp_path)

    preview = dashboard.api_preview_orders()

    assert preview["risk_score_gate"]["status"] == "off"
    order = preview["orders"][0]
    assert order["status"] == "wr_gate_phantom"
    assert order["qty"] == 0
    assert "sensor-only" in order["skip_reason"]
    assert preview["wr_gate"] == {
        "configured": True,
        "active": True,
        "activation_mode": "sensor_only",
        "risk_score": None,
        "activation_threshold": None,
        "sensor": {
            "ema": 0.48,
            "sma": 0.6,
            "breakeven": 0.45,
            "degrading": True,
            "window": "12/12",
            "window_full": True,
            "open_pending": 1,
            "fed_this_run": 0,
        },
        "open_phantoms": [],
        "phantomed_today": ["PHX"],
    }


def test_dashboard_html_renders_mechanical_gate_status() -> None:
    """The order preview must expose both mechanical gate panels."""

    assert "<strong>Expectancy gate</strong>" in dashboard.HTML_PAGE
    assert "WARMING" in dashboard.HTML_PAGE
    assert "PRIORITY OVERRIDE" in dashboard.HTML_PAGE
    assert "(sensor-only)" in dashboard.HTML_PAGE
    assert "expectancy_gate_phantom" in dashboard.HTML_PAGE
    assert "await previewOrders();" in dashboard.HTML_PAGE
    assert "initializeDashboard();" in dashboard.HTML_PAGE


def test_preview_applies_expectancy_priority_and_zero_capital_phantom(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The cron heartbeat reorders one daily contest and Tier 1 preserves
    the winning slot as a zero-capital phantom."""

    signal_date = "2026-05-15"
    config_path = tmp_path / "multi_bucket_production.json"
    config_path.write_text(
        json.dumps({
            "max_position_count": 1,
            "starting_cash": 10_000,
            "margin": 1.0,
            "withdraw": 0,
            "expectancy_gate": {
                "enabled": True,
                "window": 2,
                "baseline_mean": 0.0,
                "baseline_sigma": 0.1,
                "sigma_multiplier": 1.0,
                "cold_start": "open",
                "priority_override": {
                    "enabled": True,
                    "sigma_multiplier": 0.5,
                    "priorities": {
                        "fish_head_production": 2,
                        "fish_tail_production": 1,
                    },
                },
            },
            "adaptive_tp_sl": {"window": 2},
            "buckets": [
                {
                    "label": "fish_head_production",
                    "strategy_id": "fish_head_vacuum_turn",
                    "dollar_volume_filter": "dollar_volume=Top5",
                    "priority": 1,
                },
                {
                    "label": "fish_tail_production",
                    "strategy_id": "fish_tail_blow_off_top",
                    "dollar_volume_filter": "dollar_volume=Top5",
                    "priority": 2,
                },
            ],
        }),
        encoding="utf-8",
    )
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    (log_directory / f"{signal_date}.log").write_text(
        "[EXPECTANCY_GATE_SENSOR] status=ready mean=-0.200000 "
        "stop_threshold=-0.100000 soft_threshold=-0.050000 "
        "gate_closed=True priority_override_active=True window=2/2 "
        "window_full=True open_pending=0 fed_this_run=0 "
        "closed_episodes=0 expectancy_gated_trades=0 "
        "priority_override_entries=0\n"
        "[FROZEN_TP_SL] entry_date=2026-05-15 "
        "bucket=fish_head_production strategy_id=fish_head_vacuum_turn "
        "symbol=HEAD dollar_volume_rank=0 tp_pct=0.050000 sl_pct=0.000000 "
        "rolling_mp=0.040000 slope_60=0.1 near_delta=0.1 "
        "fuel_drawdown=None min_hold_tp=1 disable_sl_trigger=True "
        "max_hold=7 wr_degrading=False\n"
        "[FROZEN_TP_SL] entry_date=2026-05-15 "
        "bucket=fish_tail_production strategy_id=fish_tail_blow_off_top "
        "symbol=TAIL dollar_volume_rank=0 tp_pct=0.050000 sl_pct=0.000000 "
        "rolling_mp=0.040000 slope_60=0.1 near_delta=0.1 "
        "fuel_drawdown=None min_hold_tp=1 disable_sl_trigger=True "
        "max_hold=7 wr_degrading=False\n",
        encoding="utf-8",
    )
    _patch_dashboard_paths(
        monkeypatch,
        config_path=config_path,
        log_directory=log_directory,
        repository_root=tmp_path,
    )
    _patch_phantom_paths(monkeypatch, tmp_path)
    expectancy_state_path = (
        tmp_path / "live_state" / "expectancy_gate_state.json"
    )
    expectancy_gate_config = (
        live_expectancy_gate.parse_live_expectancy_gate_config(
            json.loads(config_path.read_text(encoding="utf-8"))
        )
    )
    assert expectancy_gate_config is not None
    expectancy_state_document = (
        live_expectancy_gate.empty_expectancy_gate_state_document(
            expectancy_gate_config
        )
    )
    expectancy_state_document["rolling_outcomes"] = [
        {"percentage_change": -0.2},
        {"percentage_change": -0.2},
    ]
    expectancy_state_document["last_evaluation_date"] = signal_date
    live_expectancy_gate.save_expectancy_gate_state_atomically(
        expectancy_state_path,
        expectancy_state_document,
        expectancy_gate_config,
    )
    monkeypatch.setattr(
        dashboard,
        "EXPECTANCY_GATE_STATE_PATH",
        expectancy_state_path,
    )

    preview = dashboard.api_preview_orders()

    assert preview["expectancy_gate"]["status"] == "closed"
    assert preview["orders"][0]["symbol"] == "TAIL"
    assert preview["orders"][0]["status"] == "expectancy_gate_phantom"
    assert preview["orders"][0]["qty"] == 0
    assert preview["orders"][0]["expectancy_priority_override"] is True
    assert preview["orders"][1]["symbol"] == "HEAD"
    assert preview["orders"][1]["status"] == "slot_full"


def test_phantom_still_open_detects_tp_close_and_open_hold(
    tmp_path: Path, monkeypatch
) -> None:
    """_phantom_still_open: a TP hit frees the slot; an unexited position
    within its holding window keeps it."""
    data_directory = _patch_phantom_paths(monkeypatch, tmp_path)
    base = {
        "symbol": "PHX",
        "signal_date": "2026-05-15",
        "fill_date": "2026-05-18",
        "entry_price": 10.0,
        "tp_pct": 0.05,
        "min_hold_tp": 1,
        "max_hold": 7,
    }
    # TP hit at the first post-entry bar (high 11.0 -> +10% >= 5%).
    _write_phantom_csv(
        data_directory,
        "PHX",
        [
            ("2026-05-18", 10.0, 10.1, 9.9),
            ("2026-05-19", 10.1, 11.0, 10.0),
        ],
    )
    assert dashboard._phantom_still_open(dict(base), "2026-05-20") is False

    # No TP, max_hold not reached -> still open.
    _write_phantom_csv(
        data_directory,
        "PHX",
        [
            ("2026-05-18", 10.0, 10.1, 9.9),
            ("2026-05-19", 10.1, 10.2, 10.0),
        ],
    )
    assert dashboard._phantom_still_open(dict(base), "2026-05-20") is True


def test_open_phantom_consumes_a_slot(tmp_path: Path, monkeypatch) -> None:
    """A still-open phantom occupies a slot, so a fresh candidate is
    slot_full when the real+phantom book is at capacity."""
    config_path, log_directory = _write_phantom_fixture(
        tmp_path,
        signal_date="2026-05-15",
        risk_score=20,  # gate OFF so AAA candidate is a normal BUY...
        wr_degrading=False,
        symbol="AAA",
        max_position_count=1,
    )
    _patch_dashboard_paths(
        monkeypatch,
        config_path=config_path,
        log_directory=log_directory,
        repository_root=tmp_path,
    )
    data_directory = _patch_phantom_paths(monkeypatch, tmp_path)
    # An already-open phantom holds the single slot.
    _write_phantom_csv(
        data_directory,
        "OLDPH",
        [
            ("2026-05-11", 10.0, 10.1, 9.9),
            ("2026-05-12", 10.1, 10.2, 10.0),
        ],
    )
    dashboard._save_phantom_positions([
        {
            "symbol": "OLDPH",
            "signal_date": "2026-05-08",
            "fill_date": "2026-05-11",
            "entry_price": 10.0,
            "bucket": "fish_tail_production",
            "tp_pct": 0.05,
            "min_hold_tp": 1,
            "max_hold": 7,
        }
    ])

    order = dashboard.api_preview_orders()["orders"][0]
    assert order["symbol"] == "AAA"
    assert order["status"] == "slot_full"


def test_execute_records_phantom_and_places_no_order(
    tmp_path: Path, monkeypatch
) -> None:
    """Executing a phantom order persists it to the phantom list and never
    calls Futu place_order (status is in PREVIEW_SKIP_STATUSES)."""
    repository_root = tmp_path
    log_directory = tmp_path / "logs"
    log_directory.mkdir(exist_ok=True)
    (log_directory / "2026-05-15.log").write_text(
        "max_position_count=6 held_before_today=0 same_day_closes=0\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "multi_bucket_production.json"
    config_path.write_text(json.dumps({"max_position_count": 6}), encoding="utf-8")

    placed: list[Any] = []

    class _ExecCtx(FakeTradeContext):
        def place_order(self, *args: Any, **kwargs: Any):
            placed.append((args, kwargs))
            return 0, pandas.DataFrame([{"order_id": "X1"}])

        def order_list_query(self, trd_env: Any = None):
            return 0, pandas.DataFrame([], columns=["code", "trd_side", "order_status", "order_id"])

    futu_module = types.SimpleNamespace(
        TrdEnv=types.SimpleNamespace(REAL="REAL"),
        TrdSide=types.SimpleNamespace(BUY="BUY", SELL="SELL"),
        OrderType=types.SimpleNamespace(MARKET="MARKET"),
        ModifyOrderOp=types.SimpleNamespace(CANCEL="CANCEL"),
    )
    monkeypatch.setitem(sys.modules, "futu", futu_module)
    monkeypatch.setattr(dashboard, "PRODUCTION_CONFIG_PATH", config_path)
    monkeypatch.setattr(dashboard, "REPOSITORY_ROOT", repository_root)
    monkeypatch.setattr(dashboard, "DATA_DIRECTORY", repository_root)
    monkeypatch.setattr(dashboard, "LOGS_DIRECTORY", log_directory)
    monkeypatch.setattr(dashboard, "_get_futu_trd_ctx", lambda: _ExecCtx())
    monkeypatch.setattr(dashboard, "_get_trd_env", lambda: object())
    data_directory = _patch_phantom_paths(monkeypatch, repository_root)
    monkeypatch.setattr(dashboard, "_load_risk_score_gate_state", lambda d: {"status": "open"})

    request = dashboard.ExecuteRequest(
        orders=[
            {
                "side": "BUY",
                "symbol": "PHX",
                "qty": 0,
                "status": "wr_gate_phantom",
                "skip_reason": "WR-gate phantom",
                "phantom_record": {
                    "symbol": "PHX",
                    "signal_date": "2026-05-15",
                    "bucket": "fish_tail_production",
                    "tp_pct": 0.05,
                    "min_hold_tp": 1,
                    "max_hold": 7,
                },
            }
        ]
    )
    monkeypatch.setattr(
        dashboard,
        "api_preview_orders",
        lambda: {"orders": [dict(request.orders[0])]},
    )
    response = dashboard.api_execute_orders(request)

    assert placed == []  # no real order placed for a phantom
    assert response["results"][0]["status"] == "skipped"
    stored = dashboard._load_phantom_positions()
    assert [p["symbol"] for p in stored] == ["PHX"]
    # Re-executing the same phantom must not double-book the slot.
    dashboard.api_execute_orders(request)
    assert len(dashboard._load_phantom_positions()) == 1


def test_execute_records_funded_buy_in_expectancy_sensor_before_broker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A funded confirmed allocation enters the sensor before Futu submit."""

    config_document = {
        "max_position_count": 1,
        "expectancy_gate": {
            "enabled": True,
            "window": 2,
            "baseline_mean": 0.0,
            "baseline_sigma": 0.1,
            "sigma_multiplier": 1.0,
            "cold_start": "open",
        },
        "buckets": [{"label": "fish_head_production"}],
    }
    config_path = tmp_path / "multi_bucket_production.json"
    config_path.write_text(json.dumps(config_document), encoding="utf-8")
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    (log_directory / "2026-05-15.log").write_text(
        "[multi_bucket_daily_signal mode=live]\n",
        encoding="utf-8",
    )
    state_path = tmp_path / "live_state" / "expectancy_gate_state.json"
    gate_config = live_expectancy_gate.parse_live_expectancy_gate_config(
        config_document
    )
    assert gate_config is not None
    expectancy_state_document = (
        live_expectancy_gate.empty_expectancy_gate_state_document(
            gate_config
        )
    )
    expectancy_state_document["last_evaluation_date"] = "2026-05-15"
    live_expectancy_gate.save_expectancy_gate_state_atomically(
        state_path, expectancy_state_document, gate_config
    )
    monkeypatch.setattr(dashboard, "PRODUCTION_CONFIG_PATH", config_path)
    monkeypatch.setattr(dashboard, "LOGS_DIRECTORY", log_directory)
    monkeypatch.setattr(dashboard, "EXPECTANCY_GATE_STATE_PATH", state_path)
    monkeypatch.setattr(
        dashboard,
        "_load_risk_score_gate_state",
        lambda _signal_date: {"status": "open"},
    )
    operation_order: list[str] = []

    class _BuyContext(FakeTradeContext):
        def place_order(self, **_keyword_arguments: Any):
            operation_order.append("broker_submit")
            return 0, pandas.DataFrame([{"order_id": "BUY-1"}])

    monkeypatch.setattr(
        dashboard, "_get_futu_trd_ctx", lambda: _BuyContext()
    )
    monkeypatch.setattr(dashboard, "_get_trd_env", lambda: object())
    futu_module = types.SimpleNamespace(
        TrdSide=types.SimpleNamespace(BUY="BUY", SELL="SELL"),
        OrderType=types.SimpleNamespace(MARKET="MARKET"),
        ModifyOrderOp=types.SimpleNamespace(CANCEL="CANCEL"),
    )
    monkeypatch.setitem(sys.modules, "futu", futu_module)
    order = {
        "side": "BUY",
        "symbol": "AAA",
        "qty": 5,
        "bucket": "fish_head_production",
        "strategy_id": "fish_head_vacuum_turn",
        "signal_date": "2026-05-15",
        "tp_pct": 0.05,
        "sl_pct": 0.03,
        "min_hold_tp": 1,
        "min_hold_sl": 1,
        "disable_sl_trigger": True,
        "max_hold": 7,
        "expectancy_gated": False,
        "expectancy_priority_override": False,
    }
    monkeypatch.setattr(
        dashboard,
        "api_preview_orders",
        lambda: {"orders": [dict(order)]},
    )
    original_record_function = (
        dashboard.live_expectancy_gate.record_accepted_trades_at_path
    )

    def record_then_mark(**keyword_arguments: Any) -> int:
        operation_order.append("sensor_acceptance")
        return original_record_function(**keyword_arguments)

    monkeypatch.setattr(
        dashboard.live_expectancy_gate,
        "record_accepted_trades_at_path",
        record_then_mark,
    )

    response = dashboard.api_execute_orders(
        dashboard.ExecuteRequest(orders=[order])
    )

    assert response["results"][0]["status"] == "sent"
    assert operation_order == ["sensor_acceptance", "broker_submit"]
    state_document = live_expectancy_gate.load_expectancy_gate_state(
        state_path, gate_config
    )
    assert state_document["pending_trades"][0]["symbol"] == "AAA"


# TODO: review
def test_successful_dashboard_buy_does_not_write_adaptive_tp_sl_history(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Real execution must not admit or reconcile statistical observations."""

    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    (log_directory / "2026-05-15.log").write_text(
        "[multi_bucket_daily_signal] eval_date=2026-05-15\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "multi_bucket_production.json"
    config_path.write_text(
        json.dumps({"max_position_count": 6}),
        encoding="utf-8",
    )
    _patch_dashboard_paths(
        monkeypatch,
        config_path=config_path,
        log_directory=log_directory,
        repository_root=tmp_path,
    )
    _patch_phantom_paths(monkeypatch, tmp_path)

    adaptive_state_path = (
        tmp_path / "live_state" / "adaptive_state.json"
    )
    original_adaptive_state_text = json.dumps({
        "schema_version": 2,
        "adaptive_tp_sl_virtual_trade_history": {
            "schema_version": 1,
            "open_trades": [{"symbol": "REFERENCE_ONLY"}],
            "closed_trades": [],
            "pending_returns": [],
            "winner_returns": [],
            "loser_returns": [],
            "raw_returns": [],
        },
    })
    adaptive_state_path.write_text(
        original_adaptive_state_text,
        encoding="utf-8",
    )

    placed_orders: list[dict[str, Any]] = []

    class SuccessfulBuyTradeContext(FakeTradeContext):
        """Record one successful market BUY for boundary verification."""

        def place_order(self, **order_arguments: Any):
            placed_orders.append(order_arguments)
            return 0, pandas.DataFrame([{"order_id": "BUY-1"}])

    futu_module = types.SimpleNamespace(
        TrdSide=types.SimpleNamespace(BUY="BUY", SELL="SELL"),
        OrderType=types.SimpleNamespace(MARKET="MARKET"),
        ModifyOrderOp=types.SimpleNamespace(CANCEL="CANCEL"),
    )
    monkeypatch.setitem(sys.modules, "futu", futu_module)
    monkeypatch.setattr(
        dashboard,
        "_get_futu_trd_ctx",
        lambda: SuccessfulBuyTradeContext(),
    )
    monkeypatch.setattr(dashboard, "_get_trd_env", lambda: object())
    monkeypatch.setattr(
        dashboard,
        "_load_risk_score_gate_state",
        lambda signal_date_text: {"status": "open"},
    )

    canonical_buy_order = {
        "side": "BUY",
        "symbol": "REALBUY",
        "signal_symbol": "REALBUY",
        "qty": 1,
        "bucket": "fish_head_production",
        "strategy_id": "fish_head_vacuum_turn",
        "tp_pct": 0.05,
        "sl_pct": 0.03,
        "min_hold_sl": 1,
        "disable_sl_trigger": True,
        "reset_hold_on_reentry_signal": False,
    }
    monkeypatch.setattr(
        dashboard,
        "api_preview_orders",
        lambda: {"orders": [dict(canonical_buy_order)]},
    )

    response = dashboard.api_execute_orders(
        dashboard.ExecuteRequest(orders=[dict(canonical_buy_order)])
    )

    assert response["results"][0]["status"] == "sent", response
    assert len(placed_orders) == 1
    assert adaptive_state_path.read_text(
        encoding="utf-8"
    ) == original_adaptive_state_text


def test_failed_real_buy_persists_intent_for_pre_open_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A buying-power rejection must retain the confirmed BUY intention."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    (log_directory / "2026-05-15.log").write_text(
        "[multi_bucket_daily_signal] eval_date=2026-05-15\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "multi_bucket_production.json"
    config_path.write_text(
        json.dumps({"max_position_count": 6}),
        encoding="utf-8",
    )
    _patch_dashboard_paths(
        monkeypatch,
        config_path=config_path,
        log_directory=log_directory,
        repository_root=tmp_path,
    )
    _patch_phantom_paths(monkeypatch, tmp_path)

    class RejectedBuyTradeContext(FakeTradeContext):
        """Reject one market BUY with the incident's broker error."""

        def place_order(self, **order_arguments: Any):
            return 1, "Insufficient buying power"

    futu_module = types.SimpleNamespace(
        TrdSide=types.SimpleNamespace(BUY="BUY", SELL="SELL"),
        OrderType=types.SimpleNamespace(MARKET="MARKET"),
        ModifyOrderOp=types.SimpleNamespace(CANCEL="CANCEL"),
    )
    monkeypatch.setitem(sys.modules, "futu", futu_module)
    monkeypatch.setattr(
        dashboard,
        "_get_futu_trd_ctx",
        lambda: RejectedBuyTradeContext(),
    )
    monkeypatch.setattr(dashboard, "_get_trd_env", lambda: object())
    monkeypatch.setattr(dashboard, "TRADING_ENV", "REAL")
    monkeypatch.setattr(
        dashboard,
        "_load_risk_score_gate_state",
        lambda signal_date_text: {"status": "open"},
    )
    canonical_buy_order = {
        "side": "BUY",
        "symbol": "OKLO",
        "qty": 108,
        "bucket": "fish_head_production",
        "strategy_id": "fish_head_vacuum_turn",
        "tp_pct": 0.0784,
        "sl_pct": 0.0335,
        "min_hold_sl": 1,
        "disable_sl_trigger": True,
        "reset_hold_on_reentry_signal": False,
    }
    monkeypatch.setattr(
        dashboard,
        "api_preview_orders",
        lambda: {"orders": [dict(canonical_buy_order)]},
    )

    response = dashboard.api_execute_orders(
        dashboard.ExecuteRequest(orders=[dict(canonical_buy_order)])
    )

    assert response["results"][0]["status"] == "failed"
    intent_state_path = (
        tmp_path / "live_state" / "pending_entry_intents.json"
    )
    intent_state = json.loads(intent_state_path.read_text(encoding="utf-8"))
    persisted_intent = intent_state["intents"][0]
    assert persisted_intent["symbol"] == "OKLO"
    assert persisted_intent["requested_qty"] == 108
    assert persisted_intent["status"] == "failed"
    assert persisted_intent["error"] == "Insufficient buying power"


def test_execute_rejects_order_not_in_fresh_server_preview(monkeypatch) -> None:
    """Arbitrary client JSON must never reach the Futu trade context."""
    context_created = False

    def create_trade_context() -> Any:
        nonlocal context_created
        context_created = True
        return FakeTradeContext()

    monkeypatch.setattr(dashboard, "_get_futu_trd_ctx", create_trade_context)
    monkeypatch.setattr(dashboard, "api_preview_orders", lambda: {"orders": []})

    response = dashboard.api_execute_orders(
        dashboard.ExecuteRequest(
            orders=[{"side": "BUY", "symbol": "ARBITRARY", "qty": 999999}]
        )
    )

    assert context_created is False
    assert response["results"] == [
        {
            "symbol": "ARBITRARY",
            "status": "rejected",
            "reason": "order is not present in the current server preview",
        }
    ]
