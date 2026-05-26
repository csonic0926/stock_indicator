"""Tests for live TP/SL placement safety gates."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pandas

from stock_indicator import place_tp_sl


class FakeTradeContext:
    """Minimal Futu trade context for TP/SL placement tests."""

    def __init__(
        self,
        *,
        positions: list[dict[str, Any]] | None = None,
        orders: list[dict[str, Any]] | None = None,
        historical_deals: list[dict[str, Any]] | None = None,
        historical_orders: list[dict[str, Any]] | None = None,
    ) -> None:
        self.positions = positions or []
        self.orders = orders or []
        self.historical_deals = historical_deals or []
        self.historical_orders = historical_orders or []
        self.placed_orders: list[dict[str, Any]] = []
        self.is_closed = False

    def position_list_query(self, trd_env: Any = None) -> tuple[int, pandas.DataFrame]:
        """Return deterministic live positions."""
        return 0, pandas.DataFrame(
            self.positions,
            columns=["code", "qty", "cost_price"],
        )

    def order_list_query(self, trd_env: Any = None) -> tuple[int, pandas.DataFrame]:
        """Return deterministic open orders."""
        return 0, pandas.DataFrame(
            self.orders,
            columns=["code", "trd_side", "order_status", "order_type"],
        )

    def history_deal_list_query(
        self,
        code: str = "",
        start: str = "",
        end: str = "",
        trd_env: Any = None,
    ) -> tuple[int, pandas.DataFrame]:
        """Return deterministic historical deals."""
        return 0, pandas.DataFrame(
            self.historical_deals,
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
        """Return deterministic historical order remarks."""
        return 0, pandas.DataFrame(
            self.historical_orders,
            columns=["order_id", "remark", "create_time"],
        )

    def place_order(self, **kwargs: Any) -> tuple[int, pandas.DataFrame]:
        """Capture order placement requests."""
        self.placed_orders.append(kwargs)
        return 0, pandas.DataFrame([{"order_id": str(len(self.placed_orders))}])

    def close(self) -> None:
        """Match the Futu context close contract."""
        self.is_closed = True


def _install_fake_futu_module(monkeypatch, fake_context: FakeTradeContext) -> None:
    """Install a deterministic futu module for place_tp_sl.main."""
    fake_futu_module = types.SimpleNamespace(
        OpenSecTradeContext=lambda **_kwargs: fake_context,
        OrderType=types.SimpleNamespace(NORMAL="NORMAL", STOP="STOP"),
        SecurityFirm=types.SimpleNamespace(FUTUSECURITIES="FUTUSECURITIES"),
        TimeInForce=types.SimpleNamespace(GTC="GTC"),
        TrdEnv=types.SimpleNamespace(REAL="REAL", SIMULATE="SIMULATE"),
        TrdMarket=types.SimpleNamespace(US="US"),
        TrdSide=types.SimpleNamespace(SELL="SELL"),
    )
    monkeypatch.setitem(sys.modules, "futu", fake_futu_module)


def test_entry_disables_stop_loss_trigger_when_flag_is_true() -> None:
    """SL order placement must follow the Futu entry disable flag."""
    entry_record: dict[str, Any] = {"disable_sl_trigger": True}

    assert place_tp_sl._entry_disables_stop_loss_trigger(entry_record) is True


def test_entry_keeps_stop_loss_enabled_when_flag_is_absent() -> None:
    """Entries keep SL enabled unless the Futu remark disables it."""
    entry_record: dict[str, Any] = {}

    assert place_tp_sl._entry_disables_stop_loss_trigger(entry_record) is False


def test_place_tp_sl_does_not_require_or_trust_futu_v2_remark_for_tp_sl(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Futu remarks are debug metadata, not TP/SL placement source."""
    fake_context = FakeTradeContext(
        positions=[{"code": "US.HELD", "qty": 10, "cost_price": 100.0}],
        historical_deals=[
            {
                "code": "US.HELD",
                "trd_side": "BUY",
                "qty": 10,
                "create_time": "2026-01-02 09:30:00",
                "order_id": "1001",
            }
        ],
        historical_orders=[
            {
                "order_id": "1001",
                "remark": "si2|s=b|tp=658|sl=417|ms=1|ds=1|mh=14|rr=1",
            }
        ],
    )
    _install_fake_futu_module(monkeypatch, fake_context)
    monkeypatch.setattr(place_tp_sl, "LOGS_DIRECTORY", tmp_path)

    place_tp_sl.main()

    assert fake_context.is_closed is True
    assert fake_context.placed_orders == []


def test_place_tp_sl_does_not_duplicate_existing_orders(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Existing Futu TP/SL orders should prevent duplicate placements."""
    fake_context = FakeTradeContext(
        positions=[{"code": "US.HELD", "qty": 10, "cost_price": 100.0}],
        orders=[
            {
                "code": "US.HELD",
                "trd_side": "SELL",
                "order_status": "SUBMITTED",
                "order_type": "NORMAL",
            },
            {
                "code": "US.HELD",
                "trd_side": "SELL",
                "order_status": "SUBMITTED",
                "order_type": "STOP",
            },
        ],
        historical_deals=[
            {
                "code": "US.HELD",
                "trd_side": "BUY",
                "qty": 10,
                "create_time": "2026-01-02 09:30:00",
                "order_id": "1001",
            }
        ],
        historical_orders=[
            {
                "order_id": "1001",
                "remark": "si2|s=h|tp=500|sl=300|ms=1|ds=0|rr=0",
            }
        ],
    )
    _install_fake_futu_module(monkeypatch, fake_context)
    monkeypatch.setattr(place_tp_sl, "LOGS_DIRECTORY", tmp_path)

    place_tp_sl.main()

    assert fake_context.placed_orders == []


def test_place_tp_sl_does_not_use_local_accepted_entries(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    """Local accepted_entries alone must not create live TP/SL orders."""
    (tmp_path / "adaptive_state.json").write_text(
        json.dumps(
            {
                "accepted_entries": [
                    {
                        "symbol": "LOCAL_ONLY",
                        "tp_pct": 0.05,
                        "sl_pct": 0.03,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    fake_context = FakeTradeContext(
        positions=[{"code": "US.LOCAL_ONLY", "qty": 10, "cost_price": 100.0}],
        historical_deals=[],
    )
    _install_fake_futu_module(monkeypatch, fake_context)
    monkeypatch.setattr(place_tp_sl, "LOGS_DIRECTORY", tmp_path)
    monkeypatch.setattr(place_tp_sl, "DATA_DIRECTORY", tmp_path)

    place_tp_sl.main()

    assert fake_context.placed_orders == []
    assert "[TP_SL_METADATA_MISSING]" in caplog.text


def test_place_tp_sl_uses_api_position_and_production_signal_without_remark(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Futu position plus production signal metadata should place TP."""
    (tmp_path / "2026-05-18.log").write_text(
        "[FROZEN_TP_SL] entry_date=2026-05-18 "
        "bucket=fish_tail_explore strategy_id=fish_tail_blow_off_top "
        "symbol=NO_REMARK tp_pct=0.167497 sl_pct=0.041702 "
        "min_hold_sl=1 disable_sl_trigger=True\n",
        encoding="utf-8",
    )
    fake_context = FakeTradeContext(
        positions=[{"code": "US.NO_REMARK", "qty": 3, "cost_price": 1430.55}],
        historical_deals=[
            {
                "code": "US.NO_REMARK",
                "trd_side": "BUY",
                "qty": 3,
                "create_time": "2026-05-18 09:30:00",
                "order_id": "3001",
            }
        ],
        historical_orders=[
            {
                "order_id": "3001",
                "remark": "",
            }
        ],
    )
    _install_fake_futu_module(monkeypatch, fake_context)
    monkeypatch.setattr(place_tp_sl, "LOGS_DIRECTORY", tmp_path)
    monkeypatch.setattr(sys, "argv", ["place_tp_sl"])

    place_tp_sl.main()

    assert len(fake_context.placed_orders) == 1
    take_profit_order = fake_context.placed_orders[0]
    assert take_profit_order["code"] == "US.NO_REMARK"
    assert take_profit_order["qty"] == 3
    assert take_profit_order["price"] == 1670.16


def test_futu_open_entries_sort_deals_before_fifo() -> None:
    """Newest-first Futu deal history must not attach stale BUY metadata."""
    fake_context = FakeTradeContext(
        historical_deals=[
            {
                "code": "US.SNDK",
                "trd_side": "BUY",
                "qty": 3,
                "create_time": "2026-05-18 09:30:01",
                "order_id": "new_buy",
            },
            {
                "code": "US.SNDK",
                "trd_side": "SELL",
                "qty": 14,
                "create_time": "2026-03-23 09:30:01",
                "order_id": "old_sell",
            },
            {
                "code": "US.SNDK",
                "trd_side": "BUY",
                "qty": 14,
                "create_time": "2026-03-17 09:30:00",
                "order_id": "old_buy",
            },
        ],
        historical_orders=[
            {"order_id": "new_buy", "remark": ""},
            {"order_id": "old_buy", "remark": "si|sid=old|b=old|mh=10|rr=0"},
        ],
    )

    open_entries = place_tp_sl._load_futu_open_trade_entries(
        fake_context,
        "REAL",
        as_of_date_text="2026-05-18",
    )

    assert open_entries["SNDK"]["futu_buy_order_id"] == "new_buy"


def test_signal_metadata_can_match_futu_order_create_date() -> None:
    """Production signal matching should use API order date before fill date."""
    fake_context = FakeTradeContext(
        historical_deals=[
            {
                "code": "US.PEP",
                "trd_side": "BUY",
                "qty": 36,
                "create_time": "2026-05-18 09:30:01",
                "order_id": "buy_order",
            },
        ],
        historical_orders=[
            {
                "order_id": "buy_order",
                "remark": "",
                "create_time": "2026-05-15 21:20:52",
            },
        ],
    )

    open_entries = place_tp_sl._load_futu_open_trade_entries(
        fake_context,
        "REAL",
        as_of_date_text="2026-05-18",
    )
    merged_entry = place_tp_sl._merge_production_signal_metadata(
        symbol="PEP",
        entry=open_entries["PEP"],
        signal_entries_by_symbol_and_date={
            (
                "PEP",
                "2026-05-15",
            ): {
                "symbol": "PEP",
                "entry_date": "2026-05-15",
                "bucket": "fish_head_production",
                "strategy_id": "fish_head_vacuum_turn",
                "tp_pct": 0.0658,
                "sl_pct": 0.0417,
            }
        },
        bucket_tp_sl_entries_by_key_and_date={},
        production_exit_rules={},
    )

    assert merged_entry is not None
    assert merged_entry["supports_tp_sl"] is True
    assert merged_entry["tp_pct"] == 0.0658


def test_signal_metadata_can_fallback_to_entry_date_bucket_snapshot() -> None:
    """TP/SL metadata can come from the Futu entry date's bucket log snapshot."""
    merged_entry = place_tp_sl._merge_production_signal_metadata(
        symbol="PEP",
        entry={
            "symbol": "PEP",
            "entry_date": "2026-05-15",
            "bucket": "fish_head_production",
            "strategy_id": "fish_head_vacuum_turn",
        },
        signal_entries_by_symbol_and_date={},
        bucket_tp_sl_entries_by_key_and_date={
            (
                "fish_head_production",
                "2026-05-15",
            ): {
                "date": "2026-05-15",
                "bucket": "fish_head_production",
                "strategy_id": "fish_head_vacuum_turn",
                "tp_pct": 0.0742,
                "sl_pct": 0.0434,
                "min_hold_sl": 1,
                "disable_sl_trigger": True,
            }
        },
        production_exit_rules={},
    )

    assert merged_entry is not None
    assert merged_entry["supports_tp_sl"] is True
    assert merged_entry["tp_pct"] == 0.0742
    assert merged_entry["sl_pct"] == 0.0434


def test_bucket_snapshot_walks_back_when_anchor_date_has_no_entry() -> None:
    """Holiday cron gap: anchor 05-26 but only 05-22 has a snapshot."""
    merged_entry = place_tp_sl._merge_production_signal_metadata(
        symbol="UNH",
        entry={
            "symbol": "UNH",
            "entry_date": "2026-05-26",
            "bucket": "fish_tail_explore",
            "strategy_id": "fish_tail_blow_off_top",
        },
        signal_entries_by_symbol_and_date={},
        bucket_tp_sl_entries_by_key_and_date={
            (
                "fish_tail_explore",
                "2026-05-22",
            ): {
                "date": "2026-05-22",
                "bucket": "fish_tail_explore",
                "strategy_id": "fish_tail_blow_off_top",
                "tp_pct": 0.0586,
                "sl_pct": 0.0420,
            }
        },
        production_exit_rules={},
    )

    assert merged_entry is not None
    assert merged_entry["supports_tp_sl"] is True
    assert merged_entry["tp_pct"] == 0.0586


def test_bucket_snapshot_walk_back_rejects_older_than_cap() -> None:
    """A snapshot older than BUCKET_TP_SL_WALK_BACK_DAYS must not match."""
    merged_entry = place_tp_sl._merge_production_signal_metadata(
        symbol="UNH",
        entry={
            "symbol": "UNH",
            "entry_date": "2026-05-26",
            "bucket": "fish_tail_explore",
            "strategy_id": "fish_tail_blow_off_top",
        },
        signal_entries_by_symbol_and_date={},
        bucket_tp_sl_entries_by_key_and_date={
            (
                "fish_tail_explore",
                "2026-05-11",
            ): {
                "date": "2026-05-11",
                "bucket": "fish_tail_explore",
                "strategy_id": "fish_tail_blow_off_top",
                "tp_pct": 0.0586,
                "sl_pct": 0.0420,
            }
        },
        production_exit_rules={},
    )

    assert merged_entry is None


def test_bucket_snapshot_walk_back_prefers_closest_date() -> None:
    """When multiple snapshots are within cap, pick the most recent one."""
    merged_entry = place_tp_sl._merge_production_signal_metadata(
        symbol="UNH",
        entry={
            "symbol": "UNH",
            "entry_date": "2026-05-26",
            "bucket": "fish_tail_explore",
            "strategy_id": "fish_tail_blow_off_top",
        },
        signal_entries_by_symbol_and_date={},
        bucket_tp_sl_entries_by_key_and_date={
            (
                "fish_tail_explore",
                "2026-05-22",
            ): {
                "date": "2026-05-22",
                "bucket": "fish_tail_explore",
                "strategy_id": "fish_tail_blow_off_top",
                "tp_pct": 0.0586,
                "sl_pct": 0.0420,
            },
            (
                "fish_tail_explore",
                "2026-05-25",
            ): {
                "date": "2026-05-25",
                "bucket": "fish_tail_explore",
                "strategy_id": "fish_tail_blow_off_top",
                "tp_pct": 0.0633,
                "sl_pct": 0.0420,
            },
        },
        production_exit_rules={},
    )

    assert merged_entry is not None
    assert merged_entry["tp_pct"] == 0.0633


def test_legacy_remark_does_not_block_production_signal_tp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A legacy remark should not block production-config TP placement."""
    (tmp_path / "2026-01-02.log").write_text(
        "[FROZEN_TP_SL] entry_date=2026-01-02 "
        "bucket=fish_head_b30_35 strategy_id=fish_head_b30_35 "
        "symbol=LEGACY tp_pct=0.0658 sl_pct=0.0417 "
        "min_hold_sl=1 disable_sl_trigger=True\n",
        encoding="utf-8",
    )
    fake_context = FakeTradeContext(
        positions=[{"code": "US.LEGACY", "qty": 10, "cost_price": 100.0}],
        historical_deals=[
            {
                "code": "US.LEGACY",
                "trd_side": "BUY",
                "qty": 10,
                "create_time": "2026-01-02 09:30:00",
                "order_id": "2001",
            }
        ],
        historical_orders=[
            {
                "order_id": "2001",
                "remark": "si|sid=fish_head_b30_35|b=fish_head_b30_35|mh=10|rr=0",
            }
        ],
    )
    _install_fake_futu_module(monkeypatch, fake_context)
    monkeypatch.setattr(place_tp_sl, "LOGS_DIRECTORY", tmp_path)

    place_tp_sl.main()

    assert len(fake_context.placed_orders) == 1
    assert fake_context.placed_orders[0]["price"] == 106.58
