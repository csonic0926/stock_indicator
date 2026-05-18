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
            columns=["order_id", "remark"],
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


def test_place_tp_sl_uses_futu_v2_remark_and_honors_disable_sl(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """V2 Futu remark metadata should place TP and suppress disabled SL."""
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
    assert len(fake_context.placed_orders) == 1
    take_profit_order = fake_context.placed_orders[0]
    assert take_profit_order["code"] == "US.HELD"
    assert take_profit_order["order_type"] == "NORMAL"
    assert take_profit_order["price"] == 106.58


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

    place_tp_sl.main()

    assert fake_context.placed_orders == []
    assert "[ORPHAN_POSITION]" in caplog.text


def test_place_tp_sl_treats_legacy_remark_as_orphan_for_tp_sl(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    """Legacy remarks support max-hold only and must not drive TP/SL."""
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

    assert fake_context.placed_orders == []
    assert "[ORPHAN_POSITION]" in caplog.text
