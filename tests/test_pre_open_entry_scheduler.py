"""Tests for pre-open market-entry quantity reconciliation."""

from __future__ import annotations

# TODO: review

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas
import pytest

from stock_indicator import pre_open_entry_scheduler


class FakeQuoteContext:
    """Return deterministic real-time and snapshot quote rows."""

    def __init__(self) -> None:
        self.subscribed_codes: list[str] = []

    def subscribe(
        self,
        code_list: list[str],
        subtype_list: list[Any],
        subscribe_push: bool = False,
    ) -> tuple[int, None]:
        """Capture quote subscriptions."""
        self.subscribed_codes = code_list
        return 0, None

    def get_stock_quote(
        self,
        code_list: list[str],
    ) -> tuple[int, pandas.DataFrame]:
        """Return live pre-market trade prices."""
        return 0, pandas.DataFrame(
            [
                {"code": "US.BE", "pre_price": 200.0},
                {"code": "US.OKLO", "pre_price": 40.0},
            ]
        )

    def get_market_snapshot(
        self,
        code_list: list[str],
    ) -> tuple[int, pandas.DataFrame]:
        """Return fallback snapshot fields."""
        return 0, pandas.DataFrame(
            [
                {
                    "code": "US.BE",
                    "pre_price": 200.0,
                    "ask_price": 201.0,
                    "last_price": 198.0,
                },
                {
                    "code": "US.OKLO",
                    "pre_price": 40.0,
                    "ask_price": 40.1,
                    "last_price": 39.0,
                },
            ]
        )


class FakeTradeContext:
    """Capture pre-open order resizing and retry requests."""

    def __init__(self) -> None:
        self.maximum_quantity_queries: list[dict[str, Any]] = []
        self.modified_orders: list[dict[str, Any]] = []
        self.placed_orders: list[dict[str, Any]] = []

    def accinfo_query(self, trd_env: Any) -> tuple[int, pandas.DataFrame]:
        """Return account assets in HKD."""
        return 0, pandas.DataFrame([{"total_assets": 78_000.0}])

    def order_list_query(
        self,
        trd_env: Any,
        refresh_cache: bool,
    ) -> tuple[int, pandas.DataFrame]:
        """Return one accepted market BUY awaiting the open."""
        return 0, pandas.DataFrame(
            [
                {
                    "code": "US.BE",
                    "trd_side": "BUY",
                    "order_type": "MARKET",
                    "order_status": "SUBMITTED",
                    "order_id": "be_order",
                    "qty": 22,
                }
            ]
        )

    def position_list_query(
        self,
        trd_env: Any,
    ) -> tuple[int, pandas.DataFrame]:
        """Return no filled positions before the regular session."""
        return 0, pandas.DataFrame(columns=["code", "qty"])

    def acctradinginfo_query(self, **kwargs: Any) -> tuple[int, pandas.DataFrame]:
        """Return broker-safe quantities for modify and new-order paths."""
        self.maximum_quantity_queries.append(kwargs)
        maximum_quantity = 30 if kwargs.get("order_id") else 45
        return 0, pandas.DataFrame(
            [{"max_cash_and_margin_buy": maximum_quantity}]
        )

    def modify_order(self, **kwargs: Any) -> tuple[int, pandas.DataFrame]:
        """Capture an accepted-order quantity modification."""
        self.modified_orders.append(kwargs)
        return 0, pandas.DataFrame([{"order_id": kwargs["order_id"]}])

    def place_order(self, **kwargs: Any) -> tuple[int, pandas.DataFrame]:
        """Capture a retry for a previously rejected BUY intent."""
        self.placed_orders.append(kwargs)
        return 0, pandas.DataFrame([{"order_id": "oklo_retry"}])


def _write_entry_intents(state_path: Path) -> None:
    """Write accepted and rejected confirmed BUY intentions."""
    state_path.write_text(
        json.dumps(
            {
                "intents": [
                    {
                        "market_date": "2026-07-21",
                        "symbol": "BE",
                        "requested_qty": 22,
                        "remark": "si2|s=h|tp=784|sl=335|ms=1|ds=1|rr=0",
                        "status": "submitted",
                        "order_id": "be_order",
                    },
                    {
                        "market_date": "2026-07-21",
                        "symbol": "OKLO",
                        "requested_qty": 108,
                        "remark": "si2|s=h|tp=784|sl=335|ms=1|ds=1|rr=0",
                        "status": "failed",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def test_pre_open_scheduler_runs_only_inside_new_york_window(
    tmp_path: Path,
) -> None:
    """The entry adjustment runs at 09:27-09:29 and never after the open."""
    scheduler_state_path = tmp_path / "scheduler_state.json"
    invocation_times: list[datetime] = []

    before_window = pre_open_entry_scheduler.run_due_pre_open_entry_reconciliation(
        current_time=datetime(
            2026,
            7,
            21,
            9,
            26,
            tzinfo=pre_open_entry_scheduler.NEW_YORK_TIME_ZONE,
        ),
        scheduler_state_path=scheduler_state_path,
        reconciliation_function=invocation_times.append,
    )
    inside_window = pre_open_entry_scheduler.run_due_pre_open_entry_reconciliation(
        current_time=datetime(
            2026,
            7,
            21,
            9,
            27,
            tzinfo=pre_open_entry_scheduler.NEW_YORK_TIME_ZONE,
        ),
        scheduler_state_path=scheduler_state_path,
        reconciliation_function=invocation_times.append,
    )
    repeated_window = pre_open_entry_scheduler.run_due_pre_open_entry_reconciliation(
        current_time=datetime(
            2026,
            7,
            21,
            9,
            28,
            tzinfo=pre_open_entry_scheduler.NEW_YORK_TIME_ZONE,
        ),
        scheduler_state_path=scheduler_state_path,
        reconciliation_function=invocation_times.append,
    )
    after_open = pre_open_entry_scheduler.run_due_pre_open_entry_reconciliation(
        current_time=datetime(
            2026,
            7,
            22,
            9,
            30,
            tzinfo=pre_open_entry_scheduler.NEW_YORK_TIME_ZONE,
        ),
        scheduler_state_path=tmp_path / "next_day_state.json",
        reconciliation_function=invocation_times.append,
    )

    assert before_window is False
    assert inside_window is True
    assert repeated_window is False
    assert after_open is False
    assert len(invocation_times) == 1


def test_failed_pre_open_run_can_retry_before_open(tmp_path: Path) -> None:
    """A failed 09:27 attempt remains eligible at 09:28."""
    scheduler_state_path = tmp_path / "scheduler_state.json"
    invocation_count = 0

    def reconcile_with_one_failure(current_time: datetime) -> None:
        nonlocal invocation_count
        invocation_count += 1
        if invocation_count == 1:
            raise RuntimeError("temporary Futu error")

    with pytest.raises(RuntimeError, match="temporary Futu error"):
        pre_open_entry_scheduler.run_due_pre_open_entry_reconciliation(
            current_time=datetime(
                2026,
                7,
                21,
                9,
                27,
                tzinfo=pre_open_entry_scheduler.NEW_YORK_TIME_ZONE,
            ),
            scheduler_state_path=scheduler_state_path,
            reconciliation_function=reconcile_with_one_failure,
        )

    was_retried = pre_open_entry_scheduler.run_due_pre_open_entry_reconciliation(
        current_time=datetime(
            2026,
            7,
            21,
            9,
            28,
            tzinfo=pre_open_entry_scheduler.NEW_YORK_TIME_ZONE,
        ),
        scheduler_state_path=scheduler_state_path,
        reconciliation_function=reconcile_with_one_failure,
    )

    assert was_retried is True
    assert invocation_count == 2


def test_reconciliation_modifies_accepted_order_and_retries_rejected_intent(
    tmp_path: Path,
) -> None:
    """Use pre-market prices and Futu max-buy values for both order paths."""
    state_path = tmp_path / "pending_entry_intents.json"
    config_path = tmp_path / "multi_bucket_production.json"
    _write_entry_intents(state_path)
    config_path.write_text(
        json.dumps({"max_position_count": 7}),
        encoding="utf-8",
    )
    trade_context = FakeTradeContext()
    quote_context = FakeQuoteContext()

    pre_open_entry_scheduler.reconcile_pre_open_entry_orders(
        trade_context=trade_context,
        quote_context=quote_context,
        trading_environment="REAL",
        market_order_type="MARKET",
        buy_side="BUY",
        normal_modify_operation="NORMAL",
        regular_session="RTH",
        quote_subtype="QUOTE",
        current_time=datetime(
            2026,
            7,
            21,
            9,
            28,
            tzinfo=pre_open_entry_scheduler.NEW_YORK_TIME_ZONE,
        ),
        state_path=state_path,
        config_path=config_path,
    )

    assert quote_context.subscribed_codes == ["US.BE", "US.OKLO"]
    assert trade_context.maximum_quantity_queries[0]["order_id"] == "be_order"
    assert "order_id" not in trade_context.maximum_quantity_queries[1]
    assert trade_context.modified_orders[0]["order_id"] == "be_order"
    assert trade_context.modified_orders[0]["qty"] == 10
    assert trade_context.placed_orders[0]["code"] == "US.OKLO"
    assert trade_context.placed_orders[0]["qty"] == 45
    assert trade_context.placed_orders[0]["session"] == "RTH"

    persisted_state = json.loads(state_path.read_text(encoding="utf-8"))
    persisted_intents = {
        intent["symbol"]: intent for intent in persisted_state["intents"]
    }
    assert persisted_intents["BE"]["status"] == "quantity_adjusted"
    assert persisted_intents["BE"]["reference_price_source"] == "pre_price"
    assert persisted_intents["OKLO"]["status"] == "retry_submitted"
    assert persisted_intents["OKLO"]["order_id"] == "oklo_retry"


def test_reconciliation_does_not_resubmit_intentionally_cancelled_entry(
    tmp_path: Path,
) -> None:
    """A user's cancelled market BUY must stay cancelled."""
    state_path = tmp_path / "pending_entry_intents.json"
    config_path = tmp_path / "multi_bucket_production.json"
    state_path.write_text(
        json.dumps(
            {
                "intents": [
                    {
                        "market_date": "2026-07-21",
                        "symbol": "BE",
                        "requested_qty": 22,
                        "remark": (
                            "si2|s=h|tp=784|sl=335|ms=1|ds=1|rr=0"
                        ),
                        "status": "submitted",
                        "order_id": "cancelled_be_order",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps({"max_position_count": 7}),
        encoding="utf-8",
    )

    class CancelledOrderTradeContext(FakeTradeContext):
        """Expose the confirmed order as explicitly cancelled."""

        def order_list_query(
            self,
            trd_env: Any,
            refresh_cache: bool,
        ) -> tuple[int, pandas.DataFrame]:
            return 0, pandas.DataFrame(
                [
                    {
                        "code": "US.BE",
                        "trd_side": "BUY",
                        "order_type": "MARKET",
                        "order_status": "CANCELLED_ALL",
                        "order_id": "cancelled_be_order",
                        "qty": 22,
                    }
                ]
            )

    trade_context = CancelledOrderTradeContext()

    pre_open_entry_scheduler.reconcile_pre_open_entry_orders(
        trade_context=trade_context,
        quote_context=FakeQuoteContext(),
        trading_environment="REAL",
        market_order_type="MARKET",
        buy_side="BUY",
        normal_modify_operation="NORMAL",
        regular_session="RTH",
        quote_subtype="QUOTE",
        current_time=datetime(
            2026,
            7,
            21,
            9,
            28,
            tzinfo=pre_open_entry_scheduler.NEW_YORK_TIME_ZONE,
        ),
        state_path=state_path,
        config_path=config_path,
    )

    assert trade_context.maximum_quantity_queries == []
    assert trade_context.modified_orders == []
    assert trade_context.placed_orders == []
    persisted_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted_state["intents"][0]["status"] == "cancelled"
