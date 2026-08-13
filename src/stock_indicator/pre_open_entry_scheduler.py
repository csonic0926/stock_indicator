"""Adjust confirmed US market-entry quantities shortly before the open.

The dashboard records every confirmed real BUY intent before calling Futu.
Between 09:27 and 09:30 New York time this scheduler uses the live pre-market
price and Futu's market-order buying-power calculation to resize an accepted
order or retry a previously rejected intent.
"""

from __future__ import annotations

# TODO: review

import json
import logging
import math
import os
from datetime import datetime, time
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from stock_indicator.futu_trade_metadata import format_futu_order_remark
from stock_indicator.live_order_sizing import compute_target_share_quantity

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LIVE_STATE_DIRECTORY = PROJECT_ROOT / "data" / "live_state"
ENTRY_INTENT_STATE_PATH = LIVE_STATE_DIRECTORY / "pending_entry_intents.json"
SCHEDULER_STATE_PATH = (
    LIVE_STATE_DIRECTORY / "pre_open_entry_scheduler_state.json"
)
PRODUCTION_CONFIG_PATH = PROJECT_ROOT / "data" / "multi_bucket_production.json"
NEW_YORK_TIME_ZONE = ZoneInfo("America/New_York")
RECONCILIATION_START_TIME = time(9, 27)
REGULAR_SESSION_OPEN_TIME = time(9, 30)
TRADING_ENVIRONMENT = os.environ.get(
    "STOCK_INDICATOR_TRADING_ENV",
    "SIMULATE",
).strip().upper()

TERMINAL_ORDER_STATUSES = {
    "CANCELLED_PART",
    "CANCELLED_ALL",
    "DELETED",
    "DISABLED",
    "FAILED",
    "FILLED_ALL",
    "FILL_CANCELLED",
    "SUBMIT_FAILED",
    "TIMEOUT",
}
CANCELLED_ORDER_STATUSES = {
    "CANCELLED_PART",
    "CANCELLED_ALL",
    "DELETED",
    "DISABLED",
    "FILL_CANCELLED",
}


def _load_json_document(path: Path) -> dict[str, Any]:
    """Load a JSON object, treating missing or corrupt state as empty."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return document if isinstance(document, dict) else {}


def _save_json_document(path: Path, document: dict[str, Any]) -> None:
    """Atomically persist a JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(document, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _new_york_market_date(current_time: datetime | None = None) -> str:
    """Return the current New York market date."""
    evaluation_time = current_time or datetime.now(tz=NEW_YORK_TIME_ZONE)
    if evaluation_time.tzinfo is None:
        raise ValueError("current_time must be timezone-aware")
    return evaluation_time.astimezone(NEW_YORK_TIME_ZONE).date().isoformat()


def record_confirmed_entry_intent(
    order: dict[str, Any],
    *,
    state_path: Path = ENTRY_INTENT_STATE_PATH,
    current_time: datetime | None = None,
) -> dict[str, Any]:
    """Persist one user-confirmed real market BUY before broker submission."""
    market_date_text = _new_york_market_date(current_time)
    symbol = str(order.get("symbol") or "").strip().upper()
    requested_quantity = int(order.get("qty", 0))
    if not symbol or requested_quantity <= 0:
        raise ValueError("entry intent requires a symbol and positive quantity")

    intent = {
        "market_date": market_date_text,
        "symbol": symbol,
        "requested_qty": requested_quantity,
        "remark": format_futu_order_remark(order),
        "bucket": order.get("bucket"),
        "strategy_id": order.get("strategy_id"),
        "status": "confirmed",
        "updated_at": datetime.now(tz=NEW_YORK_TIME_ZONE).isoformat(),
    }
    state_document = _load_json_document(state_path)
    existing_intents = state_document.get("intents", [])
    if not isinstance(existing_intents, list):
        existing_intents = []
    retained_intents = [
        existing_intent
        for existing_intent in existing_intents
        if not (
            isinstance(existing_intent, dict)
            and existing_intent.get("market_date") == market_date_text
            and existing_intent.get("symbol") == symbol
        )
    ]
    retained_intents.append(intent)
    _save_json_document(state_path, {"intents": retained_intents[-100:]})
    return intent


def record_entry_submission_result(
    *,
    market_date_text: str,
    symbol: str,
    status: str,
    state_path: Path = ENTRY_INTENT_STATE_PATH,
    order_id: str | None = None,
    error: str | None = None,
) -> None:
    """Update the broker-submission result for one persisted entry intent."""
    state_document = _load_json_document(state_path)
    intents = state_document.get("intents", [])
    if not isinstance(intents, list):
        return
    changed = False
    for intent in intents:
        if not isinstance(intent, dict):
            continue
        if (
            intent.get("market_date") != market_date_text
            or intent.get("symbol") != symbol
        ):
            continue
        intent["status"] = status
        intent["order_id"] = order_id
        intent["error"] = error
        intent["updated_at"] = datetime.now(
            tz=NEW_YORK_TIME_ZONE
        ).isoformat()
        changed = True
        break
    if changed:
        _save_json_document(state_path, {"intents": intents})


def _load_maximum_position_count(config_path: Path) -> int:
    """Load the production portfolio slot count."""
    config_document = json.loads(config_path.read_text(encoding="utf-8"))
    maximum_position_count = int(config_document["max_position_count"])
    if maximum_position_count <= 0:
        raise ValueError("max_position_count must be positive")
    return maximum_position_count


def _normalize_broker_symbol(code_value: Any) -> str:
    """Return an uppercase ticker without the Futu market prefix."""
    normalized_code = str(code_value or "").strip().upper()
    if normalized_code.startswith("US."):
        return normalized_code[3:]
    return normalized_code


def _active_market_buy_orders(order_data: Any) -> dict[str, Any]:
    """Return non-terminal market BUY rows keyed by normalized symbol."""
    active_orders: dict[str, Any] = {}
    if order_data is None or len(order_data) == 0:
        return active_orders
    for _order_index, order_row in order_data.iterrows():
        if "BUY" not in str(order_row.get("trd_side", "")).upper():
            continue
        if str(order_row.get("order_type", "")).upper() != "MARKET":
            continue
        order_status = str(order_row.get("order_status", "")).upper()
        if order_status in TERMINAL_ORDER_STATUSES:
            continue
        symbol = _normalize_broker_symbol(order_row.get("code", ""))
        if symbol:
            active_orders[symbol] = order_row
    return active_orders


def _market_buy_orders_by_id(order_data: Any) -> dict[str, Any]:
    """Return all visible market BUY rows keyed by Futu order identifier."""
    orders_by_id: dict[str, Any] = {}
    if order_data is None or len(order_data) == 0:
        return orders_by_id
    for _order_index, order_row in order_data.iterrows():
        if "BUY" not in str(order_row.get("trd_side", "")).upper():
            continue
        if str(order_row.get("order_type", "")).upper() != "MARKET":
            continue
        order_id = str(order_row.get("order_id", ""))
        if order_id:
            orders_by_id[order_id] = order_row
    return orders_by_id


def _positive_position_symbols(position_data: Any) -> set[str]:
    """Return symbols with positive live Futu position quantities."""
    positive_symbols: set[str] = set()
    if position_data is None or len(position_data) == 0:
        return positive_symbols
    for _position_index, position_row in position_data.iterrows():
        try:
            position_quantity = float(position_row.get("qty", 0))
        except (TypeError, ValueError):
            continue
        if position_quantity <= 0:
            continue
        symbol = _normalize_broker_symbol(position_row.get("code", ""))
        if symbol:
            positive_symbols.add(symbol)
    return positive_symbols


def _load_pre_market_prices(
    *,
    quote_context: Any,
    broker_codes: list[str],
    quote_subtype: Any,
) -> dict[str, tuple[float, str]]:
    """Return live pre-market prices, with safe fallbacks when no trade exists."""
    subscribe_return_code, subscribe_data = quote_context.subscribe(
        broker_codes,
        [quote_subtype],
        subscribe_push=False,
    )
    if subscribe_return_code != 0:
        raise RuntimeError(f"Failed to subscribe to Futu quotes: {subscribe_data}")

    quote_return_code, quote_data = quote_context.get_stock_quote(broker_codes)
    if quote_return_code != 0:
        raise RuntimeError(f"Failed to query Futu real-time quotes: {quote_data}")
    snapshot_return_code, snapshot_data = quote_context.get_market_snapshot(
        broker_codes
    )
    if snapshot_return_code != 0:
        raise RuntimeError(f"Failed to query Futu snapshots: {snapshot_data}")

    quote_rows_by_code = {
        str(quote_row.get("code", "")): quote_row
        for _quote_index, quote_row in quote_data.iterrows()
    }
    prices_by_symbol: dict[str, tuple[float, str]] = {}
    for _snapshot_index, snapshot_row in snapshot_data.iterrows():
        broker_code = str(snapshot_row.get("code", ""))
        quote_row = quote_rows_by_code.get(broker_code)
        price_candidates = [
            (
                quote_row.get("pre_price")
                if quote_row is not None else None,
                "pre_price",
            ),
            (snapshot_row.get("pre_price"), "pre_price"),
            (snapshot_row.get("ask_price"), "ask_price"),
            (snapshot_row.get("last_price"), "last_price"),
        ]
        for raw_price, price_source in price_candidates:
            try:
                price = float(raw_price)
            except (TypeError, ValueError):
                continue
            if math.isfinite(price) and price > 0:
                prices_by_symbol[_normalize_broker_symbol(broker_code)] = (
                    price,
                    price_source,
                )
                break
    return prices_by_symbol


def _update_intent_after_reconciliation(
    intent: dict[str, Any],
    *,
    status: str,
    reference_price: float | None = None,
    reference_price_source: str | None = None,
    adjusted_quantity: int | None = None,
    order_id: str | None = None,
    error: str | None = None,
) -> None:
    """Mutate one in-memory intent with its latest reconciliation result."""
    intent["status"] = status
    intent["reference_price"] = reference_price
    intent["reference_price_source"] = reference_price_source
    intent["adjusted_qty"] = adjusted_quantity
    if order_id is not None:
        intent["order_id"] = order_id
    intent["error"] = error
    intent["updated_at"] = datetime.now(tz=NEW_YORK_TIME_ZONE).isoformat()


def reconcile_pre_open_entry_orders(
    *,
    trade_context: Any,
    quote_context: Any,
    trading_environment: Any,
    market_order_type: Any,
    buy_side: Any,
    normal_modify_operation: Any,
    regular_session: Any,
    quote_subtype: Any,
    current_time: datetime,
    state_path: Path = ENTRY_INTENT_STATE_PATH,
    config_path: Path = PRODUCTION_CONFIG_PATH,
) -> None:
    """Resize or retry today's confirmed entry orders before the US open."""
    market_date_text = _new_york_market_date(current_time)
    state_document = _load_json_document(state_path)
    all_intents = state_document.get("intents", [])
    if not isinstance(all_intents, list):
        all_intents = []
    todays_intents = [
        intent
        for intent in all_intents
        if isinstance(intent, dict)
        and intent.get("market_date") == market_date_text
    ]
    if not todays_intents:
        LOGGER.info("No confirmed entry intents for %s", market_date_text)
        return

    account_return_code, account_data = trade_context.accinfo_query(
        trd_env=trading_environment
    )
    if account_return_code != 0 or len(account_data) == 0:
        raise RuntimeError(f"Failed to query Futu account: {account_data}")
    total_assets_hkd = float(account_data.iloc[0].get("total_assets", 0))
    maximum_position_count = _load_maximum_position_count(config_path)

    order_return_code, order_data = trade_context.order_list_query(
        trd_env=trading_environment,
        refresh_cache=True,
    )
    if order_return_code != 0:
        raise RuntimeError(f"Failed to query Futu orders: {order_data}")
    active_orders_by_symbol = _active_market_buy_orders(order_data)
    market_buy_orders_by_id = _market_buy_orders_by_id(order_data)

    position_return_code, position_data = trade_context.position_list_query(
        trd_env=trading_environment
    )
    if position_return_code != 0:
        raise RuntimeError(f"Failed to query Futu positions: {position_data}")
    held_symbols = _positive_position_symbols(position_data)

    broker_codes = [
        f"US.{str(intent['symbol']).upper()}" for intent in todays_intents
    ]
    prices_by_symbol = _load_pre_market_prices(
        quote_context=quote_context,
        broker_codes=broker_codes,
        quote_subtype=quote_subtype,
    )

    reconciliation_errors: list[str] = []
    for intent in todays_intents:
        symbol = str(intent.get("symbol") or "").upper()
        if symbol in held_symbols:
            _update_intent_after_reconciliation(
                intent,
                status="already_filled",
            )
            continue

        price_record = prices_by_symbol.get(symbol)
        if price_record is None:
            error_message = f"{symbol}: no usable pre-market quote"
            _update_intent_after_reconciliation(
                intent,
                status="quote_failed",
                error=error_message,
            )
            reconciliation_errors.append(error_message)
            continue
        reference_price, reference_price_source = price_record
        strategy_target_quantity = compute_target_share_quantity(
            total_assets_hkd=total_assets_hkd,
            price_usd=reference_price,
            maximum_position_count=maximum_position_count,
        )

        active_order = active_orders_by_symbol.get(symbol)
        recorded_order_id = str(intent.get("order_id") or "")
        recorded_order = market_buy_orders_by_id.get(recorded_order_id)
        recorded_order_status = (
            str(recorded_order.get("order_status", "")).upper()
            if recorded_order is not None else ""
        )
        if active_order is None and recorded_order_status in CANCELLED_ORDER_STATUSES:
            _update_intent_after_reconciliation(
                intent,
                status="cancelled",
                reference_price=reference_price,
                reference_price_source=reference_price_source,
                order_id=recorded_order_id,
            )
            continue
        if active_order is None and recorded_order_status == "FILLED_ALL":
            _update_intent_after_reconciliation(
                intent,
                status="already_filled",
                reference_price=reference_price,
                reference_price_source=reference_price_source,
                order_id=recorded_order_id,
            )
            continue
        active_order_id = (
            str(active_order.get("order_id", "")) if active_order is not None else ""
        )
        maximum_quantity_arguments: dict[str, Any] = {
            "order_type": market_order_type,
            "code": f"US.{symbol}",
            "price": reference_price,
            "trd_env": trading_environment,
            "session": regular_session,
        }
        if active_order_id:
            maximum_quantity_arguments["order_id"] = active_order_id
        maximum_return_code, maximum_data = (
            trade_context.acctradinginfo_query(**maximum_quantity_arguments)
        )
        if maximum_return_code != 0 or len(maximum_data) == 0:
            error_message = f"{symbol}: max-buy query failed: {maximum_data}"
            _update_intent_after_reconciliation(
                intent,
                status="max_buy_query_failed",
                reference_price=reference_price,
                reference_price_source=reference_price_source,
                error=error_message,
            )
            reconciliation_errors.append(error_message)
            continue

        maximum_broker_quantity = max(
            0,
            math.floor(
                float(maximum_data.iloc[0].get("max_cash_and_margin_buy", 0))
            ),
        )
        adjusted_quantity = min(
            strategy_target_quantity,
            maximum_broker_quantity,
        )
        if adjusted_quantity <= 0:
            error_message = f"{symbol}: Futu returned zero market-buy capacity"
            _update_intent_after_reconciliation(
                intent,
                status="no_buying_power",
                reference_price=reference_price,
                reference_price_source=reference_price_source,
                adjusted_quantity=0,
                order_id=active_order_id or None,
                error=error_message,
            )
            reconciliation_errors.append(error_message)
            continue

        if active_order is not None:
            current_order_quantity = int(float(active_order.get("qty", 0)))
            if current_order_quantity == adjusted_quantity:
                _update_intent_after_reconciliation(
                    intent,
                    status="quantity_confirmed",
                    reference_price=reference_price,
                    reference_price_source=reference_price_source,
                    adjusted_quantity=adjusted_quantity,
                    order_id=active_order_id,
                )
                continue
            modify_return_code, modify_data = trade_context.modify_order(
                modify_order_op=normal_modify_operation,
                order_id=active_order_id,
                qty=adjusted_quantity,
                price=reference_price,
                trd_env=trading_environment,
            )
            if modify_return_code != 0:
                error_message = f"{symbol}: quantity adjustment failed: {modify_data}"
                _update_intent_after_reconciliation(
                    intent,
                    status="modify_failed",
                    reference_price=reference_price,
                    reference_price_source=reference_price_source,
                    adjusted_quantity=adjusted_quantity,
                    order_id=active_order_id,
                    error=error_message,
                )
                reconciliation_errors.append(error_message)
                continue
            _update_intent_after_reconciliation(
                intent,
                status="quantity_adjusted",
                reference_price=reference_price,
                reference_price_source=reference_price_source,
                adjusted_quantity=adjusted_quantity,
                order_id=active_order_id,
            )
            continue

        place_return_code, place_data = trade_context.place_order(
            price=reference_price,
            qty=adjusted_quantity,
            code=f"US.{symbol}",
            trd_side=buy_side,
            order_type=market_order_type,
            trd_env=trading_environment,
            remark=str(intent.get("remark") or ""),
            session=regular_session,
        )
        if place_return_code != 0:
            error_message = f"{symbol}: pre-open retry failed: {place_data}"
            _update_intent_after_reconciliation(
                intent,
                status="retry_failed",
                reference_price=reference_price,
                reference_price_source=reference_price_source,
                adjusted_quantity=adjusted_quantity,
                error=error_message,
            )
            reconciliation_errors.append(error_message)
            continue
        new_order_id = str(place_data.iloc[0].get("order_id", ""))
        _update_intent_after_reconciliation(
            intent,
            status="retry_submitted",
            reference_price=reference_price,
            reference_price_source=reference_price_source,
            adjusted_quantity=adjusted_quantity,
            order_id=new_order_id,
        )

    _save_json_document(state_path, {"intents": all_intents})
    if reconciliation_errors:
        raise RuntimeError("; ".join(reconciliation_errors))


def _run_live_reconciliation(current_time: datetime) -> None:
    """Create Futu contexts and reconcile real or simulated entry orders."""
    from futu import (
        ModifyOrderOp,
        OpenQuoteContext,
        OpenSecTradeContext,
        OrderType,
        SecurityFirm,
        Session,
        SubType,
        TrdEnv,
        TrdMarket,
        TrdSide,
    )

    trading_environment = (
        TrdEnv.REAL if TRADING_ENVIRONMENT == "REAL" else TrdEnv.SIMULATE
    )
    quote_context = OpenQuoteContext(host="127.0.0.1", port=11111)
    trade_context = OpenSecTradeContext(
        host="127.0.0.1",
        port=11111,
        filter_trdmarket=TrdMarket.US,
        security_firm=SecurityFirm.FUTUSECURITIES,
    )
    try:
        reconcile_pre_open_entry_orders(
            trade_context=trade_context,
            quote_context=quote_context,
            trading_environment=trading_environment,
            market_order_type=OrderType.MARKET,
            buy_side=TrdSide.BUY,
            normal_modify_operation=ModifyOrderOp.NORMAL,
            regular_session=Session.RTH,
            quote_subtype=SubType.QUOTE,
            current_time=current_time,
        )
    finally:
        quote_context.close()
        trade_context.close()


def run_due_pre_open_entry_reconciliation(
    *,
    current_time: datetime | None = None,
    scheduler_state_path: Path = SCHEDULER_STATE_PATH,
    reconciliation_function: Callable[[datetime], None] | None = None,
) -> bool:
    """Run once from 09:27 through 09:29 New York time, retrying failures."""
    evaluation_time = current_time or datetime.now(tz=NEW_YORK_TIME_ZONE)
    if evaluation_time.tzinfo is None:
        raise ValueError("current_time must be timezone-aware")
    new_york_time = evaluation_time.astimezone(NEW_YORK_TIME_ZONE)
    if new_york_time.weekday() >= 5:
        return False
    local_clock_time = new_york_time.time().replace(tzinfo=None)
    if not (
        RECONCILIATION_START_TIME
        <= local_clock_time
        < REGULAR_SESSION_OPEN_TIME
    ):
        return False

    checkpoint_key = f"{new_york_time.date().isoformat()}:pre_open_entry"
    scheduler_state = _load_json_document(scheduler_state_path)
    if scheduler_state.get("last_completed_checkpoint") == checkpoint_key:
        return False

    LOGGER.info(
        "Running pre-open entry reconciliation %s at %s",
        checkpoint_key,
        new_york_time.isoformat(),
    )
    if reconciliation_function is None:
        _run_live_reconciliation(evaluation_time)
    else:
        reconciliation_function(evaluation_time)
    _save_json_document(
        scheduler_state_path,
        {
            "last_completed_checkpoint": checkpoint_key,
            "completed_at_new_york": new_york_time.isoformat(),
        },
    )
    return True


def main() -> None:
    """Run one launchd scheduler tick."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        run_due_pre_open_entry_reconciliation()
    except Exception:
        LOGGER.exception("Pre-open entry reconciliation failed")
        raise


if __name__ == "__main__":
    main()
