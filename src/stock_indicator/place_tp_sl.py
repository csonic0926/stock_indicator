"""Place per-position TP/SL orders from Futu live source of truth.

Source of truth:
- Futu API for current positions, open orders, historical deals, and order remarks.
- Local JSON files are not used for live TP/SL decisions.

Contract:
- Each live position must have a Futu BUY order remark with v2 metadata
  (`si2|...|tp=...|sl=...`). Positions without that metadata are marked
  [ORPHAN_POSITION] and skipped.
- TP price = Futu position cost_price * (1 + remark.tp_pct), GTC limit sell.
- SL price = Futu position cost_price * (1 - remark.sl_pct), GTC stop, only
  after remark.min_hold_sl trading bars unless remark.disable_sl_trigger is true.

Usage:
    venv/bin/python -m stock_indicator.place_tp_sl [--dry-run]
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas

from stock_indicator.futu_trade_metadata import parse_futu_order_remark

LOGGER = logging.getLogger(__name__)

LOGS_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "logs"

TRADING_ENV = "REAL"
DEFAULT_HISTORY_LOOKBACK_DAYS = 180
DEFAULT_SL_MIN_HOLD_BARS = 1
TERMINAL_ORDER_STATUSES = {"CANCELLED_ALL", "FILLED_ALL", "FAILED", "DELETED"}


def _log_order(order_data: dict[str, Any]) -> None:
    """Append a TP/SL order result to today's local execution log."""
    today = date.today().isoformat()
    log_path = LOGS_DIRECTORY / f"{today}_orders.json"
    orders: list[dict[str, Any]] = []
    if log_path.exists():
        try:
            loaded_orders = json.loads(log_path.read_text(encoding="utf-8"))
            if isinstance(loaded_orders, list):
                orders = loaded_orders
        except (json.JSONDecodeError, OSError):
            pass
    orders.append(order_data)
    log_path.write_text(json.dumps(orders, indent=2), encoding="utf-8")


def _entry_disables_stop_loss_trigger(entry: dict[str, Any]) -> bool:
    """Return whether live SL placement is disabled for this Futu entry."""
    return bool(entry.get("disable_sl_trigger", False))


def _row_value(row: Any, candidate_names: list[str]) -> Any:
    """Return the first matching value from a Futu pandas row."""
    row_index_by_lower_name = {str(key).lower(): key for key in row.index}
    for candidate_name in candidate_names:
        matching_key = row_index_by_lower_name.get(candidate_name.lower())
        if matching_key is None:
            continue
        value = row.get(matching_key)
        if value is not None and not pandas.isna(value):
            return value
    return None


def _extract_deal_date(row: Any) -> str | None:
    """Extract a YYYY-MM-DD date from a Futu deal row."""
    raw_time = _row_value(
        row,
        ["create_time", "updated_time", "dealt_time", "deal_time", "time"],
    )
    if raw_time is None:
        return None
    try:
        return pandas.Timestamp(str(raw_time)).date().isoformat()
    except ValueError:
        return None


def _count_weekday_bars_held(entry_date_text: str, today_text: str) -> int:
    """Count weekday bars held using the same convention as dashboard max-hold."""
    entry_date = date.fromisoformat(entry_date_text)
    today_date = date.fromisoformat(today_text)
    if today_date < entry_date:
        return 0

    bars_held = 0
    current_date = entry_date
    while current_date <= today_date:
        if current_date.weekday() < 5:
            bars_held += 1
        current_date += timedelta(days=1)
    return bars_held


def _load_futu_order_remarks(
    trade_context: Any,
    trading_environment: Any,
    *,
    start_date_text: str,
    end_date_text: str,
) -> dict[str, str]:
    """Load Futu order remarks keyed by order id for the history window."""
    if not hasattr(trade_context, "history_order_list_query"):
        return {}
    return_code, order_data = trade_context.history_order_list_query(
        start=start_date_text,
        end=end_date_text,
        trd_env=trading_environment,
    )
    if return_code != 0 or order_data is None or len(order_data) == 0:
        return {}

    remarks_by_order_id: dict[str, str] = {}
    for _, order_row in order_data.iterrows():
        order_identifier = _row_value(order_row, ["order_id"])
        remark_text = _row_value(order_row, ["remark", "order_remark"])
        if order_identifier is not None and remark_text:
            remarks_by_order_id[str(order_identifier)] = str(remark_text)
    return remarks_by_order_id


def _load_futu_open_trade_entries(
    trade_context: Any,
    trading_environment: Any,
    *,
    as_of_date_text: str,
) -> dict[str, dict[str, Any]]:
    """Infer live open entries from Futu historical deals and order remarks."""
    as_of_date = date.fromisoformat(as_of_date_text)
    start_date_text = (
        as_of_date - timedelta(days=DEFAULT_HISTORY_LOOKBACK_DAYS)
    ).isoformat()

    if not hasattr(trade_context, "history_deal_list_query"):
        return {}
    return_code, deal_data = trade_context.history_deal_list_query(
        start=start_date_text,
        end=as_of_date_text,
        trd_env=trading_environment,
    )
    if return_code != 0 or deal_data is None or len(deal_data) == 0:
        return {}

    remarks_by_order_id = _load_futu_order_remarks(
        trade_context,
        trading_environment,
        start_date_text=start_date_text,
        end_date_text=as_of_date_text,
    )
    open_lots_by_symbol: dict[str, list[dict[str, Any]]] = {}

    for _, deal_row in deal_data.iterrows():
        code_text = str(_row_value(deal_row, ["code"]) or "")
        symbol = code_text.replace("US.", "")
        if not symbol:
            continue
        side_text = str(_row_value(deal_row, ["trd_side", "side"]) or "").upper()
        quantity_value = _row_value(deal_row, ["qty", "quantity"])
        try:
            quantity = abs(float(quantity_value))
        except (TypeError, ValueError):
            continue
        if quantity <= 0:
            continue

        if "BUY" in side_text:
            order_identifier = _row_value(deal_row, ["order_id"])
            remark_text = str(_row_value(deal_row, ["remark", "order_remark"]) or "")
            if not remark_text and order_identifier is not None:
                remark_text = remarks_by_order_id.get(str(order_identifier), "")
            metadata = parse_futu_order_remark(remark_text)
            entry_date_text = _extract_deal_date(deal_row)
            if entry_date_text is None:
                continue
            lot = {
                "symbol": symbol,
                "entry_date": entry_date_text,
                "remaining_quantity": quantity,
                **metadata,
            }
            open_lots_by_symbol.setdefault(symbol, []).append(lot)
        elif "SELL" in side_text:
            remaining_sell_quantity = quantity
            for open_lot in open_lots_by_symbol.get(symbol, []):
                if remaining_sell_quantity <= 0:
                    break
                consumed_quantity = min(
                    remaining_sell_quantity,
                    float(open_lot.get("remaining_quantity", 0)),
                )
                open_lot["remaining_quantity"] = (
                    float(open_lot.get("remaining_quantity", 0)) - consumed_quantity
                )
                remaining_sell_quantity -= consumed_quantity

    open_entries_by_symbol: dict[str, dict[str, Any]] = {}
    for symbol, open_lots in open_lots_by_symbol.items():
        remaining_lots = [
            lot for lot in open_lots
            if float(lot.get("remaining_quantity", 0)) > 0
        ]
        if remaining_lots:
            open_entries_by_symbol[symbol] = remaining_lots[0]
    return open_entries_by_symbol


def _load_futu_positions(position_data: Any) -> dict[str, dict[str, Any]]:
    """Convert Futu position rows to a code-keyed live position mapping."""
    positions: dict[str, dict[str, Any]] = {}
    if position_data is None or len(position_data) == 0:
        return positions
    for _, row in position_data.iterrows():
        try:
            quantity = int(row.get("qty", 0))
        except (TypeError, ValueError):
            continue
        if quantity <= 0:
            continue
        code = str(row.get("code", ""))
        positions[code] = {
            "qty": quantity,
            "cost_price": float(row.get("cost_price", 0)),
        }
    return positions


def _load_existing_sell_order_codes(order_data: Any) -> tuple[set[str], set[str]]:
    """Return current symbols with live TP and SL orders already open."""
    existing_take_profit_codes: set[str] = set()
    existing_stop_loss_codes: set[str] = set()
    if order_data is None or len(order_data) == 0:
        return existing_take_profit_codes, existing_stop_loss_codes

    for _, row in order_data.iterrows():
        if "SELL" not in str(row.get("trd_side", "")).upper():
            continue
        order_status = str(row.get("order_status", "")).upper()
        if order_status in TERMINAL_ORDER_STATUSES:
            continue
        code = str(row.get("code", ""))
        order_type_value = str(row.get("order_type", "")).upper()
        if "STOP" in order_type_value:
            existing_stop_loss_codes.add(code)
        elif "NORMAL" in order_type_value:
            existing_take_profit_codes.add(code)
    return existing_take_profit_codes, existing_stop_loss_codes


def _entry_supports_take_profit_stop_loss(entry: dict[str, Any]) -> bool:
    """Return whether a Futu remark has enough metadata for TP/SL orders."""
    return bool(entry.get("supports_tp_sl")) and "tp_pct" in entry and "sl_pct" in entry


def main() -> None:
    """Place missing TP/SL orders for live Futu positions."""
    from futu import (
        OpenSecTradeContext,
        OrderType,
        SecurityFirm,
        TimeInForce,
        TrdEnv,
        TrdMarket,
        TrdSide,
    )

    dry_run = "--dry-run" in sys.argv
    trading_environment = TrdEnv.REAL if TRADING_ENV == "REAL" else TrdEnv.SIMULATE

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    trade_context = OpenSecTradeContext(
        host="127.0.0.1",
        port=11111,
        filter_trdmarket=TrdMarket.US,
        security_firm=SecurityFirm.FUTUSECURITIES,
    )

    try:
        return_code, position_data = trade_context.position_list_query(
            trd_env=trading_environment
        )
        if return_code != 0:
            LOGGER.error("Failed to query positions: %s", position_data)
            return

        positions = _load_futu_positions(position_data)
        if not positions:
            LOGGER.info("No open positions")
            return

        LOGGER.info(
            "Positions: %s",
            ", ".join(
                f"{code.replace('US.', '')} qty={position['qty']} "
                f"cost=${position['cost_price']:.2f}"
                for code, position in positions.items()
            ),
        )

        return_code, order_data = trade_context.order_list_query(
            trd_env=trading_environment
        )
        if return_code != 0:
            LOGGER.error("Failed to query open orders: %s", order_data)
            return
        existing_take_profit_codes, existing_stop_loss_codes = (
            _load_existing_sell_order_codes(order_data)
        )

        today_text = date.today().isoformat()
        open_entries_by_symbol = _load_futu_open_trade_entries(
            trade_context,
            trading_environment,
            as_of_date_text=today_text,
        )

        LOGGER.info("--- TP check ---")
        for code, position in positions.items():
            symbol = code.replace("US.", "")
            entry = open_entries_by_symbol.get(symbol)
            if entry is None or not _entry_supports_take_profit_stop_loss(entry):
                LOGGER.warning(
                    "[ORPHAN_POSITION] code=%s qty=%d cost=$%.2f — "
                    "no Futu v2 BUY remark metadata, skipping TP/SL",
                    code,
                    position["qty"],
                    position["cost_price"],
                )
                continue

            if code in existing_take_profit_codes:
                LOGGER.info("%s: TP already exists, skip", symbol)
                continue

            take_profit_pct = float(entry.get("tp_pct", 0))
            if take_profit_pct <= 0:
                LOGGER.info("%s: tp_pct <= 0 (entry=%s), skip TP", symbol, take_profit_pct)
                continue

            take_profit_price = round(position["cost_price"] * (1 + take_profit_pct), 2)
            bucket = entry.get("bucket", "?")
            LOGGER.info(
                "%s [%s]: cost=$%.2f qty=%d -> TP=$%.2f (+%.2f%%) [GTC limit sell]",
                symbol,
                bucket,
                position["cost_price"],
                position["qty"],
                take_profit_price,
                take_profit_pct * 100,
            )

            if dry_run:
                LOGGER.info("  [DRY RUN] skipping")
                continue

            return_code, take_profit_data = trade_context.place_order(
                price=take_profit_price,
                qty=position["qty"],
                code=code,
                trd_side=TrdSide.SELL,
                order_type=OrderType.NORMAL,
                trd_env=trading_environment,
                time_in_force=TimeInForce.GTC,
            )
            take_profit_result = {
                "symbol": symbol,
                "bucket": bucket,
                "side": "TP_SELL",
                "qty": position["qty"],
                "entry_price": position["cost_price"],
                "price": take_profit_price,
                "tp_pct": round(take_profit_pct * 100, 4),
                "status": "sent" if return_code == 0 else "failed",
                "order_id": (
                    str(take_profit_data.iloc[0].get("order_id", ""))
                    if return_code == 0 else None
                ),
                "error": str(take_profit_data) if return_code != 0 else None,
                "env": TRADING_ENV,
                "timestamp": datetime.now().isoformat(),
            }
            LOGGER.info("  TP: %s", take_profit_result["status"])
            _log_order(take_profit_result)

        LOGGER.info("--- SL check ---")
        for code, position in positions.items():
            symbol = code.replace("US.", "")
            entry = open_entries_by_symbol.get(symbol)
            if entry is None or not _entry_supports_take_profit_stop_loss(entry):
                continue

            if _entry_disables_stop_loss_trigger(entry):
                LOGGER.info("%s: disable_sl_trigger=True, skip SL placement", symbol)
                continue

            if code in existing_stop_loss_codes:
                LOGGER.info("%s: SL already exists, skip", symbol)
                continue

            stop_loss_pct = float(entry.get("sl_pct", 0))
            if stop_loss_pct <= 0:
                LOGGER.info("%s: sl_pct <= 0 (entry=%s), skip SL", symbol, stop_loss_pct)
                continue

            try:
                bars_held = _count_weekday_bars_held(
                    str(entry.get("entry_date", "")),
                    today_text,
                )
            except ValueError:
                LOGGER.info(
                    "%s: invalid Futu entry_date=%s, skip SL",
                    symbol,
                    entry.get("entry_date"),
                )
                continue

            stop_loss_min_hold = int(
                entry.get("min_hold_sl", DEFAULT_SL_MIN_HOLD_BARS)
                or DEFAULT_SL_MIN_HOLD_BARS
            )
            if bars_held < stop_loss_min_hold:
                LOGGER.info(
                    "%s: bars_held=%d < min_hold_sl=%d, SL deferred",
                    symbol,
                    bars_held,
                    stop_loss_min_hold,
                )
                continue

            stop_loss_price = round(position["cost_price"] * (1 - stop_loss_pct), 2)
            bucket = entry.get("bucket", "?")
            LOGGER.info(
                "%s [%s]: bars_held=%d -> SL=$%.2f (-%.2f%%) [GTC stop]",
                symbol,
                bucket,
                bars_held,
                stop_loss_price,
                stop_loss_pct * 100,
            )

            if dry_run:
                LOGGER.info("  [DRY RUN] skipping")
                continue

            return_code, stop_loss_data = trade_context.place_order(
                price=stop_loss_price,
                qty=position["qty"],
                code=code,
                trd_side=TrdSide.SELL,
                order_type=OrderType.STOP,
                trd_env=trading_environment,
                aux_price=stop_loss_price,
                time_in_force=TimeInForce.GTC,
            )
            stop_loss_result = {
                "symbol": symbol,
                "bucket": bucket,
                "side": "SL_SELL",
                "qty": position["qty"],
                "entry_price": position["cost_price"],
                "price": stop_loss_price,
                "sl_pct": round(stop_loss_pct * 100, 4),
                "bars_held": bars_held,
                "status": "sent" if return_code == 0 else "failed",
                "order_id": (
                    str(stop_loss_data.iloc[0].get("order_id", ""))
                    if return_code == 0 else None
                ),
                "error": str(stop_loss_data) if return_code != 0 else None,
                "env": TRADING_ENV,
                "timestamp": datetime.now().isoformat(),
            }
            LOGGER.info("  SL: %s", stop_loss_result["status"])
            _log_order(stop_loss_result)
    finally:
        trade_context.close()

    LOGGER.info("Done")


if __name__ == "__main__":
    main()
