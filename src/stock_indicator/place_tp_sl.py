"""Place per-position TP/SL orders from Futu live source of truth.

Source of truth:
- Futu API for current positions, open orders, and historical deals.
- Production signal logs identify which production bucket opened the API-confirmed
  lot; production config controls TP/SL behavior.

Contract:
- TP price = Futu position cost_price * (1 + production tp_pct), GTC limit sell.
- SL price = Futu position cost_price * (1 - production sl_pct), GTC stop, only
  after production min_hold_sl bars unless disable_sl_trigger is true.
- Futu BUY remarks are optional debug metadata, not a TP/SL prerequisite.

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
LOGS_DIRECTORY = PROJECT_ROOT / "logs"
PRODUCTION_CONFIG_PATH = DATA_DIRECTORY / "multi_bucket_production.json"

TRADING_ENV = "REAL"
DEFAULT_HISTORY_LOOKBACK_DAYS = 180
DEFAULT_SL_MIN_HOLD_BARS = 1
TERMINAL_ORDER_STATUSES = {"CANCELLED_ALL", "FILLED_ALL", "FAILED", "DELETED"}
BUCKET_TP_SL_WALK_BACK_DAYS = 14


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


def _sort_deals_for_fifo(deal_data: pandas.DataFrame) -> pandas.DataFrame:
    """Sort Futu deals oldest-first before reconstructing open lots.

    Futu history can return rows newest-first. FIFO lot reconstruction must
    process buys and sells chronologically, otherwise a later sell can consume a
    newer buy before an older buy and attach the wrong order metadata.
    """
    sortable_deal_data = deal_data.copy()
    sort_values: list[pandas.Timestamp] = []
    for _, deal_row in sortable_deal_data.iterrows():
        raw_time = _row_value(
            deal_row,
            ["create_time", "updated_time", "dealt_time", "deal_time", "time"],
        )
        sort_values.append(pandas.to_datetime(raw_time, errors="coerce"))
    sortable_deal_data["_fifo_sort_time"] = sort_values
    return sortable_deal_data.sort_values(
        by="_fifo_sort_time",
        kind="stable",
        na_position="last",
    )


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


def _load_futu_order_history(
    trade_context: Any,
    trading_environment: Any,
    *,
    start_date_text: str,
    end_date_text: str,
) -> dict[str, dict[str, Any]]:
    """Load Futu order history keyed by order id for the history window."""
    if not hasattr(trade_context, "history_order_list_query"):
        return {}
    return_code, order_data = trade_context.history_order_list_query(
        start=start_date_text,
        end=end_date_text,
        trd_env=trading_environment,
    )
    if return_code != 0 or order_data is None or len(order_data) == 0:
        return {}

    order_history_by_order_id: dict[str, dict[str, Any]] = {}
    for _, order_row in order_data.iterrows():
        order_identifier = _row_value(order_row, ["order_id"])
        remark_text = _row_value(order_row, ["remark", "order_remark"])
        order_date_text = _extract_deal_date(order_row)
        if order_identifier is not None:
            order_history_by_order_id[str(order_identifier)] = {
                "remark": str(remark_text or ""),
                "order_date": order_date_text,
            }
    return order_history_by_order_id


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

    order_history_by_order_id = _load_futu_order_history(
        trade_context,
        trading_environment,
        start_date_text=start_date_text,
        end_date_text=as_of_date_text,
    )
    open_lots_by_symbol: dict[str, list[dict[str, Any]]] = {}

    for _, deal_row in _sort_deals_for_fifo(deal_data).iterrows():
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
            order_history = (
                order_history_by_order_id.get(str(order_identifier), {})
                if order_identifier is not None else {}
            )
            if not remark_text and order_identifier is not None:
                remark_text = str(order_history.get("remark") or "")
            metadata = parse_futu_order_remark(remark_text)
            entry_date_text = _extract_deal_date(deal_row)
            if entry_date_text is None:
                continue
            lot = {
                "symbol": symbol,
                "entry_date": entry_date_text,
                "remaining_quantity": quantity,
                "futu_buy_order_id": str(order_identifier or ""),
                "futu_buy_remark_present": bool(remark_text),
                "futu_buy_order_date": order_history.get("order_date"),
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


def _parse_signal_log_tokens(token_text: str) -> dict[str, Any]:
    """Parse key=value tokens from production signal log lines."""
    parsed_values: dict[str, Any] = {}
    for raw_token in token_text.strip().split():
        key_text, separator, raw_value_text = raw_token.partition("=")
        if not separator:
            continue
        try:
            if raw_value_text.lower() in {"true", "false"}:
                parsed_values[key_text] = raw_value_text.lower() == "true"
            elif "." in raw_value_text or "e" in raw_value_text.lower():
                parsed_values[key_text] = float(raw_value_text)
            else:
                parsed_values[key_text] = int(raw_value_text)
        except ValueError:
            parsed_values[key_text] = raw_value_text
    return parsed_values


def _load_production_signal_entries() -> dict[tuple[str, str], dict[str, Any]]:
    """Load production [FROZEN_TP_SL] entries keyed by symbol and entry date.

    Futu confirms that a live lot exists and when it opened. The production log
    maps that confirmed lot back to the bucket/config that generated the signal.
    """
    entries_by_symbol_and_date: dict[tuple[str, str], dict[str, Any]] = {}
    for log_path in sorted(LOGS_DIRECTORY.glob("*.log")):
        try:
            log_text = log_path.read_text(encoding="utf-8")
        except OSError:
            continue
        for log_line in log_text.splitlines():
            if not log_line.startswith("[FROZEN_TP_SL]"):
                continue
            entry = _parse_signal_log_tokens(log_line[len("[FROZEN_TP_SL]"):])
            symbol = str(entry.get("symbol") or "").upper()
            entry_date_text = str(entry.get("entry_date") or "")
            if symbol and entry_date_text:
                entries_by_symbol_and_date[(symbol, entry_date_text)] = entry
    return entries_by_symbol_and_date


def _load_production_bucket_tp_sl_entries() -> dict[tuple[str, str], dict[str, Any]]:
    """Load daily [BUCKET_TP_SL] snapshots keyed by bucket or strategy and date."""
    entries_by_bucket_or_strategy_and_date: dict[tuple[str, str], dict[str, Any]] = {}
    for log_path in sorted(LOGS_DIRECTORY.glob("*.log")):
        try:
            log_text = log_path.read_text(encoding="utf-8")
        except OSError:
            continue
        for log_line in log_text.splitlines():
            if not log_line.startswith("[BUCKET_TP_SL]"):
                continue
            entry = _parse_signal_log_tokens(log_line[len("[BUCKET_TP_SL]"):])
            date_text = str(entry.get("date") or log_path.stem)
            bucket = str(entry.get("bucket") or "")
            strategy_identifier = str(entry.get("strategy_id") or "")
            if bucket:
                entries_by_bucket_or_strategy_and_date[(bucket, date_text)] = entry
            if strategy_identifier:
                entries_by_bucket_or_strategy_and_date[
                    (strategy_identifier, date_text)
                ] = entry
    return entries_by_bucket_or_strategy_and_date


def _load_production_exit_rules() -> dict[str, dict[str, Any]]:
    """Load production TP/SL management rules keyed by bucket and strategy id."""
    from stock_indicator import multi_bucket_today

    config = multi_bucket_today.load_multi_bucket_config(PRODUCTION_CONFIG_PATH)
    adaptive_config = config.adaptive_tp_sl
    default_min_hold_stop_loss = (
        adaptive_config.min_hold_sl if adaptive_config is not None else 1
    )
    default_disable_stop_loss_trigger = (
        adaptive_config.disable_sl_trigger if adaptive_config is not None else False
    )

    rules_by_key: dict[str, dict[str, Any]] = {}
    for bucket_label, bucket_definition in config.bucket_definitions.items():
        min_hold_stop_loss = (
            bucket_definition.min_hold_sl
            if bucket_definition.min_hold_sl is not None
            else default_min_hold_stop_loss
        )
        rule = {
            "bucket": bucket_label,
            "strategy_id": bucket_definition.strategy_identifier,
            "min_hold_sl": min_hold_stop_loss,
            "disable_sl_trigger": default_disable_stop_loss_trigger,
            "max_hold": bucket_definition.max_hold,
            "reset_hold_on_reentry_signal": (
                bucket_definition.reset_hold_on_reentry_signal
            ),
        }
        rules_by_key[bucket_label] = rule
        rules_by_key[bucket_definition.strategy_identifier] = rule
    return rules_by_key


def _walk_back_bucket_tp_sl_entry(
    *,
    bucket_candidate: str,
    anchor_date_text: str,
    bucket_tp_sl_entries_by_key_and_date: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    """Find the most recent BUCKET_TP_SL entry for `bucket_candidate` at or
    before `anchor_date_text`, within BUCKET_TP_SL_WALK_BACK_DAYS calendar days.

    Covers the case where cron freezes a bucket snapshot on a trading day but
    Futu fills the entry on a later non-trading-aligned day (e.g., holiday
    cron run vs. next-session execution).
    """
    if not bucket_candidate or not anchor_date_text:
        return None, None
    try:
        anchor_date = date.fromisoformat(anchor_date_text)
    except ValueError:
        return None, None
    best_entry: dict[str, Any] | None = None
    best_date_text: str | None = None
    best_distance = BUCKET_TP_SL_WALK_BACK_DAYS + 1
    for (bucket_key, candidate_date_text), candidate_entry in (
        bucket_tp_sl_entries_by_key_and_date.items()
    ):
        if bucket_key != bucket_candidate:
            continue
        try:
            candidate_date = date.fromisoformat(candidate_date_text)
        except ValueError:
            continue
        if candidate_date > anchor_date:
            continue
        distance_days = (anchor_date - candidate_date).days
        if distance_days > BUCKET_TP_SL_WALK_BACK_DAYS:
            continue
        if distance_days < best_distance:
            best_entry = candidate_entry
            best_date_text = candidate_date_text
            best_distance = distance_days
            if distance_days == 0:
                break
    return best_entry, best_date_text


def _merge_production_signal_metadata(
    *,
    symbol: str,
    entry: dict[str, Any] | None,
    signal_entries_by_symbol_and_date: dict[tuple[str, str], dict[str, Any]],
    bucket_tp_sl_entries_by_key_and_date: dict[tuple[str, str], dict[str, Any]],
    production_exit_rules: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Resolve TP/SL metadata for an API-confirmed open lot."""
    entry_record = dict(entry or {})
    entry_date_text = str(entry_record.get("entry_date") or "")
    signal_date_candidates = [
        str(entry_record.get("futu_buy_order_date") or ""),
        entry_date_text,
    ]
    signal_entry = None
    for signal_date_text in signal_date_candidates:
        if not signal_date_text:
            continue
        signal_entry = signal_entries_by_symbol_and_date.get(
            (symbol, signal_date_text)
        )
        if signal_entry is not None:
            break
    if signal_entry is None:
        bucket_candidates = [
            str(entry_record.get("bucket") or ""),
            str(entry_record.get("strategy_id") or ""),
        ]
        for signal_date_text in signal_date_candidates:
            if not signal_date_text:
                continue
            for bucket_candidate in bucket_candidates:
                if not bucket_candidate:
                    continue
                candidate_entry, matched_date_text = (
                    _walk_back_bucket_tp_sl_entry(
                        bucket_candidate=bucket_candidate,
                        anchor_date_text=signal_date_text,
                        bucket_tp_sl_entries_by_key_and_date=(
                            bucket_tp_sl_entries_by_key_and_date
                        ),
                    )
                )
                if candidate_entry is not None:
                    signal_entry = candidate_entry
                    if (
                        matched_date_text
                        and matched_date_text != signal_date_text
                    ):
                        LOGGER.info(
                            "[BUCKET_TP_SL_WALK_BACK] symbol=%s bucket=%s "
                            "anchor=%s matched=%s",
                            symbol,
                            bucket_candidate,
                            signal_date_text,
                            matched_date_text,
                        )
                    break
            if signal_entry is not None:
                break
    if signal_entry is None:
        return None

    bucket = str(signal_entry.get("bucket") or "")
    strategy_identifier = str(signal_entry.get("strategy_id") or "")
    production_rule = (
        production_exit_rules.get(bucket)
        or production_exit_rules.get(strategy_identifier)
        or {}
    )
    merged_entry = {
        **entry_record,
        **signal_entry,
        **production_rule,
        "symbol": symbol,
        "supports_tp_sl": "tp_pct" in signal_entry and "sl_pct" in signal_entry,
        "metadata_source": "production_signal_log",
    }
    return merged_entry


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

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    dry_run = "--dry-run" in sys.argv
    trading_environment = TrdEnv.REAL if TRADING_ENV == "REAL" else TrdEnv.SIMULATE

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
        signal_entries_by_symbol_and_date = _load_production_signal_entries()
        bucket_tp_sl_entries_by_key_and_date = _load_production_bucket_tp_sl_entries()
        production_exit_rules = _load_production_exit_rules()

        LOGGER.info("--- TP check ---")
        for code, position in positions.items():
            symbol = code.replace("US.", "")
            entry = _merge_production_signal_metadata(
                symbol=symbol,
                entry=open_entries_by_symbol.get(symbol),
                signal_entries_by_symbol_and_date=signal_entries_by_symbol_and_date,
                bucket_tp_sl_entries_by_key_and_date=bucket_tp_sl_entries_by_key_and_date,
                production_exit_rules=production_exit_rules,
            )
            if entry is None or not _entry_supports_take_profit_stop_loss(entry):
                LOGGER.warning(
                    "[TP_SL_METADATA_MISSING] code=%s qty=%d cost=$%.2f "
                    "entry_date=%s buy_order_id=%s — no production signal "
                    "metadata for API-confirmed lot, skipping TP/SL",
                    code,
                    position["qty"],
                    position["cost_price"],
                    (open_entries_by_symbol.get(symbol) or {}).get("entry_date"),
                    (open_entries_by_symbol.get(symbol) or {}).get(
                        "futu_buy_order_id",
                    ),
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
            entry = _merge_production_signal_metadata(
                symbol=symbol,
                entry=open_entries_by_symbol.get(symbol),
                signal_entries_by_symbol_and_date=signal_entries_by_symbol_and_date,
                bucket_tp_sl_entries_by_key_and_date=bucket_tp_sl_entries_by_key_and_date,
                production_exit_rules=production_exit_rules,
            )
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
