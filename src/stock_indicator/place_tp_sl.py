"""System B: Place per-position TP/SL orders against Futu live positions.

Source of truth:
- Futu API for positions, orders, history.
- adaptive_state.json:accepted_entries for per-position frozen TP/SL.

Contract:
- Each open position MUST have a matching record in
  state["accepted_entries"]. Without one we mark it [ORPHAN_POSITION] and
  skip — never fall back to a global TP/SL value, since that would silently
  re-introduce the per-bucket asymmetry breakage we built this script to
  fix.
- TP price = cost_price * (1 + entry.tp_pct), GTC limit sell.
- SL price = cost_price * (1 - entry.sl_pct), GTC stop, only after
  sl_min_hold trading bars.

Usage:
    venv/bin/python -m stock_indicator.place_tp_sl [--dry-run]
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict

import pandas

LOGGER = logging.getLogger(__name__)

LOGS_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "logs"
DATA_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "data"

TRADING_ENV = "REAL"
# Default SL min_hold (used when entry record does not specify). SL is
# risk control, not STATE confirmation, so it does not inherit signal
# latency. Mirrors AdaptiveTPSLConfig.min_hold_sl in strategy.py.
DEFAULT_SL_MIN_HOLD_BARS = 1


def _load_accepted_entries() -> Dict[str, Dict[str, Any]]:
    """Return map of US.<symbol> -> entry record from adaptive_state.json.

    Returns empty mapping when the state file is absent or unreadable.
    Any read failure is logged at ERROR — a silent empty mapping would
    make every live position look like an [ORPHAN_POSITION], skipping all
    TP/SL placement. Real money: visible failure beats silent skip.
    """
    state_path = DATA_DIRECTORY / "adaptive_state.json"
    if not state_path.exists():
        LOGGER.error(
            "adaptive_state.json not found at %s — no entries loaded",
            state_path,
        )
        return {}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as load_error:
        LOGGER.error(
            "Failed to parse adaptive_state.json: %s. "
            "All open Futu positions will be flagged [ORPHAN_POSITION] "
            "until the state file is repaired.",
            load_error,
        )
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for entry in state.get("accepted_entries", []):
        symbol_value = entry.get("symbol")
        if not symbol_value:
            continue
        result[f"US.{symbol_value}"] = entry
    return result


def _log_order(order_data: dict) -> None:
    """Append order to today's order log."""
    today = date.today().isoformat()
    log_path = LOGS_DIRECTORY / f"{today}_orders.json"
    orders = []
    if log_path.exists():
        try:
            orders = json.loads(log_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    orders.append(order_data)
    log_path.write_text(json.dumps(orders, indent=2), encoding="utf-8")


def main() -> None:
    from datetime import datetime

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
    trd_env = TrdEnv.REAL if TRADING_ENV == "REAL" else TrdEnv.SIMULATE

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )

    # --- Load per-position frozen TP/SL from System A state ---
    accepted_entries = _load_accepted_entries()
    if not accepted_entries:
        LOGGER.warning(
            "No accepted_entries in adaptive_state.json — nothing to manage. "
            "Either no positions accepted yet, or migration not run."
        )

    # --- Connect to Futu ---
    trd_ctx = OpenSecTradeContext(
        host="127.0.0.1",
        port=11111,
        filter_trdmarket=TrdMarket.US,
        security_firm=SecurityFirm.FUTUSECURITIES,
    )

    # --- 1. Query positions ---
    ret_pos, pos_data = trd_ctx.position_list_query(trd_env=trd_env)
    if ret_pos != 0:
        LOGGER.error("Failed to query positions: %s", pos_data)
        trd_ctx.close()
        return

    positions: Dict[str, Dict[str, Any]] = {}
    if len(pos_data) > 0:
        for _, row in pos_data.iterrows():
            qty = int(row.get("qty", 0))
            if qty <= 0:
                continue
            code = str(row.get("code", ""))
            positions[code] = {
                "qty": qty,
                "cost_price": float(row.get("cost_price", 0)),
            }

    if not positions:
        LOGGER.info("No open positions")
        trd_ctx.close()
        return

    LOGGER.info("Positions: %s", ", ".join(
        f"{c.replace('US.', '')} qty={p['qty']} cost=${p['cost_price']:.2f}"
        for c, p in positions.items()
    ))

    # --- 2. Query existing open sell orders (TP / SL by order_type) ---
    ret_ord, ord_data = trd_ctx.order_list_query(trd_env=trd_env)
    existing_tp_codes: set[str] = set()
    existing_sl_codes: set[str] = set()
    if ret_ord == 0 and len(ord_data) > 0:
        for _, row in ord_data.iterrows():
            if str(row.get("trd_side", "")) != "SELL":
                continue
            if str(row.get("order_status", "")) in (
                "CANCELLED_ALL", "FAILED", "DELETED",
            ):
                continue
            code = str(row.get("code", ""))
            order_type_value = str(row.get("order_type", ""))
            if order_type_value == "STOP":
                existing_sl_codes.add(code)
            elif order_type_value == "NORMAL":
                existing_tp_codes.add(code)

    # --- 3. Place TP for positions missing it ---
    LOGGER.info("--- TP check ---")
    for code, pos in positions.items():
        symbol = code.replace("US.", "")

        entry = accepted_entries.get(code)
        if entry is None:
            LOGGER.warning(
                "[ORPHAN_POSITION] code=%s qty=%d cost=$%.2f — "
                "no accepted_entries match, skipping TP/SL",
                code, pos["qty"], pos["cost_price"],
            )
            continue

        if code in existing_tp_codes:
            LOGGER.info("%s: TP already exists, skip", symbol)
            continue

        tp_pct = float(entry.get("tp_pct", 0))
        if tp_pct <= 0:
            LOGGER.info("%s: tp_pct <= 0 (entry=%s), skip TP", symbol, tp_pct)
            continue

        tp_price = round(pos["cost_price"] * (1 + tp_pct), 2)
        bucket = entry.get("bucket", "?")
        LOGGER.info(
            "%s [%s]: cost=$%.2f qty=%d -> TP=$%.2f (+%.2f%%) [GTC limit sell]",
            symbol, bucket, pos["cost_price"], pos["qty"],
            tp_price, tp_pct * 100,
        )

        if dry_run:
            LOGGER.info("  [DRY RUN] skipping")
            continue

        ret_tp, data_tp = trd_ctx.place_order(
            price=tp_price,
            qty=pos["qty"],
            code=code,
            trd_side=TrdSide.SELL,
            order_type=OrderType.NORMAL,
            trd_env=trd_env,
            time_in_force=TimeInForce.GTC,
        )
        tp_result = {
            "symbol": symbol,
            "bucket": bucket,
            "side": "TP_SELL",
            "qty": pos["qty"],
            "entry_price": pos["cost_price"],
            "price": tp_price,
            "tp_pct": round(tp_pct * 100, 4),
            "status": "sent" if ret_tp == 0 else "failed",
            "order_id": (
                str(data_tp.iloc[0].get("order_id", ""))
                if ret_tp == 0 else None
            ),
            "error": str(data_tp) if ret_tp != 0 else None,
            "env": TRADING_ENV,
            "timestamp": datetime.now().isoformat(),
        }
        LOGGER.info("  TP: %s", tp_result["status"])
        _log_order(tp_result)

    # --- 4. Place SL for positions past sl_min_hold and missing SL ---
    LOGGER.info("--- SL check ---")
    today_str = date.today().isoformat()
    for code, pos in positions.items():
        symbol = code.replace("US.", "")

        entry = accepted_entries.get(code)
        if entry is None:
            # Already warned in TP loop; skip silently here.
            continue

        if code in existing_sl_codes:
            LOGGER.info("%s: SL already exists, skip", symbol)
            continue

        sl_pct = float(entry.get("sl_pct", 0))
        if sl_pct <= 0:
            LOGGER.info("%s: sl_pct <= 0 (entry=%s), skip SL", symbol, sl_pct)
            continue

        entry_date_str = entry.get("entry_date") or ""
        if not entry_date_str:
            LOGGER.info(
                "%s: no entry_date in accepted_entries, skip SL", symbol,
            )
            continue

        try:
            entry_ts = pandas.Timestamp(entry_date_str)
            today_ts = pandas.Timestamp(today_str)
            trading_days = pandas.bdate_range(entry_ts, today_ts)
            bars_held = max(0, len(trading_days) - 1)
        except Exception:  # noqa: BLE001
            bars_held = 0

        sl_min_hold = int(
            entry.get("min_hold_sl", DEFAULT_SL_MIN_HOLD_BARS) or DEFAULT_SL_MIN_HOLD_BARS
        )
        if bars_held < sl_min_hold:
            LOGGER.info(
                "%s: bars_held=%d < min_hold_sl=%d, SL deferred",
                symbol, bars_held, sl_min_hold,
            )
            continue

        sl_price = round(pos["cost_price"] * (1 - sl_pct), 2)
        bucket = entry.get("bucket", "?")
        LOGGER.info(
            "%s [%s]: bars_held=%d -> SL=$%.2f (-%.2f%%) [GTC stop]",
            symbol, bucket, bars_held, sl_price, sl_pct * 100,
        )

        if dry_run:
            LOGGER.info("  [DRY RUN] skipping")
            continue

        ret_sl, data_sl = trd_ctx.place_order(
            price=sl_price,
            qty=pos["qty"],
            code=code,
            trd_side=TrdSide.SELL,
            order_type=OrderType.STOP,
            trd_env=trd_env,
            aux_price=sl_price,
            time_in_force=TimeInForce.GTC,
        )
        sl_result = {
            "symbol": symbol,
            "bucket": bucket,
            "side": "SL_SELL",
            "qty": pos["qty"],
            "entry_price": pos["cost_price"],
            "price": sl_price,
            "sl_pct": round(sl_pct * 100, 4),
            "bars_held": bars_held,
            "status": "sent" if ret_sl == 0 else "failed",
            "order_id": (
                str(data_sl.iloc[0].get("order_id", ""))
                if ret_sl == 0 else None
            ),
            "error": str(data_sl) if ret_sl != 0 else None,
            "env": TRADING_ENV,
            "timestamp": datetime.now().isoformat(),
        }
        LOGGER.info("  SL: %s", sl_result["status"])
        _log_order(sl_result)

    trd_ctx.close()
    LOGGER.info("Done")


if __name__ == "__main__":
    main()
