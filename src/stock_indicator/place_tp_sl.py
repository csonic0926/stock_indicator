"""System B: Place TP/SL orders based on Futu live positions.

Source of truth: Futu API (positions, orders, history).
TP/SL percentages come from adaptive_state.json (System A output).

Logic:
1. Query Futu positions → code, qty, cost_price
2. Query open sell orders → which positions already have TP / SL
3. Query history filled BUY → earliest fill date per code = entry date
4. Missing TP → GTC limit sell at cost_price * (1 + tp%)
5. Missing SL + bars_held >= sl_min_hold → GTC stop at cost_price * (1 - sl%)

Usage:
    venv/bin/python -m stock_indicator.place_tp_sl [--dry-run]
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

import pandas

LOGGER = logging.getLogger(__name__)

LOGS_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "logs"

TRADING_ENV = "REAL"
# SL-specific min_hold (decoupled from signal-exit min_hold). SL is risk
# control, not STATE confirmation, so it should not inherit signal latency.
# Mirrors AdaptiveTPSLConfig.min_hold_sl in strategy.py.
SL_MIN_HOLD_BARS = 1
# Skip SL order placement entirely. Matches new design where SL is computed
# as rolling regime indicator (drives dynamic min_hold throttle in backtest)
# but never fires as an actual exit. Live trades exit only via TP or signal.
SL_PLACEMENT_DISABLED = True


DATA_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "data"


def _get_tp_sl_from_adaptive_state() -> tuple[float | None, float | None]:
    """Read TP/SL percentages from adaptive_state.json (System A output)."""
    state_path = DATA_DIRECTORY / "adaptive_state.json"
    if not state_path.exists():
        return None, None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        tp = state.get("tp_pct")
        sl = state.get("sl_pct")
        if tp is not None:
            return float(tp), float(sl) if sl is not None else None
    except (json.JSONDecodeError, OSError, TypeError):
        pass
    return None, None


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

    # --- Adaptive TP/SL from adaptive_state.json (System A output) ---
    tp_pct, sl_pct = _get_tp_sl_from_adaptive_state()
    if tp_pct is None:
        LOGGER.error("No TP/SL found in logs")
        return
    LOGGER.info("Adaptive TP: %.2f%%, SL: %.2f%%", tp_pct * 100, sl_pct * 100)

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

    positions: dict[str, dict] = {}
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

    # --- 2. Query existing open sell orders ---
    ret_ord, ord_data = trd_ctx.order_list_query(trd_env=trd_env)
    existing_tp_codes: set[str] = set()  # NORMAL limit sell
    existing_sl_codes: set[str] = set()  # STOP sell
    if ret_ord == 0 and len(ord_data) > 0:
        for _, row in ord_data.iterrows():
            if str(row.get("trd_side", "")) != "SELL":
                continue
            if str(row.get("order_status", "")) in (
                "CANCELLED_ALL", "FAILED", "DELETED",
            ):
                continue
            code = str(row.get("code", ""))
            order_type = str(row.get("order_type", ""))
            if order_type == "STOP":
                existing_sl_codes.add(code)
            elif order_type == "NORMAL":
                existing_tp_codes.add(code)

    # --- 3. Query history filled BUY → entry date per code ---
    entry_dates: dict[str, str] = {}
    ret_hist, hist_data = trd_ctx.history_order_list_query(
        trd_env=trd_env,
    )
    if ret_hist == 0 and len(hist_data) > 0:
        for _, row in hist_data.iterrows():
            if str(row.get("trd_side", "")) != "BUY":
                continue
            if str(row.get("order_status", "")) not in (
                "FILLED_ALL", "FILLED_PART",
            ):
                continue
            code = str(row.get("code", ""))
            if code not in positions:
                continue
            # create_time format: "2026-04-22 21:34:27.949"
            create_time = str(row.get("create_time", ""))
            if not create_time:
                continue
            fill_date = create_time[:10]  # "2026-04-22"
            # Keep latest date per code — current position corresponds to
            # the most recent BUY, not the earliest (which may be a
            # previously closed trade).
            if code not in entry_dates or fill_date > entry_dates[code]:
                entry_dates[code] = fill_date

    # Also check active orders for today's fills not yet in history
    if ret_ord == 0 and len(ord_data) > 0:
        for _, row in ord_data.iterrows():
            if str(row.get("trd_side", "")) != "BUY":
                continue
            if str(row.get("order_status", "")) not in (
                "FILLED_ALL", "FILLED_PART",
            ):
                continue
            code = str(row.get("code", ""))
            if code not in positions:
                continue
            create_time = str(row.get("create_time", ""))
            if not create_time:
                continue
            fill_date = create_time[:10]
            if code not in entry_dates or fill_date > entry_dates[code]:
                entry_dates[code] = fill_date

    today_str = date.today().isoformat()

    # --- 4. Place TP for positions missing it ---
    LOGGER.info("--- TP check ---")
    for code, pos in positions.items():
        symbol = code.replace("US.", "")
        if code in existing_tp_codes:
            LOGGER.info("%s: TP already exists, skip", symbol)
            continue

        tp_price = round(pos["cost_price"] * (1 + tp_pct), 2)
        LOGGER.info(
            "%s: cost=$%.2f qty=%d → TP=$%.2f (+%.2f%%) [GTC limit sell]",
            symbol, pos["cost_price"], pos["qty"], tp_price, tp_pct * 100,
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
            "side": "TP_SELL",
            "qty": pos["qty"],
            "entry_price": pos["cost_price"],
            "price": tp_price,
            "tp_pct": round(tp_pct * 100, 2),
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

    # --- 5. Place SL for positions past sl_min_hold and missing SL ---
    if SL_PLACEMENT_DISABLED:
        LOGGER.info(
            "--- SL placement DISABLED — design uses rolling SL as regime "
            "indicator only, no actual SL orders placed ---"
        )
        trd_ctx.close()
        LOGGER.info("Done")
        return
    LOGGER.info("--- SL check (sl_min_hold=%d) ---", SL_MIN_HOLD_BARS)
    for code, pos in positions.items():
        symbol = code.replace("US.", "")

        if code in existing_sl_codes:
            LOGGER.info("%s: SL already exists, skip", symbol)
            continue

        entry_date_str = entry_dates.get(code)
        if not entry_date_str:
            LOGGER.info("%s: no entry date found in order history, skip SL", symbol)
            continue

        try:
            entry_ts = pandas.Timestamp(entry_date_str)
            today_ts = pandas.Timestamp(today_str)
            trading_days = pandas.bdate_range(entry_ts, today_ts)
            bars_held = max(0, len(trading_days) - 1)
        except Exception:
            bars_held = 0

        if bars_held < SL_MIN_HOLD_BARS:
            LOGGER.info(
                "%s: bars_held=%d < sl_min_hold=%d, SL deferred",
                symbol, bars_held, SL_MIN_HOLD_BARS,
            )
            continue

        sl_price = round(pos["cost_price"] * (1 - sl_pct), 2)
        LOGGER.info(
            "%s: bars_held=%d >= sl_min_hold=%d → SL=$%.2f (-%.2f%%) [GTC stop]",
            symbol, bars_held, SL_MIN_HOLD_BARS, sl_price, sl_pct * 100,
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
            "side": "SL_SELL",
            "qty": pos["qty"],
            "entry_price": pos["cost_price"],
            "price": sl_price,
            "sl_pct": round(sl_pct * 100, 2),
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
