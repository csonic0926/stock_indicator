"""Web dashboard for stock indicator signals and portfolio state."""

from __future__ import annotations

import ast
import json
import logging
import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

LOGGER = logging.getLogger(__name__)

DATA_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "data"
LOGS_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "logs"

app = FastAPI(title="Stock Indicator Dashboard")


def _parse_log_key_value_tokens(line_body: str) -> dict[str, Any]:
    """Parse machine-readable dashboard log tokens into a dict."""
    fields: dict[str, Any] = {}
    for token_text in line_body.strip().split():
        if "=" not in token_text:
            continue
        key_text, _, raw_value_text = token_text.partition("=")
        if raw_value_text == "None":
            fields[key_text] = None
            continue
        try:
            if key_text in {
                "tp_pct",
                "sl_pct",
                "rolling_mp",
                "slope_60",
                "near_delta",
            }:
                fields[key_text] = float(raw_value_text)
            elif key_text in {"min_hold_tp", "dollar_volume_rank"}:
                fields[key_text] = int(raw_value_text)
            elif key_text == "disable_sl_trigger":
                fields[key_text] = raw_value_text.lower() == "true"
            else:
                fields[key_text] = raw_value_text
        except ValueError:
            fields[key_text] = raw_value_text
    return fields


def _parse_log(log_path: Path) -> dict[str, Any]:
    """Parse a daily multi-bucket log into structured data."""
    text = log_path.read_text(encoding="utf-8")
    result: dict[str, Any] = {"raw": text, "date": log_path.stem}

    # TODO: review
    # Raw strategy entry signals are emitted before dashboard/order-layer
    # checks. These should remain visible even when no BUY order is accepted
    # or when the risk-score gate later blocks order submission.
    entry_signal_records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith("[ENTRY_SIGNAL]"):
            continue
        entry_signal_records.append(
            _parse_log_key_value_tokens(line[len("[ENTRY_SIGNAL]"):])
        )
    result["entry_signal_records"] = entry_signal_records
    result["entry_signals"] = [
        record.get("symbol")
        for record in entry_signal_records
        if record.get("symbol")
    ]

    # Bucket-aware per-entry frozen TP/SL (multi-bucket schema).
    # Format: [FROZEN_TP_SL] entry_date=... bucket=... strategy_id=...
    #         symbol=... tp_pct=... sl_pct=... rolling_mp=...
    #         slope_60=... near_delta=... min_hold_tp=... disable_sl_trigger=...
    frozen_entries: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith("[FROZEN_TP_SL]"):
            continue
        frozen_entries.append(
            _parse_log_key_value_tokens(line[len("[FROZEN_TP_SL]"):])
        )
    result["frozen_entries"] = frozen_entries
    # Convenience: lists of new BUY symbols today (from frozen entries
    # whose entry_date matches today's log date).
    todays_buy_symbols = [
        e.get("symbol") for e in frozen_entries
        if e.get("entry_date") == log_path.stem and e.get("symbol")
    ]
    result["buy_actions"] = todays_buy_symbols
    result["accepted_buy_actions"] = todays_buy_symbols

    accepted_match = re.search(r"^accepted:\s*(.+)$", text, re.MULTILINE)
    rejected_match = re.search(r"^rejected:\s*(.+)$", text, re.MULTILINE)
    result["slot_allocation"] = {
        "accepted": _parse_slot_allocation_pairs(accepted_match.group(1))
        if accepted_match else [],
        "rejected": _parse_slot_allocation_pairs(rejected_match.group(1))
        if rejected_match else [],
    }
    allocation_summary_match = re.search(
        r"max_position_count=(\d+)\s+"
        r"held_before_today=(\d+)\s+"
        r"same_day_closes=(\d+)",
        text,
    )
    if allocation_summary_match:
        result["slot_allocation"].update({
            "max_position_count": int(allocation_summary_match.group(1)),
            "held_before_today": int(allocation_summary_match.group(2)),
            "same_day_closes": int(allocation_summary_match.group(3)),
        })

    # Exit signals (machine-readable per-symbol lines).
    # Format: [EXIT_SIGNAL] symbol=X
    exit_symbols: list[str] = []
    for line in text.splitlines():
        if not line.startswith("[EXIT_SIGNAL]"):
            continue
        for token in line[len("[EXIT_SIGNAL]"):].strip().split():
            if token.startswith("symbol="):
                exit_symbols.append(token.partition("=")[2])
    result["exit_signals"] = exit_symbols
    result["sell_actions"] = exit_symbols

    # Hold (min_hold) — kept for back-compat with single-strategy
    # find_history_signal log emitter; multi_bucket log doesn't emit
    # this currently.
    hold_m = re.search(r"HOLD\s+\(min_hold\)\s+(.+)", text)
    result["hold_blocked"] = (
        re.findall(r"'(\w+)'", hold_m.group(1)) if hold_m else []
    )

    # Global rolling stats from compute_adaptive_tp_sl (legacy bridge,
    # still emitted by run_daily_job.sh for diagnostic display only —
    # NOT used for order TP/SL prices, those come from frozen_entries).
    tp_m = re.search(r"TP:\s*([\d.]+)%", text)
    sl_m = re.search(r"SL:\s*([\d.]+)%", text)
    result["tp_pct"] = float(tp_m.group(1)) if tp_m else None
    result["sl_pct"] = float(sl_m.group(1)) if sl_m else None

    mp_m = re.search(r"Rolling MP:\s*([\d.]+)%\s*\(n=(\d+)\)", text)
    ml_m = re.search(r"Rolling ML:\s*-?([\d.]+)%\s*\(n=(\d+)\)", text)
    if mp_m:
        result["rolling_mp"] = float(mp_m.group(1))
        result["rolling_mp_n"] = int(mp_m.group(2))
    if ml_m:
        result["rolling_ml"] = float(ml_m.group(1))
        result["rolling_ml_n"] = int(ml_m.group(2))

    # Positions
    pos_m = re.search(r"Concurrent positions after entry \((\d+) total\)", text)
    if pos_m:
        result["position_count"] = int(pos_m.group(1))
    elif allocation_summary_match:
        result["position_count"] = (
            int(allocation_summary_match.group(2))
            - int(allocation_summary_match.group(3))
            + len(result["slot_allocation"]["accepted"])
        )
    else:
        result["position_count"] = 0

    return result


def _parse_slot_allocation_pairs(raw_allocation_text: str) -> list[dict[str, str]]:
    """Parse cron's accepted/rejected slot-allocation tuple summaries."""
    try:
        raw_records = ast.literal_eval(raw_allocation_text)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(raw_records, list):
        return []

    parsed_records: list[dict[str, str]] = []
    for raw_record in raw_records:
        if not isinstance(raw_record, tuple) or len(raw_record) < 2:
            continue
        parsed_record = {
            "symbol": str(raw_record[0]),
            "bucket": str(raw_record[1]),
        }
        if len(raw_record) >= 3:
            parsed_record["reason"] = str(raw_record[2])
        parsed_records.append(parsed_record)
    return parsed_records


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _load_current_bucket_tp_sl(adaptive_state: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute current per-bucket base TP/SL from config and rolling state.

    This diagnostic is intentionally separate from order preview. BUY orders
    still use entry-time [FROZEN_TP_SL] values from the cron log, while the
    dashboard summary should reflect current config + rolling pool after a
    restart or code fix rather than stale historical frozen lines.
    """
    try:
        from stock_indicator import multi_bucket_today, strategy

        config = multi_bucket_today.load_multi_bucket_config(
            PRODUCTION_CONFIG_PATH
        )
        if config.adaptive_tp_sl is None:
            return []
        bucket_states: list[dict[str, Any]] = []
        for bucket_label, bucket_definition in config.bucket_definitions.items():
            tp_pct, sl_pct, rolling_mp, rolling_ml = (
                strategy.compute_frozen_tp_sl_for_bucket(
                    bucket_def=bucket_definition,
                    adaptive_tp_sl=config.adaptive_tp_sl,
                    closed_winners=adaptive_state.get("winners", []),
                    closed_losers=adaptive_state.get("losers", []),
                    entry_slope_60=None,
                )
            )
            bucket_states.append({
                "bucket": bucket_label,
                "strategy_id": bucket_definition.strategy_identifier,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "rolling_mp": rolling_mp,
                "rolling_ml": rolling_ml,
                "sigma": bucket_definition.sigma,
                "tp_regime_adjust": bucket_definition.tp_regime_adjust,
                "tp_slope_amplify": bucket_definition.tp_slope_amplify,
                "source": "current_config_base",
            })
        return bucket_states
    except (OSError, json.JSONDecodeError, ValueError) as config_error:
        LOGGER.warning("Failed to compute current bucket TP/SL: %s", config_error)
        return []


def _get_log_dates() -> list[str]:
    """Return sorted list of available log dates (newest first)."""
    dates = []
    for f in LOGS_DIRECTORY.glob("*.log"):
        try:
            date.fromisoformat(f.stem)
            dates.append(f.stem)
        except ValueError:
            continue
    return sorted(dates, reverse=True)


@app.get("/api/state")
def api_state():
    """Current system state: signal trades, adaptive state, latest log."""
    signal_trades = _load_json(DATA_DIRECTORY / "signal_trades.json")
    adaptive = _load_json(DATA_DIRECTORY / "adaptive_state.json")
    current_bucket_tp_sl = _load_current_bucket_tp_sl(adaptive)
    dates = _get_log_dates()
    latest_log = None
    if dates:
        latest_log = _parse_log(LOGS_DIRECTORY / f"{dates[0]}.log")
    return {
        "signal_trades": signal_trades,
        "adaptive_state": adaptive,
        "current_bucket_tp_sl": current_bucket_tp_sl,
        "latest_log": latest_log,
        "available_dates": dates[:30],
    }


@app.get("/api/log/{log_date}")
def api_log(log_date: str):
    """Parse a specific date's log."""
    path = LOGS_DIRECTORY / f"{log_date}.log"
    if not path.exists():
        return {"error": "not found"}
    return _parse_log(path)


@app.get("/api/trades")
def api_trades():
    """Rolling trade history from adaptive_state.json."""
    adaptive = _load_json(DATA_DIRECTORY / "adaptive_state.json")
    return {
        "raw_trade_profits": adaptive.get("raw_trade_profits", []),
        "closed_trades": adaptive.get("closed_trades", []),
    }


@app.get("/api/futu/positions")
def api_futu_positions():
    """Live positions from Futu OpenD (if connected)."""
    try:
        from futu import (
            OpenSecTradeContext,
            SecurityFirm,
            TrdEnv,
            TrdMarket,
        )

        trd_ctx = OpenSecTradeContext(
            host="127.0.0.1",
            port=11111,
            filter_trdmarket=TrdMarket.US,
            security_firm=SecurityFirm.FUTUSECURITIES,
        )
        ret_pos, pos_data = trd_ctx.position_list_query(trd_env=TrdEnv.REAL)
        ret_acc, acc_data = trd_ctx.accinfo_query(trd_env=TrdEnv.REAL)
        trd_ctx.close()

        positions = []
        if ret_pos == 0 and len(pos_data) > 0:
            for _, row in pos_data.iterrows():
                positions.append({
                    "symbol": str(row.get("code", "")).replace("US.", ""),
                    "qty": float(row.get("qty", 0)),
                    "cost_price": float(row.get("cost_price", 0)),
                    "market_price": float(row.get("nominal_price", 0)),
                    "market_val": float(row.get("market_val", 0)),
                    "unrealized_pl": float(row.get("unrealized_pl", 0)),
                    "pl_ratio": float(row.get("pl_ratio", 0)),
                })

        account = {}
        if ret_acc == 0 and len(acc_data) > 0:
            row = acc_data.iloc[0]
            account = {
                "total_assets": float(row.get("total_assets", 0)),
                "cash": float(row.get("cash", 0)),
                "us_cash": float(row.get("us_cash", 0)),
                "market_val": float(row.get("market_val", 0)),
                "power": float(row.get("power", 0)),
            }

        return {"connected": True, "positions": positions, "account": account}
    except Exception as exc:
        return {"connected": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------

MARGIN_MULTIPLIER = 1.5
# Use paper trading by default. Set to "REAL" to trade with real money.
TRADING_ENV = "REAL"
ORDER_LOG_DIR = LOGS_DIRECTORY

PRODUCTION_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "multi_bucket_production.json"
)
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
RISK_GATE_SKIP_STATUSES = {"slot_full", "risk_score_stop"}

# TODO: review

def _load_production_config() -> dict[str, Any]:
    """Read the live multi-bucket config without caching it."""
    with PRODUCTION_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config_document = json.load(config_file)
    if not isinstance(config_document, dict):
        raise ValueError("production config root must be a JSON object")
    return config_document


def _load_max_positions() -> int:
    """Read max_position_count from the production multi-bucket config.

    Falls back to 6 (legacy single-bucket sizing) if the config is missing
    or unreadable. Live cron uses this same JSON to drive multi_bucket_daily_signal,
    so dashboard sizing stays aligned without a hardcoded constant.
    """
    try:
        return int(_load_production_config().get("max_position_count", 6))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 6


def _resolve_config_path(path_text: str) -> Path:
    """Resolve a config-relative path from the repository root."""
    resolved_path = Path(path_text).expanduser()
    if resolved_path.is_absolute():
        return resolved_path
    return REPOSITORY_ROOT / resolved_path


def _load_risk_score_gate_state(signal_date: str | None) -> dict[str, Any]:
    """Return the dashboard order gate state for a signal date.

    The risk-score gate belongs to the dashboard/order layer: cron should
    keep emitting signals and updating rolling state, while this function
    decides whether BUY orders are allowed to leave the dashboard.
    """
    default_state: dict[str, Any] = {
        "enabled": False,
        "status": "off",
        "year_month": None,
        "risk_score": None,
        "stop_threshold": None,
        "reason": "risk_score_gate not configured",
    }
    if not signal_date:
        return {**default_state, "status": "error", "reason": "missing signal date"}

    try:
        signal_month = date.fromisoformat(signal_date).strftime("%Y-%m")
    except ValueError:
        return {
            **default_state,
            "status": "error",
            "reason": f"invalid signal date: {signal_date}",
        }

    try:
        config_document = _load_production_config()
    except (OSError, json.JSONDecodeError, ValueError) as config_error:
        return {
            **default_state,
            "status": "error",
            "year_month": signal_month,
            "reason": f"failed to read production config: {config_error}",
        }

    raw_gate = config_document.get("risk_score_gate")
    if raw_gate is None:
        return {**default_state, "year_month": signal_month}
    if not isinstance(raw_gate, dict):
        return {
            **default_state,
            "enabled": True,
            "status": "error",
            "year_month": signal_month,
            "reason": "risk_score_gate must be a JSON object",
        }

    try:
        stop_threshold = int(raw_gate.get("stop_threshold", 75))
    except (TypeError, ValueError) as threshold_error:
        return {
            **default_state,
            "enabled": True,
            "status": "error",
            "year_month": signal_month,
            "reason": f"invalid risk_score_gate.stop_threshold: {threshold_error}",
        }

    csv_path_text = str(raw_gate.get("csv_path", "")).strip()
    if not csv_path_text:
        return {
            **default_state,
            "enabled": True,
            "status": "error",
            "year_month": signal_month,
            "stop_threshold": stop_threshold,
            "reason": "risk_score_gate.csv_path is required",
        }

    csv_path = _resolve_config_path(csv_path_text)
    if not csv_path.exists():
        return {
            **default_state,
            "enabled": True,
            "status": "error",
            "year_month": signal_month,
            "stop_threshold": stop_threshold,
            "reason": f"risk score CSV not found: {csv_path}",
        }

    import csv as csv_module

    try:
        with csv_path.open("r", newline="") as risk_score_file:
            for row in csv_module.DictReader(risk_score_file):
                if row.get("year_month") != signal_month:
                    continue
                risk_score = int(row["risk_score"])
                status = "stop" if risk_score >= stop_threshold else "open"
                return {
                    "enabled": True,
                    "status": status,
                    "year_month": signal_month,
                    "risk_score": risk_score,
                    "stop_threshold": stop_threshold,
                    "reason": (
                        "risk_score >= stop_threshold"
                        if status == "stop"
                        else "risk_score below stop_threshold"
                    ),
                }
    except (OSError, KeyError, TypeError, ValueError) as csv_error:
        return {
            **default_state,
            "enabled": True,
            "status": "error",
            "year_month": signal_month,
            "stop_threshold": stop_threshold,
            "reason": f"failed to read risk score CSV: {csv_error}",
        }

    return {
        **default_state,
        "enabled": True,
        "status": "error",
        "year_month": signal_month,
        "stop_threshold": stop_threshold,
        "reason": f"risk score missing for {signal_month}",
    }


def _risk_gate_blocks_buy_orders(gate_state: dict[str, Any]) -> bool:
    """Return True when dashboard must not send BUY orders."""
    return gate_state.get("status") == "stop"


# NOTE: do NOT cache this at module load — re-read per request so config
# edits take effect without restarting the server.


def _get_futu_trd_ctx():
    """Create a Futu trade context. Caller must .close() it."""
    from futu import OpenSecTradeContext, SecurityFirm, TrdMarket

    return OpenSecTradeContext(
        host="127.0.0.1",
        port=11111,
        filter_trdmarket=TrdMarket.US,
        security_firm=SecurityFirm.FUTUSECURITIES,
    )


def _get_trd_env():
    from futu import TrdEnv

    return TrdEnv.REAL if TRADING_ENV == "REAL" else TrdEnv.SIMULATE


def _get_last_price(symbol: str) -> float | None:
    """Get last traded price for a US stock via Futu snapshot API."""
    try:
        from futu import OpenQuoteContext

        quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
        ret, data = quote_ctx.get_market_snapshot([f"US.{symbol}"])
        quote_ctx.close()
        if ret == 0 and len(data) > 0:
            price = float(data.iloc[0]["last_price"])
            if price > 0:
                return price
    except Exception:
        pass
    return None


def _compute_order_size(
    total_assets_hkd: float, price_usd: float, max_positions: int
) -> int:
    """Compute qty: floor(total_assets * margin / max_pos / price).

    total_assets is in HKD, price is in USD.  Uses ~7.8 HKD/USD.
    """
    hkd_per_position = total_assets_hkd * MARGIN_MULTIPLIER / max_positions
    usd_per_position = hkd_per_position / 7.8
    if price_usd <= 0:
        return 0
    return math.floor(usd_per_position / price_usd)


def _log_order(order_data: dict) -> None:
    """Append order to today's order log."""
    today = date.today().isoformat()
    log_path = ORDER_LOG_DIR / f"{today}_orders.json"
    orders = []
    if log_path.exists():
        try:
            orders = json.loads(log_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    orders.append(order_data)
    log_path.write_text(json.dumps(orders, indent=2), encoding="utf-8")


SIGNAL_TRADES_PATH = DATA_DIRECTORY / "signal_trades.json"


def _load_signal_trades() -> dict:
    """Read signal_trades.json. Returns empty dict on miss/parse error."""
    if not SIGNAL_TRADES_PATH.exists():
        return {}
    try:
        data = json.loads(SIGNAL_TRADES_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _save_signal_trades(data: dict) -> None:
    """Atomic write of signal_trades.json (real position ledger)."""
    tmp_path = SIGNAL_TRADES_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    import os as _os
    _os.replace(tmp_path, SIGNAL_TRADES_PATH)


def _record_buy_in_signal_trades(
    symbol: str, strategy_id: str, entry_date: str
) -> None:
    """Append a real BUY fill to signal_trades.json under its strategy_id.

    signal_trades.json is the real position ledger (cron does not touch
    it anymore). Adding here means 'Cal has actually executed this BUY
    via Futu'.
    """
    if not strategy_id:
        # No strategy_id (manual BUY outside cron signal) — fall back
        # to a synthetic key so the entry is still recorded.
        strategy_id = "manual"
    trades = _load_signal_trades()
    bucket_list = trades.setdefault(strategy_id, [])
    # Dedup: if symbol already present, refresh entry_date.
    for existing in bucket_list:
        if existing.get("symbol") == symbol:
            existing["entry_date"] = entry_date
            _save_signal_trades(trades)
            return
    bucket_list.append({"symbol": symbol, "entry_date": entry_date})
    _save_signal_trades(trades)


def _remove_sell_from_signal_trades(symbol: str) -> None:
    """Remove a symbol from signal_trades.json on confirmed SELL fill.

    Looks across all strategy buckets — the order layer does not
    require us to know which bucket originally entered.
    """
    trades = _load_signal_trades()
    changed = False
    for strategy_id, bucket_list in list(trades.items()):
        before = len(bucket_list)
        trades[strategy_id] = [
            entry for entry in bucket_list if entry.get("symbol") != symbol
        ]
        if len(trades[strategy_id]) != before:
            changed = True
    if changed:
        _save_signal_trades(trades)


@app.get("/api/preview_orders")
def api_preview_orders():
    """Build order preview from latest signal + account data."""
    max_positions = _load_max_positions()
    try:
        from futu import TrdEnv

        trd_ctx = _get_futu_trd_ctx()
        trd_env = _get_trd_env()
        ret_acc, acc_data = trd_ctx.accinfo_query(trd_env=trd_env)
        ret_pos, pos_data = trd_ctx.position_list_query(trd_env=trd_env)
        trd_ctx.close()

        if ret_acc != 0:
            return {"error": "Failed to query account"}

        total_assets = float(acc_data.iloc[0].get("total_assets", 0))
        held_symbols = set()
        if ret_pos == 0 and len(pos_data) > 0:
            for _, row in pos_data.iterrows():
                held_symbols.add(str(row.get("code", "")).replace("US.", ""))

    except Exception as exc:
        return {"error": f"Futu connection failed: {exc}"}

    # Parse latest log for signals
    dates = _get_log_dates()
    if not dates:
        return {"error": "No log files found"}
    log = _parse_log(LOGS_DIRECTORY / f"{dates[0]}.log")

    risk_gate_state = _load_risk_score_gate_state(log.get("date"))
    if risk_gate_state.get("status") == "error":
        return {"error": risk_gate_state.get("reason"), "risk_score_gate": risk_gate_state}
    buy_orders_blocked_by_risk_score = _risk_gate_blocks_buy_orders(risk_gate_state)

    # Build per-symbol metadata from today's [FROZEN_TP_SL] lines so the
    # preview can show bucket / tp_pct / sl_pct alongside each BUY.
    frozen_today: dict[str, dict[str, Any]] = {
        e.get("symbol"): e
        for e in log.get("frozen_entries", [])
        if e.get("entry_date") == log.get("date") and e.get("symbol")
    }

    orders = []

    # Slot cap based on REAL Futu portfolio (this is the order layer's
    # job — cron's signal layer emits Top N candidates regardless of
    # holdings). slots_remaining counts down as each BUY is queued.
    slots_remaining = max(0, max_positions - len(held_symbols))

    # BUY orders (today's accepted entries from [FROZEN_TP_SL] lines).
    # Iteration order follows log emission, which mirrors cron's
    # priority + dollar_volume sort — so the first to consume slots
    # are the highest-priority candidates.
    for symbol in log.get("buy_actions", []):
        if symbol in held_symbols:
            # Already long; cron emitted the signal but the order
            # layer skips dupes.
            continue
        ref_price = _get_last_price(symbol)
        qty = (
            _compute_order_size(total_assets, ref_price, max_positions)
            if ref_price else 0
        )
        meta = frozen_today.get(symbol, {})
        order_dict = {
            "side": "BUY",
            "symbol": symbol,
            "qty": qty,
            "ref_price": ref_price,
            "order_type": "MARKET",
            "bucket": meta.get("bucket"),
            "strategy_id": meta.get("strategy_id"),
            "tp_pct": meta.get("tp_pct"),
            "sl_pct": meta.get("sl_pct"),
            "dollar_volume_rank": meta.get("dollar_volume_rank"),
        }
        if buy_orders_blocked_by_risk_score:
            order_dict["status"] = "risk_score_stop"
            order_dict["qty"] = 0
            order_dict["skip_reason"] = (
                f"risk_score={risk_gate_state.get('risk_score')} >= "
                f"stop_threshold={risk_gate_state.get('stop_threshold')} "
                f"for {risk_gate_state.get('year_month')}"
            )
        elif slots_remaining <= 0:
            order_dict["status"] = "slot_full"
            order_dict["skip_reason"] = (
                f"max_positions={max_positions} already filled "
                f"(Futu held: {len(held_symbols)})"
            )
        else:
            slots_remaining -= 1
        orders.append(order_dict)

    # SELL orders (exit signals for held positions).
    for symbol in log.get("sell_actions", []):
        if symbol not in held_symbols:
            continue
        qty = 0
        if ret_pos == 0 and len(pos_data) > 0:
            match = pos_data[pos_data["code"] == f"US.{symbol}"]
            if len(match) > 0:
                qty = int(match.iloc[0].get("qty", 0))
        price = _get_last_price(symbol)
        orders.append({
            "side": "SELL",
            "symbol": symbol,
            "qty": qty,
            "price": price,
            "order_type": "MARKET",
        })

    return {
        "orders": orders,
        "total_assets_hkd": total_assets,
        "margin": MARGIN_MULTIPLIER,
        "max_positions": max_positions,
        "held_count": len(held_symbols),
        "trading_env": TRADING_ENV,
        "signal_date": log.get("date"),
        "risk_score_gate": risk_gate_state,
    }


class ExecuteRequest(BaseModel):
    orders: list[dict]


@app.post("/api/execute_orders")
def api_execute_orders(req: ExecuteRequest):
    """Execute confirmed orders via Futu API."""
    from futu import ModifyOrderOp, OrderType, TrdSide

    trd_ctx = _get_futu_trd_ctx()
    trd_env = _get_trd_env()
    results = []

    latest_gate_state: dict[str, Any] | None = None
    latest_dates = _get_log_dates()
    if latest_dates:
        latest_gate_state = _load_risk_score_gate_state(latest_dates[0])

    for order in req.orders:
        symbol = order["symbol"]
        side = TrdSide.BUY if order["side"] == "BUY" else TrdSide.SELL
        qty = order["qty"]
        code = f"US.{symbol}"

        # Preview may mark BUY orders as non-sendable (slot cap or
        # dashboard risk-score stop). Skip without sending so confirm cannot
        # bypass the preview-layer guardrails.
        if order.get("status") in RISK_GATE_SKIP_STATUSES:
            results.append({
                "symbol": symbol,
                "status": "skipped",
                "reason": order.get("skip_reason") or order.get("status"),
            })
            continue

        if (
            order.get("side") == "BUY"
            and latest_gate_state is not None
            and latest_gate_state.get("status") in {"stop", "error"}
        ):
            results.append({
                "symbol": symbol,
                "status": "skipped",
                "reason": latest_gate_state.get("reason") or "risk_score_gate",
            })
            continue

        if qty <= 0:
            results.append({"symbol": symbol, "status": "skipped", "reason": "qty=0"})
            continue

        try:
            # For SELL orders, cancel any existing pending sell orders
            # (TP/SL) first — Futu locks can_sell_qty when sell orders exist.
            if order["side"] == "SELL":
                ret_ord, ord_data = trd_ctx.order_list_query(trd_env=trd_env)
                if ret_ord == 0 and len(ord_data) > 0:
                    for _, orow in ord_data.iterrows():
                        if (
                            str(orow.get("code", "")) == code
                            and str(orow.get("trd_side", "")) == "SELL"
                            and str(orow.get("order_status", "")) not in (
                                "CANCELLED_ALL", "FILLED_ALL", "FAILED", "DELETED",
                            )
                        ):
                            cancel_id = str(orow.get("order_id", ""))
                            ret_cancel, _ = trd_ctx.modify_order(
                                modify_order_op=ModifyOrderOp.CANCEL,
                                order_id=cancel_id,
                                qty=0,
                                price=0,
                                trd_env=trd_env,
                            )
                            LOGGER.info(
                                "Cancelled pending sell %s for %s (ret=%s)",
                                cancel_id, symbol, ret_cancel,
                            )
                import time
                time.sleep(1)  # wait for Futu to release can_sell_qty

            # Market order for entry/exit
            ret, data = trd_ctx.place_order(
                price=0.0,
                qty=qty,
                code=code,
                trd_side=side,
                order_type=OrderType.MARKET,
                trd_env=trd_env,
            )
            if ret == 0:
                order_id = str(data.iloc[0].get("order_id", ""))
                result = {
                    "symbol": symbol,
                    "side": order["side"],
                    "qty": qty,
                    "status": "sent",
                    "order_id": order_id,
                    "env": TRADING_ENV,
                    "timestamp": datetime.now().isoformat(),
                }
                results.append(result)
                _log_order(result)
                # Real position ledger update — dashboard owns
                # signal_trades.json entirely now that cron has stopped
                # writing it.
                if order["side"] == "BUY":
                    _record_buy_in_signal_trades(
                        symbol=symbol,
                        strategy_id=order.get("strategy_id", "") or "",
                        entry_date=date.today().isoformat(),
                    )
                else:
                    _remove_sell_from_signal_trades(symbol)
            else:
                results.append({
                    "symbol": symbol,
                    "status": "failed",
                    "error": str(data),
                })
        except Exception as exc:
            results.append({
                "symbol": symbol,
                "status": "error",
                "error": str(exc),
            })

    trd_ctx.close()
    return {"results": results}


@app.post("/api/place_tp_sl")
def api_place_tp_sl():
    """Trigger TP/SL placement (same logic as place_tp_sl.py)."""
    try:
        from stock_indicator.place_tp_sl import main as _tp_sl_main

        import io
        import contextlib

        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        tp_sl_logger = logging.getLogger("stock_indicator.place_tp_sl")
        tp_sl_logger.addHandler(handler)
        tp_sl_logger.setLevel(logging.INFO)

        try:
            _tp_sl_main()
        finally:
            tp_sl_logger.removeHandler(handler)

        return {"ok": True, "log": buf.getvalue()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stock Indicator Dashboard</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --text2: #8b949e; --green: #3fb950;
    --red: #f85149; --blue: #58a6ff; --orange: #d29922;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'SF Mono', 'Cascadia Code', monospace; background: var(--bg); color: var(--text); padding: 20px; }
  h1 { font-size: 1.2em; margin-bottom: 16px; color: var(--blue); }
  h2 { font-size: 1em; margin: 16px 0 8px; color: var(--text2); text-transform: uppercase; letter-spacing: 1px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .card.full { grid-column: 1 / -1; }
  .stat { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid var(--border); }
  .stat:last-child { border-bottom: none; }
  .stat .label { color: var(--text2); }
  .stat .value { font-weight: bold; }
  .positive { color: var(--green); }
  .negative { color: var(--red); }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; margin: 2px; font-size: 0.85em; }
  .tag.buy { background: rgba(63, 185, 80, 0.15); color: var(--green); border: 1px solid rgba(63, 185, 80, 0.3); }
  .tag.sell { background: rgba(248, 81, 73, 0.15); color: var(--red); border: 1px solid rgba(248, 81, 73, 0.3); }
  .tag.hold { background: rgba(210, 153, 34, 0.15); color: var(--orange); border: 1px solid rgba(210, 153, 34, 0.3); }
  .tag.neutral { background: rgba(139, 148, 158, 0.1); color: var(--text2); border: 1px solid var(--border); }
  .signal-row { min-height: 32px; display: flex; align-items: center; flex-wrap: wrap; gap: 4px; margin-bottom: 4px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
  th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--border); }
  th { color: var(--text2); font-weight: normal; }
  .bar-container { display: flex; align-items: center; gap: 8px; }
  .bar { height: 6px; border-radius: 3px; }
  .bar.win { background: var(--green); }
  .bar.loss { background: var(--red); }
  .date-nav { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
  .tab-btn { background: var(--surface); border: 1px solid var(--border); color: var(--text2); padding: 2px 10px;
    border-radius: 4px; cursor: pointer; font-family: inherit; font-size: 0.85em; margin-right: 4px; }
  .tab-btn:hover { border-color: var(--blue); color: var(--blue); }
  .tab-btn.active { border-color: var(--blue); color: var(--blue); background: rgba(88, 166, 255, 0.1); }
  .date-btn { background: var(--surface); border: 1px solid var(--border); color: var(--text2); padding: 4px 10px;
    border-radius: 4px; cursor: pointer; font-family: inherit; font-size: 0.85em; }
  .date-btn:hover { border-color: var(--blue); color: var(--blue); }
  .date-btn.active { border-color: var(--blue); color: var(--blue); background: rgba(88, 166, 255, 0.1); }
  #status { font-size: 0.8em; color: var(--text2); margin-bottom: 12px; }
  .futu-badge { font-size: 0.75em; padding: 2px 6px; border-radius: 3px; }
  .futu-badge.on { background: rgba(63, 185, 80, 0.2); color: var(--green); }
  .futu-badge.off { background: rgba(248, 81, 73, 0.2); color: var(--red); }
  .btn { padding: 8px 20px; border-radius: 6px; border: none; cursor: pointer; font-family: inherit;
    font-size: 0.9em; font-weight: bold; transition: opacity 0.2s; }
  .btn:hover { opacity: 0.85; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-confirm { background: var(--green); color: var(--bg); }
  .btn-cancel { background: var(--border); color: var(--text2); margin-left: 8px; }
  .btn-preview { background: var(--blue); color: var(--bg); }
  .order-row { display: grid; grid-template-columns: 60px 80px 110px 50px 60px 90px 80px 80px; gap: 8px; align-items: center;
    padding: 6px 0; border-bottom: 1px solid var(--border); font-size: 0.9em; }
  .order-row:last-child { border-bottom: none; }
  .order-header { color: var(--text2); font-size: 0.8em; }
  .env-badge { font-size: 0.7em; padding: 2px 6px; border-radius: 3px; margin-left: 8px; }
  .env-simulate { background: rgba(88, 166, 255, 0.2); color: var(--blue); }
  .env-real { background: rgba(248, 81, 73, 0.3); color: var(--red); }
</style>
</head>
<body>

<h1>Stock Indicator <span style="color: var(--text2); font-weight: normal">Dashboard</span> <span style="font-size:0.5em; color:var(--text2); font-weight:normal">since 2026-04-21</span></h1>
<div id="status">Loading...</div>

<div class="grid">
  <!-- Adaptive TP/SL -->
  <div class="card" id="adaptive-card">
    <h2>Adaptive TP/SL</h2>
    <div id="adaptive-stats"></div>
  </div>

  <!-- Account -->
  <div class="card" id="account-card">
    <h2>Account <span class="futu-badge off" id="futu-badge">FUTU</span></h2>
    <div id="account-stats"></div>
  </div>

  <!-- Today's Signals -->
  <div class="card full" id="signals-card">
    <h2>Signals — <span id="signal-date"></span></h2>
    <div id="signals-content"></div>
  </div>

  <!-- Order Preview -->
  <div class="card full" id="orders-card">
    <h2>Order Preview <span class="env-badge" id="env-badge"></span></h2>
    <div id="orders-content"><div style="color:var(--text2)">Click "Preview Orders" to load</div></div>
    <div style="margin-top: 12px">
      <button class="btn btn-preview" onclick="previewOrders()">Preview Orders</button>
      <button class="btn btn-confirm" id="confirm-btn" onclick="executeOrders()" disabled>Confirm &amp; Send</button>
      <button class="btn btn-cancel" id="cancel-btn" onclick="cancelOrders()" style="display:none">Cancel</button>
      <button class="btn" style="background:var(--orange); color:var(--bg); margin-left:16px" onclick="placeTPSL()">Place TP/SL</button>
    </div>
  </div>

  <!-- Positions -->
  <div class="card full" id="positions-card">
    <h2>Positions</h2>
    <div id="positions-content"></div>
  </div>

  <!-- Date navigation -->
  <div class="card full">
    <h2>Log History</h2>
    <div class="date-nav" id="date-nav"></div>
  </div>

  <!-- Rolling Trade History -->
  <div class="card full" id="trades-card">
    <h2>Rolling Trade History
      <span style="font-size:0.7em; font-weight:normal; margin-left:8px">
        <button class="tab-btn active" data-tab="all" onclick="setTradeTab('all')">All</button>
        <button class="tab-btn" data-tab="winners" onclick="setTradeTab('winners')">Winners</button>
        <button class="tab-btn" data-tab="losers" onclick="setTradeTab('losers')">Losers</button>
      </span>
    </h2>
    <div id="trades-content"></div>
  </div>
</div>

<script>
const $ = (sel) => document.querySelector(sel);

function pct(v, decimals=2) {
  return v != null ? v.toFixed(decimals) + '%' : '—';
}

function plClass(v) {
  return v > 0 ? 'positive' : v < 0 ? 'negative' : '';
}

function stat(label, value, cls='') {
  return `<div class="stat"><span class="label">${label}</span><span class="value ${cls}">${value}</span></div>`;
}

function tag(text, type) {
  return `<span class="tag ${type}">${text}</span>`;
}

// Rolling Trade History — cached trade list + tab filter.
let __all_trades_cache = [];
let __trade_tab = 'all';
function setTradeTab(tab) {
  __trade_tab = tab;
  document.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.tab === tab);
  });
  renderTrades();
}
function renderTrades() {
  let trades = (__all_trades_cache || []).slice();
  if (__trade_tab === 'winners') {
    trades = trades.filter(t => t.raw_pct != null && t.raw_pct > 0);
  } else if (__trade_tab === 'losers') {
    trades = trades.filter(t => t.raw_pct != null && t.raw_pct < 0);
  }
  if (trades.length === 0) {
    $('#trades-content').innerHTML = '<div style="color:var(--text2)">No trade history</div>';
    return;
  }
  let html = '<table><tr><th>Symbol</th><th>Bucket</th><th>Entry</th><th>Exit</th><th>P/L %</th><th></th></tr>';
  for (const t of [...trades].reverse()) {
    const rawPct = t.raw_pct != null ? (t.raw_pct * 100) : null;
    const barWidth = rawPct != null ? Math.min(Math.abs(rawPct) * 8, 120) : 0;
    const barType = rawPct != null && rawPct >= 0 ? 'win' : 'loss';
    const bucketShort = (t.bucket || '').replace('_production', '').replace('_explore', '') || '—';
    html += `<tr>
      <td><strong>${t.symbol}</strong></td>
      <td style="color:var(--text2)">${bucketShort}</td>
      <td>${t.entry_date||'—'}</td>
      <td>${t.exit_date||'—'}</td>
      <td class="${plClass(rawPct)}">${rawPct != null ? (rawPct >= 0 ? '+' : '') + rawPct.toFixed(2) + '%' : '—'}</td>
      <td><div class="bar-container"><div class="bar ${barType}" style="width:${barWidth}px"></div></div></td>
    </tr>`;
  }
  html += '</table>';
  $('#trades-content').innerHTML = html;
}

// Build symbol -> bucket short-name map from frozen_entries (today's
// accepts) + adaptive_state.accepted_entries (held positions). The
// short name strips _production / _explore suffix so the tag is
// readable inline.
function buildBucketMap(log, adaptive) {
  const m = {};
  const shorten = (b) => (b || '').replace('_production', '').replace('_explore', '');
  for (const e of (log.entry_signal_records || [])) {
    if (e.symbol && e.bucket) m[e.symbol] = shorten(e.bucket);
  }
  for (const e of (log.frozen_entries || [])) {
    if (e.symbol && e.bucket) m[e.symbol] = shorten(e.bucket);
  }
  for (const e of (adaptive.accepted_entries || [])) {
    if (e.symbol && e.bucket && !(e.symbol in m)) m[e.symbol] = shorten(e.bucket);
  }
  return m;
}

function tagWithBucket(symbol, type, bucketMap) {
  const b = bucketMap[symbol];
  const suffix = b ? ` <span style="opacity:0.7; font-size:0.8em">[${b}]</span>` : '';
  return `<span class="tag ${type}">${symbol}${suffix}</span>`;
}

async function load() {
  try {
    const [stateRes, futuRes] = await Promise.all([
      fetch('/api/state'),
      fetch('/api/futu/positions'),
    ]);
    const state = await stateRes.json();
    const futu = await futuRes.json();
    render(state, futu);
    $('#status').textContent = 'Last updated: ' + new Date().toLocaleTimeString();
  } catch (e) {
    $('#status').textContent = 'Error: ' + e.message;
  }
}

function render(state, futu) {
  const log = state.latest_log || {};
  const adaptive = state.adaptive_state || {};
  const positions = state.signal_trades || {};

  // Per-bucket TP / SL summary comes from current config + rolling state.
  // [FROZEN_TP_SL] remains the order-preview source because those values
  // are entry-time trade constants; using them for this summary would keep
  // showing stale values after a config/code fix until a new accepted entry
  // appears in each bucket.
  let html = '';
  const currentBucketTpSl = state.current_bucket_tp_sl || [];
  if (currentBucketTpSl.length === 0) {
    html += stat('TP / SL', 'not available', '');
  } else {
    for (const e of currentBucketTpSl) {
      const b = e.bucket || '';
      const short = b.replace('_production', '').replace('_explore', '');
      const tp_v = e.tp_pct != null ? '+' + (e.tp_pct * 100).toFixed(2) + '%' : '—';
      const sl_v = e.sl_pct != null ? '-' + (e.sl_pct * 100).toFixed(2) + '%' : '—';
      html += stat(short + ' TP', tp_v, 'positive');
      html += stat(short + ' SL', sl_v, 'negative');
    }
    html += stat('TP source', 'current config base');
  }
  // Pool size: winners + losers deque length (these feed the frozen
  // TP/SL formula). closed_trades carries the full symbol/date/pct
  // records and drives the table view below.
  const winners = (adaptive.winners || []);
  const losers = (adaptive.losers || []);
  html += stat('Pool', `${winners.length} winners + ${losers.length} losers`);
  $('#adaptive-stats').innerHTML = html;

  // Account
  html = '';
  if (futu.connected) {
    $('#futu-badge').className = 'futu-badge on';
    $('#futu-badge').textContent = 'FUTU LIVE';
    const a = futu.account;
    html += stat('Total Assets', 'HK$' + (a.total_assets||0).toLocaleString(undefined,{minimumFractionDigits:2}));
    html += stat('US Cash', 'US$' + (a.us_cash||0).toLocaleString(undefined,{minimumFractionDigits:2}));
    html += stat('Market Value', 'HK$' + (a.market_val||0).toLocaleString(undefined,{minimumFractionDigits:2}));
    html += stat('Buying Power', 'HK$' + (a.power||0).toLocaleString(undefined,{minimumFractionDigits:2}));
  } else {
    html += stat('Status', 'Futu OpenD not connected', 'negative');
    // Show signal-based positions
    const fish_head = positions.fish_head_vacuum_turn || [];
    html += stat('Signal Positions', fish_head.length + '/6');
  }
  $('#account-stats').innerHTML = html;

  // Signals
  $('#signal-date').textContent = log.date || '—';
  html = '';
  const bucketMap = buildBucketMap(log, adaptive);
  html += '<div class="signal-row"><strong style="color:var(--text2)">ENTRY (raw):</strong> ';
  html += (log.entry_signals && log.entry_signals.length) ? log.entry_signals.map(s => tagWithBucket(s, 'neutral', bucketMap)).join('') : '<span style="color:var(--text2)">—</span>';
  html += '</div>';
  html += '<div class="signal-row"><strong style="color:var(--text2)">BUY (accepted):</strong> ';
  html += (log.accepted_buy_actions && log.accepted_buy_actions.length) ? log.accepted_buy_actions.map(s => tagWithBucket(s, 'buy', bucketMap)).join('') : '<span style="color:var(--text2)">—</span>';
  html += '</div>';
  html += '<div class="signal-row"><strong style="color:var(--text2)">SELL:</strong> ';
  html += (log.sell_actions && log.sell_actions.length) ? log.sell_actions.map(s => tagWithBucket(s, 'sell', bucketMap)).join('') : '<span style="color:var(--text2)">—</span>';
  html += '</div>';
  html += '<div class="signal-row"><strong style="color:var(--text2)">HOLD (min_hold):</strong> ';
  html += (log.hold_blocked && log.hold_blocked.length) ? log.hold_blocked.map(s => tagWithBucket(s, 'hold', bucketMap)).join('') : '<span style="color:var(--text2)">—</span>';
  html += '</div>';
  // Entry/exit signal count
  html += '<div style="margin-top:8px; font-size:0.85em; color:var(--text2)">';
  html += 'Raw entries: ' + (log.entry_signals||[]).length + ' | ';
  html += 'Accepted buys: ' + (log.accepted_buy_actions||[]).length + ' | ';
  html += 'Rejected: ' + (((log.slot_allocation||{}).rejected)||[]).length + ' | ';
  html += 'Exit signals: ' + (log.exit_signals||[]).length + ' | ';
  html += 'Positions: ' + (log.position_count||0);
  html += '</div>';
  $('#signals-content').innerHTML = html;

  // Positions
  html = '';
  if (futu.connected && futu.positions.length > 0) {
    html += '<table><tr><th>Symbol</th><th>Qty</th><th>Cost</th><th>Price</th><th>P/L</th><th>P/L %</th></tr>';
    for (const p of futu.positions) {
      const plPct = p.pl_ratio * 100;
      html += `<tr>
        <td><strong>${p.symbol}</strong></td>
        <td>${p.qty}</td>
        <td>$${p.cost_price.toFixed(2)}</td>
        <td>$${p.market_price.toFixed(2)}</td>
        <td class="${plClass(p.unrealized_pl)}">${p.unrealized_pl >= 0 ? '+' : ''}$${p.unrealized_pl.toFixed(2)}</td>
        <td class="${plClass(plPct)}">${plPct >= 0 ? '+' : ''}${plPct.toFixed(2)}%</td>
      </tr>`;
    }
    html += '</table>';
  } else {
    const fish_head = positions.fish_head_vacuum_turn || [];
    if (fish_head.length > 0) {
      html += '<table><tr><th>Symbol</th><th>Entry Date</th></tr>';
      for (const p of fish_head) {
        html += `<tr><td><strong>${p.symbol}</strong></td><td>${p.entry_date||'—'}</td></tr>`;
      }
      html += '</table>';
    } else {
      html += '<div style="color:var(--text2)">No open positions</div>';
    }
  }
  $('#positions-content').innerHTML = html;

  // Trade history — cache full list, render via setTradeTab.
  __all_trades_cache = adaptive.closed_trades || [];
  renderTrades();

  // Date nav
  html = '';
  for (const d of (state.available_dates || []).slice(0, 20)) {
    const cls = d === log.date ? 'date-btn active' : 'date-btn';
    html += `<button class="${cls}" onclick="loadDate('${d}')">${d}</button>`;
  }
  $('#date-nav').innerHTML = html;
}

async function loadDate(d) {
  const res = await fetch('/api/log/' + d);
  const log = await res.json();
  // Update signals card only
  $('#signal-date').textContent = log.date || d;
  let html = '';
  // For historical dates, bucket map comes only from that day's
  // [ENTRY_SIGNAL] / [FROZEN_TP_SL] lines (no current accepted_entries lookup).
  const bucketMap = buildBucketMap(log, {});
  html += '<div class="signal-row"><strong style="color:var(--text2)">ENTRY (raw):</strong> ';
  html += (log.entry_signals && log.entry_signals.length) ? log.entry_signals.map(s => tagWithBucket(s, 'neutral', bucketMap)).join('') : '<span style="color:var(--text2)">—</span>';
  html += '</div>';
  html += '<div class="signal-row"><strong style="color:var(--text2)">BUY (accepted):</strong> ';
  html += (log.accepted_buy_actions && log.accepted_buy_actions.length) ? log.accepted_buy_actions.map(s => tagWithBucket(s, 'buy', bucketMap)).join('') : '<span style="color:var(--text2)">—</span>';
  html += '</div>';
  html += '<div class="signal-row"><strong style="color:var(--text2)">SELL:</strong> ';
  html += (log.sell_actions && log.sell_actions.length) ? log.sell_actions.map(s => tagWithBucket(s, 'sell', bucketMap)).join('') : '<span style="color:var(--text2)">—</span>';
  html += '</div>';
  html += '<div class="signal-row"><strong style="color:var(--text2)">HOLD (min_hold):</strong> ';
  html += (log.hold_blocked && log.hold_blocked.length) ? log.hold_blocked.map(s => tagWithBucket(s, 'hold', bucketMap)).join('') : '<span style="color:var(--text2)">—</span>';
  html += '</div>';
  html += '<div style="margin-top:8px; font-size:0.85em; color:var(--text2)">';
  html += 'Raw entries: ' + (log.entry_signals||[]).length + ' | ';
  html += 'Accepted buys: ' + (log.accepted_buy_actions||[]).length + ' | ';
  html += 'Rejected: ' + (((log.slot_allocation||{}).rejected)||[]).length + ' | ';
  html += 'Exit signals: ' + (log.exit_signals||[]).length + ' | ';
  html += 'Positions: ' + (log.position_count||0);
  html += '</div>';
  if (log.tp_pct != null) {
    let ahtml = '';
    ahtml += stat('TP', pct(log.tp_pct), 'positive');
    ahtml += stat('SL', pct(log.sl_pct), 'negative');
    if (log.rolling_mp != null) ahtml += stat('Rolling MP', '+' + pct(log.rolling_mp) + ' (n=' + log.rolling_mp_n + ')', 'positive');
    if (log.rolling_ml != null) ahtml += stat('Rolling ML', '-' + pct(log.rolling_ml) + ' (n=' + log.rolling_ml_n + ')', 'negative');
    ahtml += stat('Window', '20 trades');
    $('#adaptive-stats').innerHTML = ahtml;
  }
  $('#signals-content').innerHTML = html;
  document.querySelectorAll('.date-btn').forEach(b => {
    b.className = b.textContent === d ? 'date-btn active' : 'date-btn';
  });
}

// --- Order management ---
let pendingOrders = null;

async function previewOrders() {
  $('#orders-content').innerHTML = '<div style="color:var(--text2)">Loading...</div>';
  $('#confirm-btn').disabled = true;
  try {
    const res = await fetch('/api/preview_orders');
    const data = await res.json();
    if (data.error) {
      $('#orders-content').innerHTML = `<div class="negative">${data.error}</div>`;
      return;
    }
    pendingOrders = data;
    const env = data.trading_env || 'SIMULATE';
    const badge = $('#env-badge');
    badge.textContent = env;
    badge.className = env === 'REAL' ? 'env-badge env-real' : 'env-badge env-simulate';

    if (data.orders.length === 0) {
      $('#orders-content').innerHTML = '<div style="color:var(--text2)">No pending orders (signal date: ' + (data.signal_date||'—') + ')</div>';
      return;
    }

    const slotsFree = Math.max(0, (data.max_positions||0) - (data.held_count||0));
    let html = '<div style="font-size:0.8em; color:var(--text2); margin-bottom:8px">';
    html += 'Assets: HK$' + (data.total_assets_hkd||0).toLocaleString(undefined,{minimumFractionDigits:0});
    html += ' | Margin: ' + data.margin + 'x';
    html += ' | Futu held: ' + data.held_count + '/' + data.max_positions;
    html += ' | Slots free: <strong' + (slotsFree===0 ? ' style="color:var(--red)"' : '') + '>' + slotsFree + '</strong>';
    html += ' | Signal: ' + (data.signal_date||'—');
    const gate = data.risk_score_gate || {};
    if (gate.status === 'stop') {
      html += ' | <strong style="color:var(--red)">Risk gate STOP: ' + gate.year_month + ' score=' + gate.risk_score + '</strong>';
    }
    html += '</div>';

    html += '<div class="order-row order-header"><span>Side</span><span>Symbol</span><span>Bucket</span><span>Rank</span><span>Qty</span><span>Ref Price</span><span>TP%</span><span>SL%</span></div>';
    for (const o of data.orders) {
      const sideClass = o.side === 'BUY' ? 'positive' : 'negative';
      const bucketShort = (o.bucket || '').replace('_production', '').replace('_explore', '') || '—';
      const tpDisplay = o.tp_pct != null ? '+' + (o.tp_pct * 100).toFixed(2) + '%' : '—';
      const slDisplay = o.sl_pct != null ? '-' + (o.sl_pct * 100).toFixed(2) + '%' : '—';
      const refDisplay = (o.ref_price != null) ? '$' + o.ref_price.toFixed(2)
                       : (o.price != null) ? '$' + o.price.toFixed(2) : '—';
      const rankDisplay = (o.dollar_volume_rank != null) ? '#' + o.dollar_volume_rank : '—';
      const skipped = o.status === 'slot_full' || o.status === 'risk_score_stop';
      const rowStyle = skipped ? ' style="opacity:0.45"' : '';
      const sideLabel = skipped
        ? `${o.side} <span style="font-size:0.75em; opacity:0.8">(${o.status})</span>`
        : o.side;
      html += `<div class="order-row"${rowStyle}>
        <span class="${sideClass}"><strong>${sideLabel}</strong></span>
        <span><strong>${o.symbol}</strong></span>
        <span style="color:var(--text2)">${bucketShort}</span>
        <span style="color:var(--text2)">${rankDisplay}</span>
        <span>${o.qty}</span>
        <span style="color:var(--text2)">${refDisplay}</span>
        <span class="positive">${tpDisplay}</span>
        <span class="negative">${slDisplay}</span>
      </div>`;
    }
    $('#orders-content').innerHTML = html;
    $('#confirm-btn').disabled = false;
    $('#cancel-btn').style.display = 'inline-block';
  } catch (e) {
    $('#orders-content').innerHTML = `<div class="negative">Error: ${e.message}</div>`;
  }
}

async function executeOrders() {
  if (!pendingOrders || !pendingOrders.orders.length) return;
  const env = pendingOrders.trading_env || 'SIMULATE';
  const msg = env === 'REAL'
    ? 'SENDING REAL ORDERS. Are you sure?'
    : 'Send orders to PAPER TRADING?';
  if (!confirm(msg)) return;

  $('#confirm-btn').disabled = true;
  $('#confirm-btn').textContent = 'Sending...';

  try {
    const res = await fetch('/api/execute_orders', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({orders: pendingOrders.orders}),
    });
    const data = await res.json();

    let html = '<div style="margin-bottom:8px"><strong>Order Results:</strong></div>';
    for (const r of data.results) {
      const icon = r.status === 'sent' ? '✓' : '✗';
      const cls = r.status === 'sent' ? 'positive' : 'negative';
      html += `<div class="${cls}" style="margin:4px 0">${icon} ${r.side||r.symbol} ${r.symbol} qty=${r.qty||0} — ${r.status}`;
      if (r.order_id) html += ` (id: ${r.order_id})`;
      if (r.price) html += ` @ $${r.price}`;
      if (r.error) html += ` — ${r.error}`;
      html += '</div>';
    }
    $('#orders-content').innerHTML = html;
    pendingOrders = null;
    $('#cancel-btn').style.display = 'none';
    $('#confirm-btn').textContent = 'Confirm & Send';
    // Refresh positions
    setTimeout(load, 2000);
  } catch (e) {
    $('#orders-content').innerHTML = `<div class="negative">Error: ${e.message}</div>`;
    $('#confirm-btn').disabled = false;
    $('#confirm-btn').textContent = 'Confirm & Send';
  }
}

function cancelOrders() {
  pendingOrders = null;
  $('#orders-content').innerHTML = '<div style="color:var(--text2)">Cancelled</div>';
  $('#confirm-btn').disabled = true;
  $('#cancel-btn').style.display = 'none';
}

async function placeTPSL() {
  if (!confirm('Place TP/SL orders for current positions?')) return;
  const btn = document.querySelector('[onclick="placeTPSL()"]');
  btn.disabled = true;
  btn.textContent = 'Placing...';
  try {
    const res = await fetch('/api/place_tp_sl', {method: 'POST'});
    const data = await res.json();
    if (data.ok) {
      const lines = data.log.trim().split('\\n').map(l => `<div>${l}</div>`).join('');
      $('#orders-content').innerHTML = '<div style="margin-bottom:8px"><strong>TP/SL Result:</strong></div>' + lines;
    } else {
      $('#orders-content').innerHTML = `<div class="negative">Error: ${data.error}</div>`;
    }
    setTimeout(load, 2000);
  } catch (e) {
    $('#orders-content').innerHTML = `<div class="negative">Error: ${e.message}</div>`;
  }
  btn.disabled = false;
  btn.textContent = 'Place TP/SL';
}

load();
</script>
</body>
</html>
"""
