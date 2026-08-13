"""Web dashboard for stock indicator signals and portfolio state."""

from __future__ import annotations

import ast
import csv
import json
import logging
import os
import re
import threading
from datetime import date, datetime, timedelta
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pandas
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from stock_indicator import (
    adaptive_tp_sl_virtual_trade_history,
    daily_job,
    live_expectancy_gate,
)
from stock_indicator.futu_trade_metadata import (
    format_futu_order_remark,
    parse_futu_order_remark,
)
from stock_indicator.live_order_sizing import (
    DEFAULT_MARGIN_MULTIPLIER,
    compute_target_share_quantity,
)
from stock_indicator.pre_open_entry_scheduler import (
    record_confirmed_entry_intent,
    record_entry_submission_result,
)

LOGGER = logging.getLogger(__name__)

DATA_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "data"
STOCK_DATA_DIRECTORY = DATA_DIRECTORY / "stock_data"
LIVE_STATE_DIRECTORY = DATA_DIRECTORY / "live_state"
LOGS_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "logs"

app = FastAPI(title="Stock Indicator Dashboard")


def _is_loopback_host(host_text: str | None) -> bool:
    """Return whether a request host resolves syntactically to loopback."""
    if not host_text:
        return False
    normalized_host = host_text.strip().lower().strip("[]")
    if normalized_host == "localhost":
        return True
    try:
        return ip_address(normalized_host).is_loopback
    except ValueError:
        return False


@app.middleware("http")
async def _restrict_dashboard_to_local_requests(
    request: Request,
    call_next: Any,
) -> Any:
    """Reject remote clients and cross-origin mutation of real orders.

    # TODO: review
    Uvicorn is also bound to 127.0.0.1 in launchd.  This application-level
    check keeps the real-order endpoints local if a future launch command
    accidentally widens the bind address, while the Origin check blocks a
    remote web page from driving a local browser session.
    """
    client_host = request.client.host if request.client is not None else None
    if not _is_loopback_host(client_host):
        return JSONResponse(
            status_code=403,
            content={"detail": "Dashboard accepts local requests only"},
        )

    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        origin_text = request.headers.get("origin")
        if origin_text:
            origin_host = urlsplit(origin_text).hostname
            if not _is_loopback_host(origin_host):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Cross-origin order mutation rejected"},
                )
    return await call_next(request)


def _normalize_symbol_text(symbol_value: Any) -> str:
    """Return an uppercase ticker without a market prefix."""
    normalized_symbol = str(symbol_value or "").strip().upper()
    if normalized_symbol.startswith("US."):
        normalized_symbol = normalized_symbol[3:]
    return normalized_symbol


def _symbol_rename_map_path() -> Path:
    """Return the data file mapping strategy symbols to broker symbols."""
    return DATA_DIRECTORY / "symbol_rename_map.csv"


def _load_strategy_to_broker_symbol_map() -> dict[str, str]:
    """Load ticker-renames where local strategy data uses an old symbol.

    The strategy layer can keep using local CSV symbols such as SATS, while
    Futu positions/orders may already use a renamed broker symbol such as ECHO.
    Missing files mean no renames are configured.
    """
    rename_map_path = _symbol_rename_map_path()
    if not rename_map_path.exists():
        return {}

    strategy_to_broker_symbol: dict[str, str] = {}
    try:
        with rename_map_path.open("r", newline="", encoding="utf-8") as rename_file:
            for rename_record in csv.DictReader(rename_file):
                strategy_symbol = _normalize_symbol_text(
                    rename_record.get("strategy_symbol")
                )
                broker_symbol = _normalize_symbol_text(
                    rename_record.get("broker_symbol")
                )
                if strategy_symbol and broker_symbol:
                    strategy_to_broker_symbol[strategy_symbol] = broker_symbol
    except OSError as rename_error:
        LOGGER.warning("Failed to load symbol rename map: %s", rename_error)
        return {}
    return strategy_to_broker_symbol


def _to_broker_symbol(
    strategy_symbol: str,
    strategy_to_broker_symbol: dict[str, str],
) -> str:
    """Map a strategy/log symbol to the symbol accepted by the broker."""
    normalized_strategy_symbol = _normalize_symbol_text(strategy_symbol)
    return strategy_to_broker_symbol.get(
        normalized_strategy_symbol,
        normalized_strategy_symbol,
    )


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
                "rolling_ml",
                "slope_60",
                "near_delta",
                "fuel_drawdown",
                "entry_priority",
                "mean",
                "stop_threshold",
                "soft_threshold",
            }:
                fields[key_text] = float(raw_value_text)
            elif key_text in {
                "min_hold_tp",
                "min_hold_sl",
                "dollar_volume_rank",
                "insertion_counter",
                "max_hold",
                "winner_returns",
                "loser_returns",
                "pending_returns",
                "closed_trades",
                "open_pending",
                "fed_this_run",
                "closed_episodes",
                "expectancy_gated_trades",
                "priority_override_entries",
            }:
                fields[key_text] = int(raw_value_text)
            elif key_text in {
                "ema",
                "sma",
                "breakeven",
            }:
                fields[key_text] = float(raw_value_text)
            elif key_text in {
                "disable_sl_trigger",
                "reset_hold_on_reentry_signal",
                "wr_degrading",
                "degrading",
                "window_full",
                "gate_closed",
                "priority_override_active",
            }:
                fields[key_text] = raw_value_text.lower() == "true"
            else:
                fields[key_text] = raw_value_text
        except ValueError:
            fields[key_text] = raw_value_text
    return fields


def _parse_log(log_path: Path) -> dict[str, Any]:
    """Parse a daily multi-bucket log into structured data."""
    raw_text = log_path.read_text(encoding="utf-8")
    latest_run_start = raw_text.rfind("[multi_bucket_daily_signal mode=")
    text = raw_text[latest_run_start:] if latest_run_start >= 0 else raw_text
    result: dict[str, Any] = {"raw": raw_text, "date": log_path.stem}

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
    # These are tradable candidates, not accepted BUY orders.  Bucket identity
    # must remain attached because one symbol can be tradable in more than one
    # bucket and only the dashboard may resolve that competition.
    result["tradable_entry_candidates"] = [
        entry
        for entry in frozen_entries
        if entry.get("entry_date") == log_path.stem and entry.get("symbol")
    ]

    bucket_tp_sl_records_by_key: dict[str, dict[str, Any]] = {}
    bucket_tp_sl_key_order: list[str] = []
    for line in text.splitlines():
        if not line.startswith("[BUCKET_TP_SL]"):
            continue
        bucket_tp_sl_record = _parse_log_key_value_tokens(
            line[len("[BUCKET_TP_SL]"):]
        )
        bucket_tp_sl_key = str(
            bucket_tp_sl_record.get("bucket")
            or bucket_tp_sl_record.get("strategy_id")
            or len(bucket_tp_sl_key_order)
        )
        if bucket_tp_sl_key not in bucket_tp_sl_records_by_key:
            bucket_tp_sl_key_order.append(bucket_tp_sl_key)
        bucket_tp_sl_records_by_key[bucket_tp_sl_key] = bucket_tp_sl_record
    result["bucket_tp_sl"] = [
        bucket_tp_sl_records_by_key[bucket_tp_sl_key]
        for bucket_tp_sl_key in bucket_tp_sl_key_order
    ]

    # WR-gate sensor heartbeat (single source of truth for the cross
    # reading; emitted every cron run). Last line wins.
    wr_gate_sensor_records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith("[WR_GATE_SENSOR]"):
            continue
        wr_gate_sensor_records.append(
            _parse_log_key_value_tokens(line[len("[WR_GATE_SENSOR]"):])
        )
    result["wr_gate_sensor"] = (
        wr_gate_sensor_records[-1] if wr_gate_sensor_records else {}
    )

    expectancy_gate_sensor_records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith("[EXPECTANCY_GATE_SENSOR]"):
            continue
        expectancy_gate_sensor_records.append(
            _parse_log_key_value_tokens(
                line[len("[EXPECTANCY_GATE_SENSOR]"):]
            )
        )
    result["expectancy_gate_sensor"] = (
        expectancy_gate_sensor_records[-1]
        if expectancy_gate_sensor_records
        else {}
    )

    adaptive_tp_sl_virtual_history_state_records: list[dict[str, Any]] = []
    for line in text.splitlines():
        history_log_prefix = (
            "[ADAPTIVE_TP_SL_VIRTUAL_TRADE_HISTORY_STATE]"
        )
        if not line.startswith(history_log_prefix):
            continue
        adaptive_tp_sl_virtual_history_state_records.append(
            _parse_log_key_value_tokens(line[len(history_log_prefix):])
        )
    result["adaptive_tp_sl_virtual_trade_history_state"] = (
        adaptive_tp_sl_virtual_history_state_records[-1]
        if adaptive_tp_sl_virtual_history_state_records
        else {}
    )

    tradable_matches = re.findall(
        r"^tradable_candidates:\s*(.+)$", text, re.MULTILINE
    )
    filtered_out_matches = re.findall(
        r"^filtered_out:\s*(.+)$", text, re.MULTILINE
    )
    result["signal_candidate_summary"] = {
        "tradable": _parse_slot_allocation_pairs(tradable_matches[-1])
        if tradable_matches else [],
        "filtered_out": _parse_slot_allocation_pairs(filtered_out_matches[-1])
        if filtered_out_matches else [],
    }
    adaptive_tp_sl_virtual_history_activity_matches = re.findall(
        r"adaptive_tp_sl_virtual_open_trades_before_today=(\d+)\s+"
        r"adaptive_tp_sl_virtual_trades_closed_today=(\d+)",
        text,
        re.MULTILINE,
    )
    legacy_virtual_history_activity_matches = re.findall(
        r"max_position_count=\d+\s+"
        r"held_before_today=(\d+)\s+"
        r"same_day_closes=(\d+)",
        text,
        re.MULTILINE,
    )
    virtual_history_activity_matches = (
        adaptive_tp_sl_virtual_history_activity_matches
        or legacy_virtual_history_activity_matches
    )
    if virtual_history_activity_matches:
        open_before_today, closed_today = virtual_history_activity_matches[-1]
        result["signal_candidate_summary"].update({
            "adaptive_tp_sl_virtual_open_trades_before_today": int(
                open_before_today
            ),
            "adaptive_tp_sl_virtual_trades_closed_today": int(closed_today),
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

    # Legacy single-strategy logs called these "positions". Multi-bucket logs
    # expose the count under its actual purpose: open ADAPTIVE TP/SL virtual
    # reference trades.
    pos_m = re.search(r"Concurrent positions after entry \((\d+) total\)", text)
    if pos_m:
        result["adaptive_tp_sl_virtual_open_trade_count"] = int(pos_m.group(1))
    elif virtual_history_activity_matches:
        open_before_today, closed_today = virtual_history_activity_matches[-1]
        result["adaptive_tp_sl_virtual_open_trade_count"] = (
            int(open_before_today) - int(closed_today)
        )
    else:
        result["adaptive_tp_sl_virtual_open_trade_count"] = 0

    return result


def _parse_slot_allocation_pairs(raw_allocation_text: str) -> list[dict[str, str]]:
    """Parse cron's candidate/filter tuple summaries."""
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


def _load_current_bucket_tp_sl_from_adaptive_tp_sl_virtual_trade_history(
    adaptive_state_document: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute current bucket TP/SL from its virtual reference history.

    This diagnostic is intentionally separate from order preview. BUY orders
    still use entry-time [FROZEN_TP_SL] values from the cron log, while the
    dashboard summary should reflect current config plus the ADAPTIVE TP/SL
    virtual trade return sample after a restart or code fix rather than stale
    historical frozen lines.
    """
    try:
        from stock_indicator import multi_bucket_today, strategy

        config = multi_bucket_today.load_multi_bucket_config(
            PRODUCTION_CONFIG_PATH
        )
        if config.adaptive_tp_sl is None:
            return []
        adaptive_tp_sl_history = (
            adaptive_tp_sl_virtual_trade_history.get_adaptive_tp_sl_virtual_trade_history(
                adaptive_state_document
            )
        )
        bucket_states: list[dict[str, Any]] = []
        for bucket_label, bucket_definition in config.bucket_definitions.items():
            tp_pct, sl_pct, rolling_mp, rolling_ml = (
                strategy.compute_frozen_tp_sl_for_bucket(
                    bucket_def=bucket_definition,
                    adaptive_tp_sl=config.adaptive_tp_sl,
                    closed_winners=adaptive_tp_sl_history.get(
                        adaptive_tp_sl_virtual_trade_history.
                        ADAPTIVE_TP_SL_VIRTUAL_WINNER_RETURNS_KEY,
                        [],
                    ),
                    closed_losers=adaptive_tp_sl_history.get(
                        adaptive_tp_sl_virtual_trade_history.
                        ADAPTIVE_TP_SL_VIRTUAL_LOSER_RETURNS_KEY,
                        [],
                    ),
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
                "max_hold": bucket_definition.max_hold,
                "reset_hold_on_reentry_signal": (
                    bucket_definition.reset_hold_on_reentry_signal
                ),
                "source": "current_config_base",
            })
        return bucket_states
    except (OSError, json.JSONDecodeError, ValueError) as config_error:
        LOGGER.warning("Failed to compute current bucket TP/SL: %s", config_error)
        return []


def _build_cron_dashboard_contract() -> dict[str, Any]:
    """Describe which layer owns each trading decision shown by dashboard."""
    # TODO: review
    # This text is user-facing dashboard copy, so keep it natural rather
    # than mirroring internal file or function names.
    return {
        "title": "Cron → Dashboard contract",
        "steps": [
            {
                "owner": "Cron",
                "label": "Signal layer",
                "detail": (
                    "Finds every tradable entry/exit signal, freezes signal "
                    "TP/SL metadata, and updates the separate ADAPTIVE TP/SL "
                    "virtual trade history used only as a statistical sensor."
                ),
            },
            {
                "owner": "Dashboard",
                "label": "Order layer",
                "detail": (
                    "Reads cron log lines, reconciles them against live Futu "
                    "positions, applies real account slots and risk gate, then "
                    "previews or sends orders."
                ),
            },
            {
                "owner": "Futu",
                "label": "Live truth",
                "detail": (
                    "Live holdings and order status are the source of truth "
                    "for what can actually be bought or sold."
                ),
            },
        ],
        "notes": [
            "Raw entries are strategy signals before order-layer checks.",
            "Cron publishes every tradable signal with frozen TP/SL metadata.",
            "The ADAPTIVE TP/SL virtual trade history is statistical state, not positions.",
            "The local signal mirror is diagnostic; it is not the live portfolio.",
        ],
    }


def _load_latest_cached_market_log_date() -> date | None:
    """Return the latest cached market date used to filter dashboard logs."""

    market_cache_path = STOCK_DATA_DIRECTORY / f"{daily_job.SP500_SYMBOL}.csv"
    if not market_cache_path.exists():
        return None
    try:
        return daily_job.determine_latest_cached_market_date(STOCK_DATA_DIRECTORY)
    except Exception as market_date_error:  # noqa: BLE001
        LOGGER.warning(
            "Could not determine latest cached market date from %s: %s",
            market_cache_path,
            market_date_error,
        )
        return None


def _get_log_dates() -> list[str]:
    """Return sorted list of available log dates (newest first)."""
    latest_market_log_date = _load_latest_cached_market_log_date()
    dates = []
    for log_file_path in LOGS_DIRECTORY.glob("*.log"):
        try:
            log_date = date.fromisoformat(log_file_path.stem)
        except ValueError:
            continue
        if latest_market_log_date is not None and log_date > latest_market_log_date:
            continue
        dates.append(log_file_path.stem)
    return sorted(dates, reverse=True)


@app.get("/api/state")
def api_state():
    """Current system state: signal trades, adaptive state, latest log."""
    signal_trades = _load_json(LIVE_STATE_DIRECTORY / "signal_trades.json")
    adaptive = _load_json(LIVE_STATE_DIRECTORY / "adaptive_state.json")
    if adaptive:
        adaptive_tp_sl_virtual_trade_history.get_adaptive_tp_sl_virtual_trade_history(
            adaptive
        )
    dates = _get_log_dates()
    latest_log = None
    if dates:
        latest_log = _parse_log(LOGS_DIRECTORY / f"{dates[0]}.log")
    return {
        "signal_trades": signal_trades,
        "adaptive_state": adaptive,
        "communication_contract": _build_cron_dashboard_contract(),
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
    """ADAPTIVE TP/SL virtual trade history from adaptive_state.json."""
    adaptive_state_document = _load_json(
        LIVE_STATE_DIRECTORY / "adaptive_state.json"
    )
    adaptive_tp_sl_history = (
        adaptive_tp_sl_virtual_trade_history.get_adaptive_tp_sl_virtual_trade_history(
            adaptive_state_document
        )
    )
    return {
        "adaptive_tp_sl_virtual_raw_returns": adaptive_tp_sl_history.get(
            adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_RAW_RETURNS_KEY,
            [],
        ),
        "adaptive_tp_sl_virtual_closed_trades": adaptive_tp_sl_history.get(
            adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_CLOSED_TRADES_KEY,
            [],
        ),
    }


@app.get("/api/futu/positions")
def api_futu_positions():
    """Live positions from Futu OpenD (if connected)."""
    try:
        trade_context = _get_futu_trd_ctx()
        try:
            trading_environment = _get_trd_env()
            position_return_code, position_data = (
                trade_context.position_list_query(trd_env=trading_environment)
            )
            account_return_code, account_data = trade_context.accinfo_query(
                trd_env=trading_environment
            )
            position_entries_by_symbol: dict[str, dict[str, Any]] = {}
            if position_return_code == 0 and len(position_data) > 0:
                try:
                    position_entries_by_symbol = _load_futu_open_trade_entries(
                        trade_context,
                        trading_environment,
                        signal_date_text=date.today().isoformat(),
                    )
                except Exception as position_entry_error:  # noqa: BLE001
                    LOGGER.warning(
                        "Failed to load Futu position bucket metadata: %s",
                        position_entry_error,
                    )
        finally:
            trade_context.close()

        positions = []
        if position_return_code == 0 and len(position_data) > 0:
            for _position_index, position_row in position_data.iterrows():
                symbol = _normalize_symbol_text(position_row.get("code", ""))
                entry_metadata = position_entries_by_symbol.get(symbol, {})
                positions.append({
                    "symbol": symbol,
                    "bucket": entry_metadata.get("bucket"),
                    "strategy_id": entry_metadata.get("strategy_id"),
                    "entry_date": entry_metadata.get("entry_date"),
                    "qty": float(position_row.get("qty", 0)),
                    "cost_price": float(position_row.get("cost_price", 0)),
                    "market_price": float(position_row.get("nominal_price", 0)),
                    "market_val": float(position_row.get("market_val", 0)),
                    "unrealized_pl": float(position_row.get("unrealized_pl", 0)),
                    "pl_ratio": float(position_row.get("pl_ratio", 0)),
                })

        account = {}
        if account_return_code == 0 and len(account_data) > 0:
            row = account_data.iloc[0]
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

MARGIN_MULTIPLIER = DEFAULT_MARGIN_MULTIPLIER
# TODO: review
# Manual launches are safe by default. The installed production LaunchAgent
# explicitly sets STOCK_INDICATOR_TRADING_ENV=REAL.
TRADING_ENV = os.environ.get(
    "STOCK_INDICATOR_TRADING_ENV",
    "SIMULATE",
).strip().upper()
ORDER_LOG_DIR = LOGS_DIRECTORY
ORDER_EXECUTION_LOCK = threading.Lock()

PRODUCTION_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "multi_bucket_production.json"
)
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
PREVIEW_SKIP_STATUSES = {
    "bucket_cap",
    "price_unavailable",
    "slot_full",
    "risk_score_stop",
    "min_hold_block",
    "wr_gate_phantom",
    "expectancy_gate_phantom",
}
TERMINAL_BROKER_ORDER_STATUSES = {
    "CANCELLED_PART",
    "CANCELLED_ALL",
    "FILL_CANCELLED",
    "DISABLED",
    "FILLED_ALL",
    "FAILED",
    "SUBMIT_FAILED",
    "TIMEOUT",
    "DELETED",
}

# Zero-capital phantom positions tracked by the ORDER LAYER (this module),
# separate from cron's adaptive_state.json. A phantom is a degrading-regime
# WR entry, an expectancy-stop entry, or both. It wins and occupies one slot
# while never placing a real Futu order, then releases the slot at its final
# adaptive exit.
PHANTOM_POSITIONS_PATH = LIVE_STATE_DIRECTORY / "phantom_positions.json"
EXPECTANCY_GATE_STATE_PATH = (
    LIVE_STATE_DIRECTORY / "expectancy_gate_state.json"
)
ADAPTIVE_STATE_PATH = LIVE_STATE_DIRECTORY / "adaptive_state.json"
# Production data_source is "daily" -> data/stock_data. Phantom exit
# detection replays the same per-symbol daily CSVs the cron sensor uses.
PRODUCTION_DATA_DIRECTORY = DATA_DIRECTORY / "stock_data"


def _load_phantom_positions() -> list[dict[str, Any]]:
    """Load the order layer's phantom-position list. Missing/corrupt file
    yields an empty list (no phantom slots held) — the safe default."""
    if not PHANTOM_POSITIONS_PATH.exists():
        return []
    try:
        with PHANTOM_POSITIONS_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _save_phantom_positions(positions: list[dict[str, Any]]) -> None:
    """Persist the phantom list atomically (tmp + replace) so a crash
    mid-write cannot corrupt slot accounting."""
    import os as _os

    PHANTOM_POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = PHANTOM_POSITIONS_PATH.with_suffix(
        PHANTOM_POSITIONS_PATH.suffix + ".tmp"
    )
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(positions, handle, indent=2)
    _os.replace(temp_path, PHANTOM_POSITIONS_PATH)


# TODO: review
def _load_wr_gate_order_config() -> dict[str, Any]:
    """Return the production WR-gate's order-layer activation settings.

    Omitting ``risk_score_activation_threshold`` makes the configured WR gate
    sensor-only: a degrading sensor directly creates a phantom.  The optional
    threshold remains supported for historical and research configurations.
    """

    default_config: dict[str, Any] = {
        "configured": False,
        "activation_mode": "off",
        "risk_score_activation_threshold": None,
    }
    try:
        raw_gate = _load_production_config().get("ft_family_wr_gate")
    except (OSError, json.JSONDecodeError, ValueError):
        return default_config
    if not isinstance(raw_gate, dict):
        return default_config
    threshold = raw_gate.get("risk_score_activation_threshold")
    if threshold is None:
        return {
            "configured": True,
            "activation_mode": "sensor_only",
            "risk_score_activation_threshold": None,
        }
    try:
        activation_threshold = int(threshold)
    except (TypeError, ValueError):
        return default_config
    return {
        "configured": True,
        "activation_mode": "risk_score",
        "risk_score_activation_threshold": activation_threshold,
    }


def _phantom_still_open(
    phantom: dict[str, Any], eval_date_string: str
) -> bool:
    """Return whether a phantom position is still holding its slot as of
    eval_date. Lazily resolves the fill date / entry price (signal+1 open)
    in place — mirroring the cron's wr_gate_pending_ft lazy fill — then
    replays the final adaptive outcome. TP/max-hold and signal exits release
    the slot only on a strictly later evaluation date, matching the
    no-lookahead rule used by both live sensors. Missing state or price data
    keeps the slot held (fail closed)."""

    if not eval_date_string:
        return True
    adaptive_state_document = _load_json(ADAPTIVE_STATE_PATH)
    try:
        adaptive_history_state = (
            adaptive_tp_sl_virtual_trade_history.
            get_adaptive_tp_sl_virtual_trade_history(
                adaptive_state_document
            )
        )
        closed_trade_by_identity = (
            live_expectancy_gate.build_closed_signal_trade_index(
                adaptive_history_state
            )
        )
        reentry_signal_dates_by_symbol = (
            live_expectancy_gate.build_reentry_signal_dates_by_symbol(
                adaptive_history_state
            )
        )
        resolved_outcome = (
            live_expectancy_gate.resolve_trade_outcome_before_date(
                pending_trade=phantom,
                evaluation_date=pandas.Timestamp(eval_date_string),
                data_directory=PRODUCTION_DATA_DIRECTORY,
                closed_trade_by_identity=closed_trade_by_identity,
                reentry_signal_dates_by_symbol=(
                    reentry_signal_dates_by_symbol
                ),
            )
        )
    except (KeyError, TypeError, ValueError):
        return True
    return resolved_outcome is None

# TODO: review

def _load_production_config() -> dict[str, Any]:
    """Read the live multi-bucket config without caching it."""
    with PRODUCTION_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config_document = json.load(config_file)
    if not isinstance(config_document, dict):
        raise ValueError("production config root must be a JSON object")
    return config_document


def _load_expectancy_gate_decision(
    parsed_log: dict[str, Any],
) -> dict[str, Any]:
    """Validate and return the cron-owned once-per-day gate decision.

    An enabled gate never guesses from missing, malformed, stale, or
    configuration-mismatched sensor output.  Such cases return ``error`` so
    the preview cannot submit BUY orders under an unknown regime.
    """

    default_state: dict[str, Any] = {
        "enabled": False,
        "status": "off",
        "gate_closed": False,
        "priority_override_active": False,
        "reason": "expectancy_gate not configured",
    }
    try:
        config_document = _load_production_config()
        gate_config = live_expectancy_gate.parse_live_expectancy_gate_config(
            config_document
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as config_error:
        return {
            **default_state,
            "enabled": True,
            "status": "error",
            "reason": f"invalid expectancy gate config: {config_error}",
        }
    if gate_config is None:
        return default_state

    try:
        persisted_state = live_expectancy_gate.load_expectancy_gate_state(
            EXPECTANCY_GATE_STATE_PATH,
            gate_config,
            allow_missing=False,
        )
    except (OSError, TypeError, ValueError) as state_error:
        return {
            **default_state,
            "enabled": True,
            "status": "error",
            "reason": f"invalid expectancy gate state: {state_error}",
        }
    persisted_sensor_state = (
        live_expectancy_gate.build_expectancy_gate_sensor_state(
            persisted_state, gate_config
        )
    )
    if str(persisted_state.get("last_evaluation_date") or "") != str(
        parsed_log.get("date") or ""
    ):
        return {
            **default_state,
            "enabled": True,
            "status": "error",
            "reason": "expectancy gate live state date does not match log date",
        }

    sensor_state = parsed_log.get("expectancy_gate_sensor")
    if not isinstance(sensor_state, dict) or not sensor_state:
        return {
            **default_state,
            "enabled": True,
            "status": "error",
            "reason": "enabled expectancy gate has no sensor heartbeat",
        }
    if sensor_state.get("status") != "ready":
        return {
            **default_state,
            "enabled": True,
            "status": "error",
            "reason": "expectancy gate sensor heartbeat is not ready",
        }

    try:
        window_count_text, window_size_text = str(
            sensor_state["window"]
        ).split("/", maxsplit=1)
        window_count = int(window_count_text)
        window_size = int(window_size_text)
        stop_threshold = float(sensor_state["stop_threshold"])
        soft_threshold_value = sensor_state.get("soft_threshold")
        soft_threshold = (
            float(soft_threshold_value)
            if soft_threshold_value is not None
            else None
        )
        gate_closed = sensor_state["gate_closed"]
        priority_override_active = sensor_state[
            "priority_override_active"
        ]
        if not isinstance(gate_closed, bool) or not isinstance(
            priority_override_active, bool
        ):
            raise ValueError("decision fields must be booleans")
    except (KeyError, TypeError, ValueError) as sensor_error:
        return {
            **default_state,
            "enabled": True,
            "status": "error",
            "reason": f"invalid expectancy gate heartbeat: {sensor_error}",
        }

    expected_soft_threshold = gate_config.priority_override_threshold
    threshold_tolerance = 0.0000006
    threshold_mismatch = abs(stop_threshold - gate_config.threshold) > (
        threshold_tolerance
    )
    soft_threshold_mismatch = (
        (soft_threshold is None) != (expected_soft_threshold is None)
        or (
            soft_threshold is not None
            and expected_soft_threshold is not None
            and abs(soft_threshold - expected_soft_threshold)
            > threshold_tolerance
        )
    )
    window_full = bool(sensor_state.get("window_full"))
    if (
        window_size != gate_config.window
        or window_count < 0
        or window_count > window_size
        or window_full != (window_count >= window_size)
        or threshold_mismatch
        or soft_threshold_mismatch
        or not live_expectancy_gate.is_finite_sensor_state(sensor_state)
    ):
        return {
            **default_state,
            "enabled": True,
            "status": "error",
            "reason": (
                "expectancy gate heartbeat does not match production config"
            ),
        }
    if not window_full and priority_override_active:
        return {
            **default_state,
            "enabled": True,
            "status": "error",
            "reason": "expectancy priority override active before full window",
        }

    persisted_rolling_mean = persisted_sensor_state.get("rolling_mean")
    heartbeat_rolling_mean = sensor_state.get("mean")
    rolling_mean_mismatch = (
        (persisted_rolling_mean is None) != (heartbeat_rolling_mean is None)
        or (
            persisted_rolling_mean is not None
            and heartbeat_rolling_mean is not None
            and abs(
                float(persisted_rolling_mean)
                - float(heartbeat_rolling_mean)
            )
            > threshold_tolerance
        )
    )
    if (
        persisted_sensor_state["count"] != window_count
        or persisted_sensor_state["gate_closed"] != gate_closed
        or persisted_sensor_state["priority_override_active"]
        != priority_override_active
        or rolling_mean_mismatch
    ):
        return {
            **default_state,
            "enabled": True,
            "status": "error",
            "reason": "expectancy gate heartbeat does not match live state",
        }

    return {
        **sensor_state,
        "enabled": True,
        "status": "closed" if gate_closed else "open",
        "cold_start": gate_config.cold_start,
        "gate_closed": gate_closed,
        "priority_override_active": priority_override_active,
        # Accepted allocations may be appended after the once-per-day cron
        # decision. Surface current ledger counters without recomputing or
        # changing that frozen decision.
        "open_pending_at_evaluation": sensor_state.get("open_pending"),
        "open_pending": persisted_sensor_state["open_pending"],
        "closed_episodes": persisted_sensor_state["closed_episodes"],
        "expectancy_gated_trades": persisted_sensor_state[
            "expectancy_gated_trades"
        ],
        "priority_override_entries": persisted_sensor_state[
            "priority_override_entries"
        ],
        "reason": (
            f"warming up; cold_start={gate_config.cold_start}"
            if not window_full
            else (
                "rolling mean below stop threshold"
                if gate_closed
                else "rolling mean at or above stop threshold"
            )
        ),
    }


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


# TODO: review
def _load_current_bucket_entry_limits(
    risk_score_value: int | None = None,
    *,
    expectancy_priority_override_active: bool = False,
) -> dict[str, dict[str, Any]]:
    """Load effective live priorities and caps for portfolio allocation.

    Risk-score priority overrides are evaluated here, in the order layer,
    rather than trusted from cron emission order.  This keeps the single live
    competition pass dependent on the current production JSON and the same
    risk score that gates today's orders.
    """
    try:
        from stock_indicator import multi_bucket_today

        config = multi_bucket_today.load_multi_bucket_config(
            PRODUCTION_CONFIG_PATH
        )
    except (OSError, json.JSONDecodeError, ValueError) as config_error:
        LOGGER.warning(
            "Failed to load current bucket entry limits: %s", config_error
        )
        return {}

    priority_overrides: dict[str, int] = {}
    raw_priority_overrides = config.raw_document.get(
        "risk_score_priority_overrides"
    )
    if isinstance(raw_priority_overrides, dict) and risk_score_value is not None:
        try:
            override_scores = {
                int(raw_score)
                for raw_score in raw_priority_overrides.get("scores", [])
            }
        except (TypeError, ValueError):
            override_scores = set()
        raw_priorities = raw_priority_overrides.get("priorities")
        if (
            int(risk_score_value) in override_scores
            and isinstance(raw_priorities, dict)
        ):
            for bucket_label, raw_priority in raw_priorities.items():
                try:
                    priority_overrides[str(bucket_label)] = int(raw_priority)
                except (TypeError, ValueError):
                    LOGGER.warning(
                        "Ignoring invalid priority override %r=%r",
                        bucket_label,
                        raw_priority,
                    )

    expectancy_priority_overrides: dict[str, int] = {}
    if (
        expectancy_priority_override_active
        and config.expectancy_gate is not None
        and config.expectancy_gate.priority_override is not None
    ):
        expectancy_priority_overrides = dict(
            config.expectancy_gate.priority_override.priorities
        )

    limits_by_key: dict[str, dict[str, Any]] = {}
    for bucket_label, bucket_definition in config.bucket_definitions.items():
        maximum_positions = (
            bucket_definition.maximum_positions
            if bucket_definition.maximum_positions is not None
            else config.maximum_position_count
        )
        limit = {
            "bucket": bucket_label,
            "strategy_id": bucket_definition.strategy_identifier,
            "entry_priority": expectancy_priority_overrides.get(
                bucket_label,
                priority_overrides.get(
                    bucket_label,
                    bucket_definition.entry_priority,
                ),
            ),
            "maximum_positions": maximum_positions,
            "fill_remaining": bucket_definition.fill_remaining,
            "max_same_symbol": config.max_same_symbol,
            "fuel_priority_threshold": (
                bucket_definition.fuel_priority_threshold
            ),
        }
        limits_by_key[bucket_label] = limit
        if bucket_definition.strategy_identifier:
            limits_by_key.setdefault(
                bucket_definition.strategy_identifier, limit
            )
    return limits_by_key


def _load_current_max_same_symbol(
    current_bucket_entry_limits: dict[str, dict[str, Any]],
) -> int:
    """Return the configured same-symbol cap from loaded allocation rules."""
    for entry_limit in current_bucket_entry_limits.values():
        try:
            return int(entry_limit.get("max_same_symbol", 1))
        except (TypeError, ValueError):
            continue
    return 1


def _count_live_positions_by_bucket(
    *,
    held_symbols: set[str],
    futu_open_entries: dict[str, dict[str, Any]],
    open_phantom_positions: list[dict[str, Any]],
    pending_buy_entries: dict[str, dict[str, Any]] | None = None,
) -> dict[str, int]:
    """Count current broker, pending BUY, and phantom slots by bucket."""
    bucket_counts: dict[str, int] = {}
    for symbol in held_symbols:
        bucket_label = futu_open_entries.get(symbol, {}).get("bucket")
        if not bucket_label:
            continue
        bucket_key = str(bucket_label)
        bucket_counts[bucket_key] = bucket_counts.get(bucket_key, 0) + 1

    for phantom_position in open_phantom_positions:
        bucket_label = phantom_position.get("bucket")
        if not bucket_label:
            continue
        bucket_key = str(bucket_label)
        bucket_counts[bucket_key] = bucket_counts.get(bucket_key, 0) + 1

    for pending_symbol, pending_entry in (pending_buy_entries or {}).items():
        if pending_symbol in held_symbols:
            continue
        bucket_label = pending_entry.get("bucket")
        if not bucket_label:
            continue
        bucket_key = str(bucket_label)
        bucket_counts[bucket_key] = bucket_counts.get(bucket_key, 0) + 1
    return bucket_counts


def _entry_limit_skip_reason(
    *,
    entry_limit: dict[str, Any] | None,
    bucket_position_counts: dict[str, int],
    occupied_position_count: int,
) -> str | None:
    """Return a skip reason when a BUY would violate its bucket cap."""
    if not entry_limit:
        return None
    bucket_label = str(entry_limit.get("bucket") or "")
    if not bucket_label:
        return None
    try:
        maximum_positions = int(entry_limit.get("maximum_positions"))
    except (TypeError, ValueError):
        return None

    if bool(entry_limit.get("fill_remaining")):
        if occupied_position_count >= maximum_positions:
            return (
                f"fill_remaining bucket {bucket_label} requires total "
                f"positions < {maximum_positions} "
                f"(current: {occupied_position_count})"
            )
        return None

    current_bucket_positions = bucket_position_counts.get(bucket_label, 0)
    if current_bucket_positions >= maximum_positions:
        return (
            f"bucket {bucket_label} max_positions={maximum_positions} "
            f"already filled (current: {current_bucket_positions})"
        )
    return None


def _tradable_candidate_sort_key(
    candidate_record: dict[str, Any],
    current_bucket_entry_limits: dict[str, dict[str, Any]],
) -> tuple[int, float, int]:
    """Return the configured greedy competition key for one candidate."""
    bucket_key = str(
        candidate_record.get("bucket")
        or candidate_record.get("strategy_id")
        or ""
    )
    entry_limit = current_bucket_entry_limits.get(bucket_key, {})
    try:
        entry_priority = int(entry_limit.get("entry_priority", 0))
    except (TypeError, ValueError):
        entry_priority = 0
    fuel_priority_threshold = entry_limit.get("fuel_priority_threshold")
    fuel_drawdown = candidate_record.get("fuel_drawdown")
    try:
        if (
            fuel_priority_threshold is not None
            and fuel_drawdown is not None
            and float(fuel_drawdown) <= float(fuel_priority_threshold)
        ):
            entry_priority -= 1
    except (TypeError, ValueError):
        pass
    try:
        within_bucket_priority = float(
            candidate_record.get(
                "entry_priority",
                candidate_record.get("dollar_volume_rank", 2**31 - 1),
            )
        )
    except (TypeError, ValueError):
        within_bucket_priority = float(2**31 - 1)
    try:
        insertion_counter = int(
            candidate_record.get("insertion_counter", 2**31 - 1)
        )
    except (TypeError, ValueError):
        insertion_counter = 2**31 - 1
    return (
        entry_priority,
        within_bucket_priority,
        insertion_counter,
    )


def _load_production_min_hold() -> int:
    """Read signal-exit `min_hold` (weekday bars) from the production config.

    Returns 0 when the field is absent or unreadable so a missing key never
    silently blocks SELL orders.
    """
    try:
        return max(0, int(_load_production_config().get("min_hold", 0)))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0


def _resolve_config_path(path_text: str) -> Path:
    """Resolve a config-relative path from the repository root."""
    resolved_path = Path(path_text).expanduser()
    if resolved_path.is_absolute():
        return resolved_path
    return REPOSITORY_ROOT / resolved_path


def _load_risk_score_gate_state(signal_date: str | None) -> dict[str, Any]:
    """Return the dashboard order gate state for a signal date.

    The risk-score gate belongs to the dashboard/order layer: cron should
    keep emitting signals and updating the ADAPTIVE TP/SL virtual trade
    history, while this function decides whether BUY orders are allowed to
    leave the dashboard.
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


def _load_pending_buy_order_entries(
    order_data: Any,
) -> dict[str, dict[str, Any]]:
    """Return non-terminal BUY metadata keyed by broker symbol."""
    pending_buy_entries: dict[str, dict[str, Any]] = {}
    if order_data is None or len(order_data) == 0:
        return pending_buy_entries
    for _order_index, order_row in order_data.iterrows():
        if "BUY" not in str(order_row.get("trd_side", "")).upper():
            continue
        order_status = str(order_row.get("order_status", "")).upper()
        if order_status in TERMINAL_BROKER_ORDER_STATUSES:
            continue
        broker_symbol = _normalize_symbol_text(order_row.get("code", ""))
        if broker_symbol:
            remark_text = str(
                order_row.get("remark", "")
                or order_row.get("order_remark", "")
            )
            pending_buy_entries[broker_symbol] = {
                "symbol": broker_symbol,
                **_parse_dashboard_order_remark(remark_text),
            }
    return pending_buy_entries


def _load_pending_buy_order_symbols(order_data: Any) -> set[str]:
    """Return broker symbols with a non-terminal BUY already submitted."""
    return set(_load_pending_buy_order_entries(order_data))


def _compute_order_size(
    total_assets_hkd: float, price_usd: float, max_positions: int
) -> int:
    """Compute qty: floor(total_assets * margin / max_pos / price).

    total_assets is in HKD, price is in USD.  Uses ~7.8 HKD/USD.
    """
    return compute_target_share_quantity(
        total_assets_hkd=total_assets_hkd,
        price_usd=price_usd,
        maximum_position_count=max_positions,
        margin_multiplier=MARGIN_MULTIPLIER,
    )


def _count_weekday_bars_held(entry_date_text: str, signal_date_text: str) -> int:
    """Count weekday trading bars held from entry date through signal date."""
    entry_date = date.fromisoformat(entry_date_text)
    signal_date = date.fromisoformat(signal_date_text)
    if signal_date < entry_date:
        return 0

    bars_held = 0
    current_date = entry_date
    while current_date <= signal_date:
        if current_date.weekday() < 5:
            bars_held += 1
        current_date += timedelta(days=1)
    return bars_held


def _load_us_trading_days(
    start_date_text: str, end_date_text: str
) -> list[str] | None:
    """Return US trading days in [start, end] as sorted 'YYYY-MM-DD' strings.

    Uses Futu's market calendar so holidays/half-day closures are excluded the
    same way they are absent from the sim's price DataFrame. Returns None on any
    failure so callers fall back to a weekday approximation rather than blocking.
    """
    try:
        from futu import OpenQuoteContext, TradeDateMarket

        quote_context = OpenQuoteContext(host="127.0.0.1", port=11111)
        try:
            return_code, day_data = quote_context.request_trading_days(
                market=TradeDateMarket.US,
                start=start_date_text,
                end=end_date_text,
            )
        finally:
            quote_context.close()
    except Exception as calendar_error:  # noqa: BLE001 - degrade gracefully
        LOGGER.warning("Failed to load US trading days: %s", calendar_error)
        return None
    if return_code != 0 or not isinstance(day_data, list):
        LOGGER.warning("Trading-day query returned code %s", return_code)
        return None
    days = [
        str(entry.get("time"))
        for entry in day_data
        if isinstance(entry, dict) and entry.get("time")
    ]
    return sorted(days)


def _count_trading_bars_held(
    entry_date_text: str,
    signal_date_text: str,
    trading_days: list[str] | None,
) -> int:
    """Trading bars held since entry, matching the sim's ``bars_since_anchor``.

    The sim sets the counter to 0 on the entry bar and increments once per
    subsequent trading bar, gating exits on ``bars_since_anchor >= min_hold``.
    To mirror that exactly this counts trading days strictly after the entry
    date up to and including the signal date (entry-exclusive). ISO date
    strings compare lexicographically in chronological order.

    Falls back to the calendar-weekday count minus one (entry-exclusive) when
    the trading calendar is unavailable.
    """
    if trading_days is not None:
        return sum(
            1 for day in trading_days
            if entry_date_text < day <= signal_date_text
        )
    return max(0, _count_weekday_bars_held(entry_date_text, signal_date_text) - 1)


def _load_current_bucket_exit_rules() -> dict[str, dict[str, Any]]:
    """Load per-bucket live exit rules keyed by bucket label and strategy id."""
    try:
        from stock_indicator import multi_bucket_today

        config = multi_bucket_today.load_multi_bucket_config(
            PRODUCTION_CONFIG_PATH
        )
    except (OSError, json.JSONDecodeError, ValueError) as config_error:
        LOGGER.warning("Failed to load current bucket exit rules: %s", config_error)
        return {}

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


def _complete_order_metadata_from_exit_rules(
    metadata: dict[str, Any],
    bucket_exit_rules: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Fill missing live-management metadata from the current bucket config."""
    lookup_keys = [
        str(metadata.get("bucket") or ""),
        str(metadata.get("strategy_id") or ""),
    ]
    for lookup_key in lookup_keys:
        if not lookup_key:
            continue
        exit_rule = bucket_exit_rules.get(lookup_key)
        if not exit_rule:
            continue
        if metadata.get("bucket") is None:
            metadata["bucket"] = exit_rule.get("bucket")
        if metadata.get("strategy_id") is None:
            metadata["strategy_id"] = exit_rule.get("strategy_id")
        if metadata.get("max_hold") is None:
            metadata["max_hold"] = exit_rule.get("max_hold")
        if metadata.get("reset_hold_on_reentry_signal") is None:
            metadata["reset_hold_on_reentry_signal"] = exit_rule.get(
                "reset_hold_on_reentry_signal"
            )
        break
    return metadata


def _format_dashboard_order_remark(order: dict[str, Any]) -> str:
    """Build a compact Futu order remark carrying strategy metadata."""
    return format_futu_order_remark(order)


def _parse_dashboard_order_remark(remark_text: str) -> dict[str, Any]:
    """Parse strategy metadata written into a Futu order remark."""
    return parse_futu_order_remark(remark_text)


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
    """Extract a YYYY-MM-DD date from a Futu deal/order row."""
    raw_time = _row_value(
        row,
        [
            "create_time",
            "updated_time",
            "dealt_time",
            "deal_time",
            "time",
        ],
    )
    if raw_time is None:
        return None
    try:
        return pandas.Timestamp(str(raw_time)).date().isoformat()
    except ValueError:
        return None


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
        order_id = _row_value(order_row, ["order_id"])
        remark = _row_value(order_row, ["remark", "order_remark"])
        if order_id is not None and remark:
            remarks_by_order_id[str(order_id)] = str(remark)
    return remarks_by_order_id


def _load_futu_open_trade_entries(
    trade_context: Any,
    trading_environment: Any,
    *,
    signal_date_text: str,
) -> dict[str, dict[str, Any]]:
    """Infer live open trade entries from Futu historical deals."""
    strategy_to_broker_symbol = _load_strategy_to_broker_symbol_map()
    bucket_exit_rules = _load_current_bucket_exit_rules()
    max_hold_values = [
        int(rule["max_hold"])
        for rule in bucket_exit_rules.values()
        if rule.get("max_hold") is not None
    ]
    maximum_max_hold = max(max_hold_values, default=30)
    lookback_days = max(120, maximum_max_hold * 3 + 30)
    signal_date_value = date.fromisoformat(signal_date_text)
    start_date_text = (signal_date_value - timedelta(days=lookback_days)).isoformat()

    if not hasattr(trade_context, "history_deal_list_query"):
        return {}
    return_code, deal_data = trade_context.history_deal_list_query(
        start=start_date_text,
        end=signal_date_text,
        trd_env=trading_environment,
    )
    if return_code != 0 or deal_data is None or len(deal_data) == 0:
        return {}

    remarks_by_order_id = _load_futu_order_remarks(
        trade_context,
        trading_environment,
        start_date_text=start_date_text,
        end_date_text=signal_date_text,
    )
    open_lots_by_symbol: dict[str, list[dict[str, Any]]] = {}

    # Futu returns deals newest-first; FIFO lot matching below assumes
    # chronological (oldest-first) order, so sort by deal time ascending.
    # Without this, a SELL can consume a newer lot instead of the older one,
    # leaving a stale earlier entry as the "open" lot and corrupting both the
    # resolved entry_date (min_hold/max_hold gate) and remaining_quantity.
    for time_column in ("create_time", "updated_time", "dealt_time", "deal_time"):
        if time_column in deal_data.columns:
            deal_data = deal_data.sort_values(time_column, kind="stable")
            break

    for _deal_index, deal_row in deal_data.iterrows():
        code_text = str(_row_value(deal_row, ["code"]) or "")
        symbol = _to_broker_symbol(code_text, strategy_to_broker_symbol)
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
            order_id = _row_value(deal_row, ["order_id"])
            remark_text = str(_row_value(deal_row, ["remark", "order_remark"]) or "")
            if not remark_text and order_id is not None:
                remark_text = remarks_by_order_id.get(str(order_id), "")
            metadata = _parse_dashboard_order_remark(remark_text)
            metadata = _complete_order_metadata_from_exit_rules(
                metadata,
                bucket_exit_rules,
            )
            entry_date = _extract_deal_date(deal_row)
            if entry_date is None:
                continue
            lot = {
                "symbol": symbol,
                "entry_date": entry_date,
                "remaining_quantity": quantity,
                "bucket": metadata.get("bucket"),
                "strategy_id": metadata.get("strategy_id"),
                "max_hold": metadata.get("max_hold"),
                "reset_hold_on_reentry_signal": (
                    metadata.get("reset_hold_on_reentry_signal")
                ),
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


def _build_max_hold_sell_orders(
    *,
    trade_context: Any,
    trading_environment: Any,
    held_symbols: set[str],
    signal_date_text: str,
    position_data: Any,
    existing_sell_symbols: set[str],
    entry_signal_symbols: set[str],
    futu_open_entries: dict[str, dict[str, Any]] | None = None,
    trading_days: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build max-hold SELL orders from Futu live position and deal history."""
    if futu_open_entries is None:
        futu_open_entries = _load_futu_open_trade_entries(
            trade_context,
            trading_environment,
            signal_date_text=signal_date_text,
        )
    sell_orders: list[dict[str, Any]] = []

    for symbol, entry_record in futu_open_entries.items():
        if symbol not in held_symbols or symbol in existing_sell_symbols:
            continue
        max_hold = entry_record.get("max_hold")
        if max_hold is None:
            continue

        reset_hold_on_reentry_signal = bool(
            entry_record.get("reset_hold_on_reentry_signal")
        )
        if reset_hold_on_reentry_signal and symbol in entry_signal_symbols:
            continue

        try:
            bars_held = _count_trading_bars_held(
                str(entry_record.get("entry_date", "")),
                signal_date_text,
                trading_days,
            )
        except ValueError:
            LOGGER.warning(
                "Skipping max-hold check for %s due to invalid Futu entry date %r",
                symbol,
                entry_record.get("entry_date"),
            )
            continue

        if bars_held < int(max_hold):
            continue

        quantity = 0
        if len(position_data) > 0:
            matching_positions = position_data[
                position_data["code"] == f"US.{symbol}"
            ]
            if len(matching_positions) > 0:
                quantity = int(matching_positions.iloc[0].get("qty", 0))
        sell_orders.append({
            "side": "SELL",
            "symbol": symbol,
            "qty": quantity,
            "price": _get_last_price(symbol),
            "order_type": "MARKET",
            "bucket": entry_record.get("bucket"),
            "strategy_id": entry_record.get("strategy_id"),
            "exit_reason": "max_hold",
            "bars_held": bars_held,
            "max_hold": int(max_hold),
            "entry_source": "futu_history_deals",
        })
        existing_sell_symbols.add(symbol)

    return sell_orders


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


def _update_persisted_entry_intent(
    entry_intent_record: dict[str, Any] | None,
    *,
    symbol: str,
    status: str,
    order_id: str | None = None,
    error: str | None = None,
) -> None:
    """Record a BUY result without misreporting an already-sent Futu order."""
    if entry_intent_record is None:
        return
    try:
        record_entry_submission_result(
            market_date_text=str(entry_intent_record["market_date"]),
            symbol=symbol,
            status=status,
            order_id=order_id,
            error=error,
            state_path=LIVE_STATE_DIRECTORY / "pending_entry_intents.json",
        )
    except (OSError, TypeError, ValueError) as state_error:
        LOGGER.error(
            "Failed to update persisted entry intent for %s: %s",
            symbol,
            state_error,
        )


def _signal_trades_path() -> Path:
    """Resolve the live signal_trades.json path at call time so tests can
    monkeypatch LIVE_STATE_DIRECTORY after import."""
    return LIVE_STATE_DIRECTORY / "signal_trades.json"


def _load_signal_trades() -> dict:
    """Read signal_trades.json. Returns empty dict on miss/parse error."""
    signal_trades_path = _signal_trades_path()
    if not signal_trades_path.exists():
        return {}
    try:
        data = json.loads(signal_trades_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _save_signal_trades(data: dict) -> None:
    """Atomic write of signal_trades.json local execution mirror."""
    signal_trades_path = _signal_trades_path()
    signal_trades_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = signal_trades_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    import os as _os
    _os.replace(tmp_path, signal_trades_path)


def _record_buy_in_signal_trades(
    symbol: str, strategy_id: str, entry_date: str
) -> None:
    """Append a sent BUY order to the local signal_trades mirror.

    Futu remains the live source of truth. This file is diagnostic only and
    must never drive live order preview, TP/SL, or max-hold decisions.
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
    """Remove a symbol from the local signal_trades mirror after SELL send.

    Looks across all strategy buckets because Futu remains responsible for
    live position truth.
    """
    strategy_to_broker_symbol = _load_strategy_to_broker_symbol_map()
    sold_broker_symbol = _to_broker_symbol(symbol, strategy_to_broker_symbol)
    trades = _load_signal_trades()
    changed = False
    for strategy_id, bucket_list in list(trades.items()):
        before = len(bucket_list)
        trades[strategy_id] = [
            entry for entry in bucket_list
            if _to_broker_symbol(
                str(entry.get("symbol") or ""),
                strategy_to_broker_symbol,
            ) != sold_broker_symbol
        ]
        if len(trades[strategy_id]) != before:
            changed = True
    if changed:
        _save_signal_trades(trades)


@app.get("/api/preview_orders")
def api_preview_orders():
    """Build order preview from latest signal + account data."""
    max_positions = _load_max_positions()
    strategy_to_broker_symbol = _load_strategy_to_broker_symbol_map()
    try:
        trd_ctx = _get_futu_trd_ctx()
        trd_env = _get_trd_env()
        ret_acc, acc_data = trd_ctx.accinfo_query(trd_env=trd_env)
        ret_pos, pos_data = trd_ctx.position_list_query(trd_env=trd_env)
        ret_orders, broker_order_data = trd_ctx.order_list_query(trd_env=trd_env)

        if ret_acc != 0:
            trd_ctx.close()
            return {"error": "Failed to query account"}
        if ret_orders != 0:
            trd_ctx.close()
            return {"error": "Failed to query current broker orders"}

        total_assets = float(acc_data.iloc[0].get("total_assets", 0))
        held_broker_symbols = set()
        if ret_pos == 0 and len(pos_data) > 0:
            for _position_index, position_row in pos_data.iterrows():
                try:
                    position_quantity = float(position_row.get("qty", 0))
                except (TypeError, ValueError):
                    continue
                # TODO: review
                # Futu can retain same-day closed positions as qty=0 rows.
                # Those rows are execution history, not occupied positions.
                if position_quantity <= 0:
                    continue
                held_broker_symbol = _normalize_symbol_text(
                    position_row.get("code", "")
                )
                if held_broker_symbol:
                    held_broker_symbols.add(held_broker_symbol)
        pending_buy_entries = _load_pending_buy_order_entries(
            broker_order_data
        )
        pending_buy_symbols = set(pending_buy_entries)
        pending_unfilled_buy_symbols = (
            pending_buy_symbols - held_broker_symbols
        )

    except Exception as exc:
        if "trd_ctx" in locals():
            trd_ctx.close()
        return {"error": f"Futu connection failed: {exc}"}

    # Parse latest log for signals
    dates = _get_log_dates()
    if not dates:
        trd_ctx.close()
        return {"error": "No log files found"}
    log = _parse_log(LOGS_DIRECTORY / f"{dates[0]}.log")
    risk_gate_state = _load_risk_score_gate_state(log.get("date"))
    if risk_gate_state.get("status") == "error":
        trd_ctx.close()
        return {"error": risk_gate_state.get("reason"), "risk_score_gate": risk_gate_state}
    buy_orders_blocked_by_risk_score = _risk_gate_blocks_buy_orders(risk_gate_state)
    expectancy_gate_state = _load_expectancy_gate_decision(log)
    if expectancy_gate_state.get("status") == "error":
        trd_ctx.close()
        return {
            "error": expectancy_gate_state.get("reason"),
            "expectancy_gate": expectancy_gate_state,
        }

    current_bucket_exit_rules = _load_current_bucket_exit_rules()
    risk_score_value = risk_gate_state.get("risk_score")
    current_bucket_entry_limits = _load_current_bucket_entry_limits(
        risk_score_value
        if isinstance(risk_score_value, int)
        else None,
        expectancy_priority_override_active=bool(
            expectancy_gate_state.get("priority_override_active", False)
        ),
    )
    if expectancy_gate_state.get("enabled") and not current_bucket_entry_limits:
        trd_ctx.close()
        return {
            "error": "failed to load bucket limits for enabled expectancy gate",
            "expectancy_gate": {
                **expectancy_gate_state,
                "status": "error",
            },
        }
    max_same_symbol = _load_current_max_same_symbol(
        current_bucket_entry_limits
    )
    tradable_candidates = sorted(
        log.get("tradable_entry_candidates", []),
        key=lambda candidate_record: _tradable_candidate_sort_key(
            candidate_record,
            current_bucket_entry_limits,
        ),
    )
    orders = []
    buy_order_index_by_symbol: dict[str, int] = {}

    # WR-gate phantom accounting (read-only here; persistence happens at
    # execute). Resolve which previously-recorded phantoms still hold a slot
    # as of today. Production uses the endogenous sensor directly; historical
    # configs may still add an optional risk-score activation threshold.
    eval_date_string = log.get("date") or ""
    wr_gate_order_config = _load_wr_gate_order_config()
    wr_activation_threshold = wr_gate_order_config[
        "risk_score_activation_threshold"
    ]
    wr_gate_armed = bool(
        wr_gate_order_config["configured"]
        and (
            wr_gate_order_config["activation_mode"] == "sensor_only"
            or (
                isinstance(wr_activation_threshold, int)
                and isinstance(risk_score_value, int)
                and risk_score_value >= wr_activation_threshold
            )
        )
    )
    open_phantom_positions = [
        phantom
        for phantom in _load_phantom_positions()
        if phantom.get("symbol")
        and _phantom_still_open(phantom, eval_date_string)
    ]
    open_phantom_symbols = {
        phantom["symbol"] for phantom in open_phantom_positions
    }
    open_phantom_broker_symbols = {
        _to_broker_symbol(
            str(phantom_symbol),
            strategy_to_broker_symbol,
        )
        for phantom_symbol in open_phantom_symbols
    }
    signal_date_text = str(log.get("date") or "")
    futu_open_entries: dict[str, dict[str, Any]] = {}
    if signal_date_text:
        try:
            futu_open_entries = _load_futu_open_trade_entries(
                trd_ctx,
                trd_env,
                signal_date_text=signal_date_text,
            )
        except Exception as load_error:
            LOGGER.warning(
                "Failed to load Futu open entries for bucket gates: %s",
                load_error,
            )
    bucket_position_counts = _count_live_positions_by_bucket(
        held_symbols=held_broker_symbols,
        futu_open_entries=futu_open_entries,
        open_phantom_positions=open_phantom_positions,
        pending_buy_entries=pending_buy_entries,
    )

    # Slot cap based on REAL Futu portfolio PLUS phantom-held slots (this
    # is the order layer's job — cron's signal layer emits Top N
    # candidates regardless of holdings). Phantoms are not real Futu
    # positions, so they must be subtracted explicitly or other buckets
    # would reclaim a slot the phantom is meant to hold. slots_remaining
    # counts down as each BUY (real or phantom) is queued.
    slots_remaining = max(
        0,
        max_positions
        - len(held_broker_symbols)
        - len(open_phantom_broker_symbols)
        - len(pending_unfilled_buy_symbols),
    )
    occupied_position_count = (
        len(held_broker_symbols)
        + len(open_phantom_broker_symbols)
        + len(pending_unfilled_buy_symbols)
    )

    occupied_symbol_counts = {
        broker_symbol: 1
        for broker_symbol in (
            held_broker_symbols
            | pending_unfilled_buy_symbols
            | open_phantom_broker_symbols
        )
    }

    # BUY competition happens exactly once here. Cron supplies every
    # tradable bucket/symbol record; this loop orders them from the current
    # settings and greedily admits against Futu positions, pending orders,
    # phantom slots, the global cap, bucket caps, and same-symbol capacity.
    for meta in tradable_candidates:
        strategy_symbol = str(meta.get("symbol") or "")
        broker_symbol = _to_broker_symbol(strategy_symbol, strategy_to_broker_symbol)
        if broker_symbol in held_broker_symbols:
            # Already long; cron emitted the signal but the order
            # layer skips dupes.
            continue
        if broker_symbol in pending_buy_symbols:
            # A previously-confirmed market order already owns this future
            # position. Never preview or execute a duplicate while it waits
            # for the next US session.
            continue
        ref_price = _get_last_price(broker_symbol)
        qty = (
            _compute_order_size(total_assets, ref_price, max_positions)
            if ref_price else 0
        )
        bucket_key = str(meta.get("bucket") or meta.get("strategy_id") or "")
        exit_rule = current_bucket_exit_rules.get(bucket_key, {})
        order_dict = {
            "side": "BUY",
            "symbol": broker_symbol,
            "qty": qty,
            "ref_price": ref_price,
            "order_type": "MARKET",
            "bucket": meta.get("bucket") or exit_rule.get("bucket"),
            "strategy_id": meta.get("strategy_id") or exit_rule.get("strategy_id"),
            "tp_pct": meta.get("tp_pct"),
            "sl_pct": meta.get("sl_pct"),
            "dollar_volume_rank": meta.get("dollar_volume_rank"),
            "min_hold_sl": (
                meta.get("min_hold_sl")
                if meta.get("min_hold_sl") is not None
                else exit_rule.get("min_hold_sl")
            ),
            "disable_sl_trigger": (
                meta.get("disable_sl_trigger")
                if meta.get("disable_sl_trigger") is not None
                else exit_rule.get("disable_sl_trigger")
            ),
            "max_hold": (
                meta.get("max_hold")
                if meta.get("max_hold") is not None
                else exit_rule.get("max_hold")
            ),
            "reset_hold_on_reentry_signal": (
                meta.get("reset_hold_on_reentry_signal")
                if meta.get("reset_hold_on_reentry_signal") is not None
                else exit_rule.get("reset_hold_on_reentry_signal")
            ),
        }
        if order_dict.get("min_hold_sl") is None:
            order_dict.pop("min_hold_sl", None)
        if order_dict.get("disable_sl_trigger") is None:
            order_dict.pop("disable_sl_trigger", None)
        if order_dict.get("max_hold") is None:
            order_dict.pop("max_hold", None)
        if order_dict.get("reset_hold_on_reentry_signal") is None:
            order_dict.pop("reset_hold_on_reentry_signal", None)
        normalized_strategy_symbol = _normalize_symbol_text(strategy_symbol)
        if broker_symbol != normalized_strategy_symbol:
            order_dict["signal_symbol"] = normalized_strategy_symbol
        # A candidate is a phantom when the cron flagged its win-rate cross
        # as degrading and the WR gate is armed. Production is sensor-only;
        # the flag is stamped on gated buckets only, so non-gated buckets
        # always read wr_degrading=False.
        is_wr_gate_phantom = wr_gate_armed and bool(meta.get("wr_degrading"))
        is_expectancy_gate_phantom = bool(
            expectancy_gate_state.get("gate_closed", False)
        )
        is_phantom = is_wr_gate_phantom or is_expectancy_gate_phantom
        if expectancy_gate_state.get("enabled"):
            order_dict["signal_date"] = eval_date_string
            order_dict["min_hold_tp"] = meta.get("min_hold_tp")
            order_dict["rolling_mp"] = meta.get("rolling_mp")
            order_dict["wr_gate_phantom"] = is_wr_gate_phantom
            order_dict["expectancy_gated"] = is_expectancy_gate_phantom
            order_dict["expectancy_priority_override"] = bool(
                expectancy_gate_state.get("priority_override_active", False)
            )
        same_symbol_limit_reached = (
            max_same_symbol < 999
            and occupied_symbol_counts.get(broker_symbol, 0)
            >= max_same_symbol
        )
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
                f"(Futu held: {len(held_broker_symbols)}, "
                f"phantom: {len(open_phantom_symbols)})"
            )
        elif (
            bucket_skip_reason := _entry_limit_skip_reason(
                entry_limit=current_bucket_entry_limits.get(bucket_key),
                bucket_position_counts=bucket_position_counts,
                occupied_position_count=occupied_position_count,
            )
        ):
            order_dict["status"] = "bucket_cap"
            order_dict["qty"] = 0
            order_dict["skip_reason"] = bucket_skip_reason
        elif same_symbol_limit_reached:
            # Do not expose a second order row with the same side/symbol.
            # Execution authorization is keyed by side/symbol, so retaining
            # the losing bucket here would overwrite the winning canonical
            # order and make the valid BUY impossible to confirm.
            continue
        elif ref_price is None or qty <= 0:
            order_dict["status"] = "price_unavailable"
            order_dict["qty"] = 0
            order_dict["skip_reason"] = (
                "positive Futu reference price unavailable"
            )
        elif is_phantom:
            # Phantom: hold the slot, deploy zero capital, place no real
            # order. Carries phantom_record so execute can persist it to
            # the order layer's phantom list (slot stays occupied until it
            # adaptively closes).
            order_dict["status"] = (
                "expectancy_gate_phantom"
                if is_expectancy_gate_phantom
                else "wr_gate_phantom"
            )
            order_dict["qty"] = 0
            if is_expectancy_gate_phantom:
                phantom_reason_parts = [
                    "expectancy mean below stop threshold"
                ]
                if is_wr_gate_phantom:
                    phantom_reason_parts.append(
                        "WR sensor degrading"
                    )
                order_dict["skip_reason"] = (
                    "Zero-capital phantom: "
                    + " and ".join(phantom_reason_parts)
                    + " (slot held, no capital)"
                )
            else:
                if (
                    wr_gate_order_config["activation_mode"]
                    == "sensor_only"
                ):
                    order_dict["skip_reason"] = (
                        "WR-gate phantom: wr_degrading=True "
                        "(sensor-only; slot held, no capital)"
                    )
                else:
                    order_dict["skip_reason"] = (
                        "WR-gate phantom: wr_degrading=True and "
                        f"risk_score={risk_score_value} >= activation="
                        f"{wr_activation_threshold} "
                        f"for {risk_gate_state.get('year_month')} "
                        "(slot held, no capital)"
                    )
            order_dict["phantom_record"] = {
                "symbol": normalized_strategy_symbol,
                "signal_date": eval_date_string,
                "bucket": meta.get("bucket"),
                "tp_pct": meta.get("tp_pct"),
                "min_hold_tp": meta.get("min_hold_tp"),
                "max_hold": meta.get("max_hold"),
            }
            if expectancy_gate_state.get("enabled"):
                order_dict["phantom_record"].update({
                    "strategy_id": meta.get("strategy_id"),
                    "sl_pct": meta.get("sl_pct"),
                    "rolling_mp": meta.get("rolling_mp"),
                    "min_hold_sl": order_dict.get("min_hold_sl", 0),
                    "disable_sl_trigger": order_dict.get(
                        "disable_sl_trigger", False
                    ),
                    "reset_hold_on_reentry_signal": order_dict.get(
                        "reset_hold_on_reentry_signal", False
                    ),
                    "wr_gate_phantom": is_wr_gate_phantom,
                    "expectancy_gated": is_expectancy_gate_phantom,
                    "expectancy_priority_override": order_dict[
                        "expectancy_priority_override"
                    ],
                })
            slots_remaining -= 1
            occupied_position_count += 1
            bucket_label = str(order_dict.get("bucket") or "")
            if bucket_label:
                bucket_position_counts[bucket_label] = (
                    bucket_position_counts.get(bucket_label, 0) + 1
                )
            occupied_symbol_counts[broker_symbol] = (
                occupied_symbol_counts.get(broker_symbol, 0) + 1
            )
        else:
            slots_remaining -= 1
            occupied_position_count += 1
            bucket_label = str(order_dict.get("bucket") or "")
            if bucket_label:
                bucket_position_counts[bucket_label] = (
                    bucket_position_counts.get(bucket_label, 0) + 1
                )
            occupied_symbol_counts[broker_symbol] = (
                occupied_symbol_counts.get(broker_symbol, 0) + 1
            )
        existing_order_index = buy_order_index_by_symbol.get(broker_symbol)
        if existing_order_index is None:
            buy_order_index_by_symbol[broker_symbol] = len(orders)
            orders.append(order_dict)
        else:
            existing_order = orders[existing_order_index]
            existing_order_won_allocation = (
                existing_order.get("status") not in PREVIEW_SKIP_STATUSES
                or existing_order.get("status")
                in {"wr_gate_phantom", "expectancy_gate_phantom"}
            )
            current_order_won_allocation = (
                order_dict.get("status") not in PREVIEW_SKIP_STATUSES
                or order_dict.get("status")
                in {"wr_gate_phantom", "expectancy_gate_phantom"}
            )
            if current_order_won_allocation and not existing_order_won_allocation:
                orders[existing_order_index] = order_dict

    # SELL orders (exit signals for held positions). The signal layer (cron)
    # treats its own sim portfolio as truth, so it can emit an exit signal for
    # a symbol that Futu only filled this signal day. The order layer must
    # re-check min_hold against the Futu entry_date before sending the SELL.
    production_min_hold = _load_production_min_hold()

    # Trading calendar covering every open-entry hold window so min_hold and
    # max_hold bar counts match the sim (actual trading bars, holidays/weekends
    # excluded). None on failure -> _count_trading_bars_held falls back to a
    # weekday approximation.
    trading_days: list[str] | None = None
    if signal_date_text and futu_open_entries:
        entry_dates = [
            str(entry.get("entry_date"))
            for entry in futu_open_entries.values()
            if entry.get("entry_date")
        ]
        if entry_dates:
            trading_days = _load_us_trading_days(
                min(entry_dates), signal_date_text
            )

    for strategy_symbol in log.get("sell_actions", []):
        broker_symbol = _to_broker_symbol(strategy_symbol, strategy_to_broker_symbol)
        if broker_symbol not in held_broker_symbols:
            continue
        qty = 0
        if ret_pos == 0 and len(pos_data) > 0:
            match = pos_data[pos_data["code"] == f"US.{broker_symbol}"]
            if len(match) > 0:
                qty = int(match.iloc[0].get("qty", 0))
        price = _get_last_price(broker_symbol)
        order_dict: dict[str, Any] = {
            "side": "SELL",
            "symbol": broker_symbol,
            "qty": qty,
            "price": price,
            "order_type": "MARKET",
            "exit_reason": "signal",
        }
        normalized_strategy_symbol = _normalize_symbol_text(strategy_symbol)
        if broker_symbol != normalized_strategy_symbol:
            order_dict["signal_symbol"] = normalized_strategy_symbol
        futu_entry = futu_open_entries.get(broker_symbol)
        if production_min_hold > 0 and futu_entry is not None:
            entry_date_text = str(futu_entry.get("entry_date") or "")
            try:
                bars_held = _count_trading_bars_held(
                    entry_date_text, signal_date_text, trading_days
                )
            except ValueError:
                bars_held = None
                LOGGER.warning(
                    "Skipping min_hold check for %s due to invalid Futu "
                    "entry date %r",
                    broker_symbol,
                    entry_date_text,
                )
            if bars_held is not None and bars_held < production_min_hold:
                order_dict["status"] = "min_hold_block"
                order_dict["qty"] = 0
                order_dict["bars_held"] = bars_held
                order_dict["min_hold"] = production_min_hold
                order_dict["entry_date"] = entry_date_text
                order_dict["skip_reason"] = (
                    f"bars_held={bars_held} < min_hold={production_min_hold} "
                    f"(entry={entry_date_text})"
                )
        orders.append(order_dict)

    existing_sell_symbols = {
        order["symbol"]
        for order in orders
        if order.get("side") == "SELL" and order.get("symbol")
    }
    orders.extend(
        _build_max_hold_sell_orders(
            trade_context=trd_ctx,
            trading_environment=trd_env,
            held_symbols=held_broker_symbols,
            signal_date_text=signal_date_text,
            position_data=pos_data,
            existing_sell_symbols=existing_sell_symbols,
            entry_signal_symbols={
                _to_broker_symbol(strategy_symbol, strategy_to_broker_symbol)
                for strategy_symbol in log.get("entry_signals", [])
            },
            futu_open_entries=futu_open_entries,
            trading_days=trading_days,
        )
    )

    trd_ctx.close()

    return {
        "orders": orders,
        "total_assets_hkd": total_assets,
        "margin": MARGIN_MULTIPLIER,
        "max_positions": max_positions,
        "held_count": len(held_broker_symbols),
        "pending_buy_count": len(pending_unfilled_buy_symbols),
        "phantom_count": len(open_phantom_symbols),
        "trading_env": TRADING_ENV,
        "signal_date": log.get("date"),
        "risk_score_gate": risk_gate_state,
        "expectancy_gate": expectancy_gate_state,
        "wr_gate": {
            "configured": wr_gate_order_config["configured"],
            "active": wr_gate_armed,
            "activation_mode": wr_gate_order_config["activation_mode"],
            "risk_score": risk_score_value,
            "activation_threshold": wr_activation_threshold,
            # The cron's per-run sensor heartbeat (ema/sma/breakeven/
            # degrading/window_full) — single source of truth for the
            # cross reading, even on a day with no gated entries.
            "sensor": log.get("wr_gate_sensor", {}),
            "open_phantoms": [
                {
                    "symbol": phantom.get("symbol"),
                    "signal_date": phantom.get("signal_date"),
                    "bucket": phantom.get("bucket"),
                }
                for phantom in open_phantom_positions
            ],
            "phantomed_today": [
                order["symbol"]
                for order in orders
                if (
                    order.get("status") == "wr_gate_phantom"
                    or order.get("wr_gate_phantom") is True
                )
            ],
        },
    }


class ExecuteRequest(BaseModel):
    orders: list[dict]


def _validate_requested_orders_against_server_preview(
    requested_orders: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return canonical preview orders selected by the dashboard client.

    # TODO: review
    The client may choose a subset, but it cannot introduce a symbol, side,
    quantity, or status not present in a fresh server-side preview. Execution
    uses the canonical server dictionaries so client-supplied metadata never
    reaches Futu.
    """
    preview_document = api_preview_orders()
    if preview_document.get("error"):
        reason = str(preview_document.get("error"))
        return [], [
            {
                "symbol": _normalize_symbol_text(order.get("symbol")),
                "status": "rejected",
                "reason": f"server preview failed: {reason}",
            }
            for order in requested_orders
        ]

    canonical_orders_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for preview_order in preview_document.get("orders", []):
        preview_side = str(preview_order.get("side") or "").upper()
        preview_symbol = _normalize_symbol_text(preview_order.get("symbol"))
        if preview_side in {"BUY", "SELL"} and preview_symbol:
            canonical_orders_by_key[(preview_side, preview_symbol)] = preview_order

    authorized_orders: list[dict[str, Any]] = []
    rejected_results: list[dict[str, Any]] = []
    selected_order_keys: set[tuple[str, str]] = set()
    for requested_order in requested_orders:
        requested_side = str(requested_order.get("side") or "").upper()
        requested_symbol = _normalize_symbol_text(requested_order.get("symbol"))
        requested_key = (requested_side, requested_symbol)
        canonical_order = canonical_orders_by_key.get(requested_key)
        rejection_reason: str | None = None
        if requested_key in selected_order_keys:
            rejection_reason = "duplicate order selection"
        elif canonical_order is None:
            rejection_reason = "order is not present in the current server preview"
        elif requested_order.get("qty") != canonical_order.get("qty"):
            rejection_reason = "quantity changed; refresh the preview"
        elif requested_order.get("status") != canonical_order.get("status"):
            rejection_reason = "order status changed; refresh the preview"

        if rejection_reason is not None:
            rejected_results.append(
                {
                    "symbol": requested_symbol,
                    "status": "rejected",
                    "reason": rejection_reason,
                }
            )
            continue

        selected_order_keys.add(requested_key)
        authorized_orders.append(dict(canonical_order))

    return authorized_orders, rejected_results


@app.post("/api/execute_orders")
def api_execute_orders(req: ExecuteRequest):
    """Serialize execution so simultaneous confirms cannot duplicate orders."""
    # TODO: review
    with ORDER_EXECUTION_LOCK:
        return _execute_orders_from_fresh_server_preview(req)


def _execute_orders_from_fresh_server_preview(req: ExecuteRequest):
    """Execute confirmed orders via Futu API after fresh authorization."""
    from futu import ModifyOrderOp, OrderType, TrdSide

    authorized_orders, results = _validate_requested_orders_against_server_preview(
        req.orders
    )
    if not authorized_orders:
        return {"results": results}

    # TODO: review
    # Persist allocation acceptance before contacting Futu.  The expectancy
    # sensor follows the selected counterfactual trade in exactly the same way
    # for funded, WR-phantom, and expectancy-phantom entries.  A state error
    # blocks the batch rather than submitting capital under an unrecorded
    # sensor transition.
    accepted_expectancy_trades: list[dict[str, Any]] = []
    for order in authorized_orders:
        order_status = order.get("status")
        order_is_zero_capital_phantom = order_status in {
            "wr_gate_phantom",
            "expectancy_gate_phantom",
        }
        order_is_funded_buy = (
            order.get("side") == "BUY"
            and order_status not in PREVIEW_SKIP_STATUSES
            and int(order.get("qty", 0)) > 0
        )
        if not order_is_zero_capital_phantom and not order_is_funded_buy:
            continue
        accepted_expectancy_trades.append({
            "signal_date": order.get("signal_date"),
            "bucket": order.get("bucket"),
            "strategy_id": order.get("strategy_id"),
            "symbol": order.get("signal_symbol") or order.get("symbol"),
            "tp_pct": order.get("tp_pct"),
            "sl_pct": order.get("sl_pct"),
            "min_hold_tp": order.get("min_hold_tp"),
            "min_hold_sl": order.get("min_hold_sl"),
            "disable_sl_trigger": order.get("disable_sl_trigger", False),
            "max_hold": order.get("max_hold"),
            "rolling_mp": order.get("rolling_mp"),
            "reset_hold_on_reentry_signal": order.get(
                "reset_hold_on_reentry_signal", False
            ),
            "wr_gate_phantom": (
                order_status == "wr_gate_phantom"
                or bool(order.get("wr_gate_phantom"))
            ),
            "expectancy_gated": (
                order_status == "expectancy_gate_phantom"
                or bool(order.get("expectancy_gated"))
            ),
            "expectancy_priority_override": bool(
                order.get("expectancy_priority_override", False)
            ),
        })
    if accepted_expectancy_trades:
        try:
            live_expectancy_gate.record_accepted_trades_at_path(
                state_path=EXPECTANCY_GATE_STATE_PATH,
                config_document=_load_production_config(),
                accepted_trades=accepted_expectancy_trades,
            )
        except (OSError, TypeError, ValueError) as expectancy_state_error:
            return {
                "results": results
                + [
                    {
                        "symbol": order.get("symbol"),
                        "status": "error",
                        "error": (
                            "Failed to persist expectancy gate acceptance: "
                            f"{expectancy_state_error}"
                        ),
                    }
                    for order in authorized_orders
                ]
            }

    trd_ctx = _get_futu_trd_ctx()
    trd_env = _get_trd_env()

    latest_gate_state: dict[str, Any] | None = None
    latest_dates = _get_log_dates()
    if latest_dates:
        latest_gate_state = _load_risk_score_gate_state(latest_dates[0])

    # Phantom records confirmed in this execute call, persisted after the
    # loop alongside a prune of any phantom that has since closed.
    confirmed_phantom_records: list[dict[str, Any]] = []

    for order in authorized_orders:
        symbol = order["symbol"]
        side = TrdSide.BUY if order["side"] == "BUY" else TrdSide.SELL
        qty = order["qty"]
        code = f"US.{symbol}"

        # Preview may mark orders as non-sendable (slot cap, dashboard
        # risk-score stop, signal-exit min_hold gate). Skip without sending so
        # confirm cannot bypass the preview-layer guardrails.
        if order.get("status") in PREVIEW_SKIP_STATUSES:
            # A phantom is recorded (not sent) so its slot stays occupied
            # across days until it adaptively closes.
            if order.get("status") in {
                "wr_gate_phantom",
                "expectancy_gate_phantom",
            } and order.get("phantom_record"):
                confirmed_phantom_records.append(order["phantom_record"])
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

        entry_intent_record: dict[str, Any] | None = None
        if order["side"] == "BUY" and TRADING_ENV == "REAL":
            try:
                # TODO: review
                # Persist confirmation before contacting Futu so a rejected
                # order can be resized and retried by the pre-open scheduler.
                entry_intent_record = record_confirmed_entry_intent(
                    order,
                    state_path=(
                        LIVE_STATE_DIRECTORY / "pending_entry_intents.json"
                    ),
                )
            except (OSError, TypeError, ValueError) as intent_error:
                results.append({
                    "symbol": symbol,
                    "status": "error",
                    "error": (
                        "Failed to persist confirmed entry intent: "
                        f"{intent_error}"
                    ),
                })
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
                            and str(orow.get("order_status", "")).upper()
                            not in TERMINAL_BROKER_ORDER_STATUSES
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
                remark=(
                    _format_dashboard_order_remark(order)
                    if order["side"] == "BUY" else None
                ),
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
                _update_persisted_entry_intent(
                    entry_intent_record,
                    symbol=symbol,
                    status="submitted",
                    order_id=order_id,
                )
                # Local mirror update only. Futu positions/orders/history remain
                # the source of truth for all live decisions.
                if order["side"] == "BUY":
                    _record_buy_in_signal_trades(
                        symbol=(
                            order.get("signal_symbol")
                            or symbol
                        ),
                        strategy_id=order.get("strategy_id", "") or "",
                        entry_date=(
                            latest_dates[0]
                            if latest_dates
                            else date.today().isoformat()
                        ),
                    )
                else:
                    _remove_sell_from_signal_trades(symbol)
            else:
                failed_result = {
                    "symbol": symbol,
                    "side": order["side"],
                    "qty": qty,
                    "status": "failed",
                    "error": str(data),
                    "env": TRADING_ENV,
                    "timestamp": datetime.now().isoformat(),
                }
                results.append(failed_result)
                _log_order(failed_result)
                _update_persisted_entry_intent(
                    entry_intent_record,
                    symbol=symbol,
                    status="failed",
                    error=str(data),
                )
        except Exception as exc:
            error_result = {
                "symbol": symbol,
                "side": order["side"],
                "qty": qty,
                "status": "error",
                "error": str(exc),
                "env": TRADING_ENV,
                "timestamp": datetime.now().isoformat(),
            }
            results.append(error_result)
            _log_order(error_result)
            _update_persisted_entry_intent(
                entry_intent_record,
                symbol=symbol,
                status="error",
                error=str(exc),
            )

    # Persist the phantom list: prune any phantom that has adaptively
    # closed (frees its slot) and append this call's confirmed phantoms,
    # deduped by (symbol, signal_date) so a re-execute cannot double-book a
    # slot. Done once after the loop regardless of new phantoms so closed
    # slots are released even on a phantom-free execute.
    eval_date_string = latest_dates[0] if latest_dates else ""
    existing_phantoms = _load_phantom_positions()
    retained_phantoms = [
        phantom
        for phantom in existing_phantoms
        if _phantom_still_open(phantom, eval_date_string)
    ]
    seen_phantom_keys = {
        (phantom.get("symbol"), phantom.get("signal_date"))
        for phantom in retained_phantoms
    }
    for record in confirmed_phantom_records:
        key = (record.get("symbol"), record.get("signal_date"))
        if key not in seen_phantom_keys:
            retained_phantoms.append(record)
            seen_phantom_keys.add(key)
    if retained_phantoms != existing_phantoms:
        _save_phantom_positions(retained_phantoms)

    trd_ctx.close()
    return {"results": results}


@app.post("/api/place_tp_sl")
def api_place_tp_sl():
    """Trigger TP/SL placement (same logic as place_tp_sl.py)."""
    try:
        from stock_indicator.place_tp_sl import main as _tp_sl_main

        import io

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

  <!-- Cron / Dashboard Communication -->
  <div class="card full" id="contract-card">
    <h2>Cron / Dashboard Communication</h2>
    <div id="contract-content"></div>
  </div>

  <!-- Order Preview -->
  <div class="card full" id="orders-card">
    <h2>Order Preview <span class="env-badge" id="env-badge"></span></h2>
    <div id="orders-content"><div style="color:var(--text2)">Loading gate status and order preview...</div></div>
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

// Build symbol -> bucket short-name map from cron signal logs only. Local
// adaptive_state is diagnostic and must not label live positions.
function shortBucket(b) {
  return (b || '').replace('_production', '').replace('_explore', '');
}

function buildBucketMap(log, adaptive) {
  const m = {};
  for (const e of (log.entry_signal_records || [])) {
    if (e.symbol && e.bucket) m[e.symbol] = shortBucket(e.bucket);
  }
  for (const e of (log.frozen_entries || [])) {
    if (e.symbol && e.bucket) m[e.symbol] = shortBucket(e.bucket);
  }
  return m;
}

function tagWithBucket(symbol, type, bucketMap) {
  const b = bucketMap[symbol];
  return tagWithBucketLabel(symbol, type, b);
}

// Per-record bucket label: needed because the same symbol can fire in
// two buckets on one day (e.g. MU in fish_tail_squeeze AND
// fish_tail_production) — a symbol-keyed map collapses them.
function tagWithBucketLabel(symbol, type, bucketLabel, extraText) {
  let labelText = bucketLabel || '';
  if (extraText) labelText += (labelText ? ' · ' : '') + extraText;
  const suffix = labelText ? ` <span style="opacity:0.7; font-size:0.8em">[${labelText}]</span>` : '';
  return `<span class="tag ${type}">${symbol}${suffix}</span>`;
}

function renderCommunicationContract(contract) {
  const fallbackContract = {
    title: 'Cron → Dashboard contract',
    steps: [
      {
        owner: 'Cron',
        label: 'Signal layer',
        detail: 'Finds every tradable signal, freezes TP/SL metadata, and updates the separate ADAPTIVE TP/SL virtual statistical history.'
      },
      {
        owner: 'Dashboard',
        label: 'Order layer',
        detail: 'Checks live Futu positions, real account slots, and risk gate before orders.'
      },
      {
        owner: 'Futu',
        label: 'Live truth',
        detail: 'Live holdings and orders decide what can actually be traded.'
      }
    ],
    notes: [
      'Raw entries are not orders.',
      'Cron publishes every tradable signal with frozen TP/SL metadata.',
      'The ADAPTIVE TP/SL virtual trade history is statistical state, not positions.',
      'The local signal mirror is diagnostic only.'
    ]
  };
  const displayedContract = contract || fallbackContract;
  let html = '<div style="font-size:0.9em">';
  for (const contractStep of (displayedContract.steps || [])) {
    html += `<div style="margin-bottom:10px">
      <strong style="color:var(--blue)">${contractStep.owner || '—'}</strong>
      <span style="color:var(--text2)"> — ${contractStep.label || ''}</span>
      <div style="color:var(--text2); margin-top:2px">${contractStep.detail || ''}</div>
    </div>`;
  }
  const contractNotes = displayedContract.notes || [];
  if (contractNotes.length) {
    html += '<div style="border-top:1px solid var(--border); padding-top:8px; color:var(--text2)">';
    for (const contractNote of contractNotes) {
      html += `<div>• ${contractNote}</div>`;
    }
    html += '</div>';
  }
  html += '</div>';
  $('#contract-content').innerHTML = html;
}

function renderAdaptiveTpSlFromLog(log, adaptive) {
  const bucketTpSlRecords = log.bucket_tp_sl || [];
  let html = '';
  if (bucketTpSlRecords.length === 0) {
    html += stat('TP / SL', 'not available in selected log', '');
  } else {
    for (const bucketRecord of bucketTpSlRecords) {
      const bucketName = bucketRecord.bucket || '';
      const shortBucketName = bucketName.replace('_production', '').replace('_explore', '');
      const takeProfitValue = bucketRecord.tp_pct != null ? '+' + (bucketRecord.tp_pct * 100).toFixed(2) + '%' : '—';
      const stopLossValue = bucketRecord.sl_pct != null ? '-' + (bucketRecord.sl_pct * 100).toFixed(2) + '%' : '—';
      const maxHoldValue = bucketRecord.max_hold != null ? bucketRecord.max_hold + ' bars' : '—';
      html += stat(shortBucketName + ' TP', takeProfitValue, 'positive');
      html += stat(shortBucketName + ' SL', stopLossValue, 'negative');
      html += stat(shortBucketName + ' MaxHold', maxHoldValue);
    }
    html += stat('TP source', (log.date || 'selected date') + ' log');
  }

  const virtualHistoryState = log.adaptive_tp_sl_virtual_trade_history_state || {};
  if (virtualHistoryState.winner_returns != null || virtualHistoryState.loser_returns != null) {
    html += stat('Virtual reference sample', `${virtualHistoryState.winner_returns || 0} winners + ${virtualHistoryState.loser_returns || 0} losers`);
  } else {
    const virtualHistory = adaptive.adaptive_tp_sl_virtual_trade_history || {};
    const winnerReturns = virtualHistory.winner_returns || [];
    const loserReturns = virtualHistory.loser_returns || [];
    html += stat('Virtual reference sample', `${winnerReturns.length} winners + ${loserReturns.length} losers`);
  }
  $('#adaptive-stats').innerHTML = html;
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
  const adaptiveTpSlVirtualTradeHistory = adaptive.adaptive_tp_sl_virtual_trade_history || {};
  const positions = state.signal_trades || {}; // diagnostic local mirror only
  renderCommunicationContract(state.communication_contract);

  renderAdaptiveTpSlFromLog(log, adaptive);

  // Account
  let html = '';
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
    html += stat('Live Positions', 'unavailable', 'negative');
    html += stat('Local mirror', 'diagnostic only');
  }
  $('#account-stats').innerHTML = html;

  // Signals
  $('#signal-date').textContent = log.date || '—';
  html = '';
  const bucketMap = buildBucketMap(log, adaptive);
  html += '<div class="signal-row"><strong style="color:var(--text2)">ENTRY (raw):</strong> ';
  if (log.entry_signal_records && log.entry_signal_records.length) {
    // Render per record (not via the symbol-keyed map): one symbol can
    // legitimately fire in two buckets on the same day.
    html += log.entry_signal_records.map(e => tagWithBucketLabel(e.symbol, 'neutral', shortBucket(e.bucket))).join('');
  } else {
    html += (log.entry_signals && log.entry_signals.length) ? log.entry_signals.map(s => tagWithBucket(s, 'neutral', bucketMap)).join('') : '<span style="color:var(--text2)">—</span>';
  }
  html += '</div>';
  const tradableCandidates = log.tradable_entry_candidates || [];
  html += '<div class="signal-row"><strong style="color:var(--text2)">TRADABLE (cron):</strong> ';
  html += tradableCandidates.length ? tradableCandidates.map(candidate => tagWithBucketLabel(candidate.symbol, 'buy', shortBucket(candidate.bucket))).join('') : '<span style="color:var(--text2)">—</span>';
  html += '</div>';
  const filteredOutRecords = ((log.signal_candidate_summary||{}).filtered_out) || [];
  html += '<div class="signal-row"><strong style="color:var(--text2)">NOT TRADABLE (cron):</strong> ';
  html += filteredOutRecords.length ? filteredOutRecords.map(record => tagWithBucketLabel(record.symbol, 'neutral', shortBucket(record.bucket), record.reason || '?')).join('') : '<span style="color:var(--text2)">—</span>';
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
  html += 'Tradable candidates: ' + tradableCandidates.length + ' | ';
  html += 'Not tradable: ' + filteredOutRecords.length + ' | ';
  html += 'Exit signals: ' + (log.exit_signals||[]).length + ' | ';
  html += 'Virtual reference open trades: ' + (log.adaptive_tp_sl_virtual_open_trade_count||0);
  html += '</div>';
  $('#signals-content').innerHTML = html;

  // Positions
  html = '';
  if (futu.connected && futu.positions.length > 0) {
    html += '<table><tr><th>Symbol</th><th>Bucket</th><th>Qty</th><th>Cost</th><th>Price</th><th>P/L</th></tr>';
    for (const position of futu.positions) {
      const bucketShort = shortBucket(position.bucket) || '—';
      html += `<tr>
        <td><strong>${position.symbol}</strong></td>
        <td style="color:var(--text2)">${bucketShort}</td>
        <td>${position.qty}</td>
        <td>$${position.cost_price.toFixed(2)}</td>
        <td>$${position.market_price.toFixed(2)}</td>
        <td class="${plClass(position.unrealized_pl)}">${position.unrealized_pl >= 0 ? '+' : ''}$${position.unrealized_pl.toFixed(2)}</td>
      </tr>`;
    }
    html += '</table>';
  } else if (futu.connected) {
    html += '<div style="color:var(--text2)">No live positions</div>';
  } else {
    html += '<div style="color:var(--text2)">Live positions unavailable — local signal mirror is diagnostic only.</div>';
  }
  $('#positions-content').innerHTML = html;

  // Trade history — cache full list, render via setTradeTab.
  __all_trades_cache = adaptiveTpSlVirtualTradeHistory.closed_trades || [];
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
  // [ENTRY_SIGNAL] / [FROZEN_TP_SL] lines only; never infer Futu positions
  // from the ADAPTIVE TP/SL virtual trade history.
  const bucketMap = buildBucketMap(log, {});
  html += '<div class="signal-row"><strong style="color:var(--text2)">ENTRY (raw):</strong> ';
  if (log.entry_signal_records && log.entry_signal_records.length) {
    // Render per record (not via the symbol-keyed map): one symbol can
    // legitimately fire in two buckets on the same day.
    html += log.entry_signal_records.map(e => tagWithBucketLabel(e.symbol, 'neutral', shortBucket(e.bucket))).join('');
  } else {
    html += (log.entry_signals && log.entry_signals.length) ? log.entry_signals.map(s => tagWithBucket(s, 'neutral', bucketMap)).join('') : '<span style="color:var(--text2)">—</span>';
  }
  html += '</div>';
  const tradableCandidates = log.tradable_entry_candidates || [];
  html += '<div class="signal-row"><strong style="color:var(--text2)">TRADABLE (cron):</strong> ';
  html += tradableCandidates.length ? tradableCandidates.map(candidate => tagWithBucketLabel(candidate.symbol, 'buy', shortBucket(candidate.bucket))).join('') : '<span style="color:var(--text2)">—</span>';
  html += '</div>';
  const filteredOutRecords = ((log.signal_candidate_summary||{}).filtered_out) || [];
  html += '<div class="signal-row"><strong style="color:var(--text2)">NOT TRADABLE (cron):</strong> ';
  html += filteredOutRecords.length ? filteredOutRecords.map(record => tagWithBucketLabel(record.symbol, 'neutral', shortBucket(record.bucket), record.reason || '?')).join('') : '<span style="color:var(--text2)">—</span>';
  html += '</div>';
  html += '<div class="signal-row"><strong style="color:var(--text2)">SELL:</strong> ';
  html += (log.sell_actions && log.sell_actions.length) ? log.sell_actions.map(s => tagWithBucket(s, 'sell', bucketMap)).join('') : '<span style="color:var(--text2)">—</span>';
  html += '</div>';
  html += '<div class="signal-row"><strong style="color:var(--text2)">HOLD (min_hold):</strong> ';
  html += (log.hold_blocked && log.hold_blocked.length) ? log.hold_blocked.map(s => tagWithBucket(s, 'hold', bucketMap)).join('') : '<span style="color:var(--text2)">—</span>';
  html += '</div>';
  html += '<div style="margin-top:8px; font-size:0.85em; color:var(--text2)">';
  html += 'Raw entries: ' + (log.entry_signals||[]).length + ' | ';
  html += 'Tradable candidates: ' + tradableCandidates.length + ' | ';
  html += 'Not tradable: ' + filteredOutRecords.length + ' | ';
  html += 'Exit signals: ' + (log.exit_signals||[]).length + ' | ';
  html += 'Virtual reference open trades: ' + (log.adaptive_tp_sl_virtual_open_trade_count||0);
  html += '</div>';
  renderAdaptiveTpSlFromLog(log, {});
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

    const phantomCount = data.phantom_count || 0;
    const pendingBuyCount = data.pending_buy_count || 0;
    const slotsFree = Math.max(0, (data.max_positions||0) - (data.held_count||0) - phantomCount - pendingBuyCount);
    let html = '<div style="font-size:0.8em; color:var(--text2); margin-bottom:8px">';
    html += 'Assets: HK$' + (data.total_assets_hkd||0).toLocaleString(undefined,{minimumFractionDigits:0});
    html += ' | Margin: ' + data.margin + 'x';
    html += ' | Futu held: ' + data.held_count + '/' + data.max_positions;
    if (pendingBuyCount > 0) html += ' | Pending buys: ' + pendingBuyCount;
    if (phantomCount > 0) {
      html += ' | <span title="Zero-capital phantom slots held by WR or expectancy gates">👻 phantom: ' + phantomCount + '</span>';
    }
    html += ' | Slots free: <strong' + (slotsFree===0 ? ' style="color:var(--red)"' : '') + '>' + slotsFree + '</strong>';
    html += ' | Signal: ' + (data.signal_date||'—');
    const gate = data.risk_score_gate || {};
    if (gate.status === 'stop') {
      html += ' | <strong style="color:var(--red)">Risk gate STOP: ' + gate.year_month + ' score=' + gate.risk_score + '</strong>';
    }
    html += '</div>';

    // Expectancy gate status comes from the cron-owned heartbeat after the
    // backend has matched it against the persisted live state and config.
    const expectancy = data.expectancy_gate || {};
    if (expectancy.enabled) {
      const expectancyWarming = !expectancy.window_full;
      const expectancyStopped = expectancy.gate_closed === true;
      const expectancyPriority = expectancy.priority_override_active === true;
      const expectancyColor = expectancyStopped
        ? 'var(--red)'
        : expectancyWarming
        ? 'var(--orange)'
        : expectancyPriority
        ? 'var(--orange)'
        : 'var(--green)';
      const expectancyStatus = expectancyWarming
        ? 'WARMING'
        : expectancyStopped
        ? 'STOP'
        : expectancyPriority
        ? 'PRIORITY OVERRIDE'
        : 'OPEN';
      const expectancyMean = expectancy.mean != null
        ? Number(expectancy.mean).toFixed(4)
        : '—';
      const stopThreshold = expectancy.stop_threshold != null
        ? Number(expectancy.stop_threshold).toFixed(4)
        : '—';
      const softThreshold = expectancy.soft_threshold != null
        ? Number(expectancy.soft_threshold).toFixed(4)
        : '—';
      let expectancyHtml = '<div style="font-size:0.78em; margin-bottom:8px; padding:6px 8px; border-left:3px solid ' + expectancyColor + '; background:rgba(127,127,127,0.06)">';
      expectancyHtml += '<strong>Expectancy gate</strong> ';
      expectancyHtml += '<strong style="color:' + expectancyColor + '">' + expectancyStatus + '</strong>';
      expectancyHtml += ' | mean=' + expectancyMean;
      expectancyHtml += ' stop&lt;' + stopThreshold;
      expectancyHtml += ' soft&lt;' + softThreshold;
      expectancyHtml += ' | window=' + (expectancy.window || '—');
      if (expectancyWarming) {
        expectancyHtml += ' (cold-start ' + (expectancy.cold_start || '—') + ')';
      }
      expectancyHtml += ' | pending=' + (expectancy.open_pending != null ? expectancy.open_pending : '—');
      expectancyHtml += ' fed=' + (expectancy.fed_this_run != null ? expectancy.fed_this_run : '—');
      expectancyHtml += ' | closed episodes=' + (expectancy.closed_episodes != null ? expectancy.closed_episodes : '—');
      expectancyHtml += '</div>';
      html += expectancyHtml;
    }

    // WR-gate (phantom) status panel — visible whenever the gate is
    // configured, so a quiet day still shows the sensor is alive.
    const wr = data.wr_gate || {};
    if (wr.configured) {
      const s = wr.sensor || {};
      const sensorKnown = s.degrading != null;
      const degrading = sensorKnown ? s.degrading : null;
      const wrTriggered = wr.active && degrading === true;
      const activeColor = wrTriggered
        ? 'var(--red)'
        : !sensorKnown || !s.window_full
        ? 'var(--orange)'
        : 'var(--green)';
      const wrStatus = wrTriggered
        ? 'PHANTOM'
        : wr.active
        ? 'monitoring'
        : 'inactive';
      let wrHtml = '<div style="font-size:0.78em; margin-bottom:8px; padding:6px 8px; border-left:3px solid ' + activeColor + '; background:rgba(127,127,127,0.06)">';
      wrHtml += '<strong>WR-gate</strong> ';
      wrHtml += '<span style="color:' + activeColor + '">' + wrStatus + '</span>';
      if (wr.activation_mode === 'sensor_only') {
        wrHtml += ' (sensor-only)';
      } else {
        wrHtml += ' (risk_score=' + (wr.risk_score != null ? wr.risk_score : '—') + ' vs activation≥' + (wr.activation_threshold != null ? wr.activation_threshold : '—') + ')';
      }
      if (sensorKnown) {
        wrHtml += ' | sensor: ' + (degrading ? '<strong style="color:var(--red)">degrading</strong>' : 'healthy');
        wrHtml += ' (ema=' + (s.ema != null ? s.ema.toFixed(3) : '—') + ' sma=' + (s.sma != null ? s.sma.toFixed(3) : '—') + ' be=' + (s.breakeven != null ? s.breakeven.toFixed(3) : '—');
        wrHtml += ' win=' + (s.window || '—') + (s.window_full ? '' : ' warming') + ' open_pending=' + (s.open_pending != null ? s.open_pending : '—') + ')';
      } else {
        wrHtml += ' | <span style="color:var(--text2)">sensor: not bootstrapped</span>';
      }
      const pt = wr.phantomed_today || [];
      if (pt.length > 0) {
        wrHtml += ' | <strong>phantomed today:</strong> ' + pt.join(', ');
      }
      const op = wr.open_phantoms || [];
      if (op.length > 0) {
        wrHtml += ' | holding: ' + op.map(p => p.symbol).join(', ');
      }
      wrHtml += '</div>';
      html += wrHtml;
    }
    html += '<div style="font-size:0.78em; color:var(--text2); margin-bottom:8px">';
    html += 'Cron supplies every tradable candidate. This table is the single settings + Futu allocation pass that decides which orders are sendable.';
    html += '</div>';

    if (data.orders.length === 0) {
      const pendingBuyText = pendingBuyCount
        ? ' Pending broker buys: ' + pendingBuyCount + '.'
        : '';
      html += '<div style="color:var(--text2)">No dashboard-sendable orders (signal date: ' + (data.signal_date||'—') + ').' + pendingBuyText + ' Cron may still have tradable candidates that Futu holdings or allocation settings make non-actionable.</div>';
      $('#orders-content').innerHTML = html;
      return;
    }

    html += '<div class="order-row order-header"><span>Side</span><span>Symbol</span><span>Bucket</span><span>Rank</span><span>Qty</span><span>Ref Price</span><span>TP%</span><span>SL%</span></div>';
    for (const o of data.orders) {
      const sideClass = o.side === 'BUY' ? 'positive' : 'negative';
      const bucketShort = (o.bucket || '').replace('_production', '').replace('_explore', '') || '—';
      const tpDisplay = o.tp_pct != null ? '+' + (o.tp_pct * 100).toFixed(2) + '%' : '—';
      const slDisplay = o.sl_pct != null ? '-' + (o.sl_pct * 100).toFixed(2) + '%' : '—';
      const refDisplay = (o.ref_price != null) ? '$' + o.ref_price.toFixed(2)
                       : (o.price != null) ? '$' + o.price.toFixed(2) : '—';
      const rankDisplay = o.exit_reason === 'max_hold'
        ? `max ${o.bars_held}/${o.max_hold}`
        : (o.dollar_volume_rank != null) ? '#' + o.dollar_volume_rank : '—';
      const isPhantom = o.status === 'wr_gate_phantom' || o.status === 'expectancy_gate_phantom';
      const skipped = o.status === 'bucket_cap' || o.status === 'slot_full' || o.status === 'risk_score_stop' || o.status === 'min_hold_block' || o.status === 'price_unavailable' || isPhantom;
      const rowStyle = skipped ? ' style="opacity:0.45"' : '';
      const statusDetail = o.status === 'min_hold_block' && o.bars_held != null && o.min_hold != null
        ? `min_hold_block ${o.bars_held}/${o.min_hold}`
        : isPhantom ? '👻 phantom: slot held, no capital'
        : o.status;
      const sideLabel = skipped
        ? `${o.side} <span style="font-size:0.75em; opacity:0.8">(${statusDetail})</span>`
        : o.exit_reason === 'max_hold'
        ? `${o.side} <span style="font-size:0.75em; opacity:0.8">(max_hold)</span>`
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

// Load the mechanical gate panels on first paint. The preview remains
// read-only; order submission still requires the explicit confirm action.
async function initializeDashboard() {
  await load();
  await previewOrders();
}

initializeDashboard();
</script>
</body>
</html>
"""
