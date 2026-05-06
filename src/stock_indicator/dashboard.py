"""Web dashboard for stock indicator signals and portfolio state."""

from __future__ import annotations

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


def _parse_log(log_path: Path) -> dict[str, Any]:
    """Parse a daily text log into structured data."""
    text = log_path.read_text(encoding="utf-8")
    result: dict[str, Any] = {"raw": text, "date": log_path.stem}

    # Entry signals
    m = re.search(r"entry signals:\s*\[([^\]]*)\]", text)
    if m and m.group(1).strip():
        result["entry_signals"] = re.findall(r"'(\w+)'", m.group(1))
    else:
        result["entry_signals"] = []

    # Exit signals
    m = re.search(r"exit signals:\s*\[([^\]]*)\]", text)
    if m and m.group(1).strip():
        result["exit_signals"] = re.findall(r"'(\w+)'", m.group(1))
    else:
        result["exit_signals"] = []

    # Actions
    buy_m = re.search(r"BUY\s+(.+)", text)
    result["buy_actions"] = re.findall(r"'(\w+)'", buy_m.group(1)) if buy_m else []
    sell_m = re.search(r"SELL\s+(.+)", text)
    result["sell_actions"] = re.findall(r"'(\w+)'", sell_m.group(1)) if sell_m else []
    hold_m = re.search(r"HOLD\s+\(min_hold\)\s+(.+)", text)
    result["hold_blocked"] = re.findall(r"'(\w+)'", hold_m.group(1)) if hold_m else []

    # Adaptive TP/SL
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
    result["position_count"] = int(pos_m.group(1)) if pos_m else 0

    return result


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


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
    dates = _get_log_dates()
    latest_log = None
    if dates:
        latest_log = _parse_log(LOGS_DIRECTORY / f"{dates[0]}.log")
    return {
        "signal_trades": signal_trades,
        "adaptive_state": adaptive,
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
MAX_POSITIONS = 6
# Use paper trading by default. Set to "REAL" to trade with real money.
TRADING_ENV = "REAL"
ORDER_LOG_DIR = LOGS_DIRECTORY


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


def _compute_order_size(total_assets_hkd: float, price_usd: float) -> int:
    """Compute qty: floor(total_assets * margin / max_pos / price).

    total_assets is in HKD, price is in USD.  Uses ~7.8 HKD/USD.
    """
    hkd_per_position = total_assets_hkd * MARGIN_MULTIPLIER / MAX_POSITIONS
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


@app.get("/api/preview_orders")
def api_preview_orders():
    """Build order preview from latest signal + account data."""
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

    # Read adaptive TP/SL from adaptive_state.json (System A output)
    adaptive = _load_json(DATA_DIRECTORY / "adaptive_state.json")
    tp_pct = adaptive.get("tp_pct")
    sl_pct = adaptive.get("sl_pct")
    if tp_pct is not None:
        tp_pct = tp_pct * 100  # match log format (percentage)
    if sl_pct is not None:
        sl_pct = sl_pct * 100

    orders = []

    # BUY orders
    for symbol in log.get("buy_actions", []):
        if symbol in held_symbols:
            continue
        ref_price = _get_last_price(symbol)
        qty = _compute_order_size(total_assets, ref_price) if ref_price else 0
        orders.append({
            "side": "BUY",
            "symbol": symbol,
            "qty": qty,
            "ref_price": ref_price,
            "order_type": "MARKET",
        })

    # SELL orders (exit signals for held positions)
    for symbol in log.get("sell_actions", []):
        if symbol not in held_symbols:
            continue
        # Find qty from positions
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
        "max_positions": MAX_POSITIONS,
        "held_count": len(held_symbols),
        "trading_env": TRADING_ENV,
        "signal_date": log.get("date"),
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

    for order in req.orders:
        symbol = order["symbol"]
        side = TrdSide.BUY if order["side"] == "BUY" else TrdSide.SELL
        qty = order["qty"]
        code = f"US.{symbol}"

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
  .order-row { display: grid; grid-template-columns: 60px 80px 60px 90px 90px 90px; gap: 8px; align-items: center;
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
    <h2>Rolling Trade History (last 20)</h2>
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

  // Adaptive stats — from adaptive_state.json (System A), not log
  let html = '';
  const a_tp = adaptive.tp_pct != null ? (adaptive.tp_pct * 100).toFixed(2) + '%' : '—';
  const a_sl = adaptive.sl_pct != null ? (adaptive.sl_pct * 100).toFixed(2) + '%' : '—';
  html += stat('TP', a_tp, 'positive');
  html += stat('SL', a_sl, 'negative');
  if (log.rolling_mp != null) html += stat('Rolling MP', '+' + pct(log.rolling_mp) + ' (n=' + log.rolling_mp_n + ')', 'positive');
  if (log.rolling_ml != null) html += stat('Rolling ML', '-' + pct(log.rolling_ml) + ' (n=' + log.rolling_ml_n + ')', 'negative');
  html += stat('Window', '20 trades');
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
    const fish_head = positions.fish_head_production || [];
    html += stat('Signal Positions', fish_head.length + '/6');
  }
  $('#account-stats').innerHTML = html;

  // Signals
  $('#signal-date').textContent = log.date || '—';
  html = '';
  html += '<div class="signal-row"><strong style="color:var(--text2)">BUY:</strong> ';
  html += (log.buy_actions && log.buy_actions.length) ? log.buy_actions.map(s => tag(s, 'buy')).join('') : '<span style="color:var(--text2)">—</span>';
  html += '</div>';
  html += '<div class="signal-row"><strong style="color:var(--text2)">SELL:</strong> ';
  html += (log.sell_actions && log.sell_actions.length) ? log.sell_actions.map(s => tag(s, 'sell')).join('') : '<span style="color:var(--text2)">—</span>';
  html += '</div>';
  html += '<div class="signal-row"><strong style="color:var(--text2)">HOLD (min_hold):</strong> ';
  html += (log.hold_blocked && log.hold_blocked.length) ? log.hold_blocked.map(s => tag(s, 'hold')).join('') : '<span style="color:var(--text2)">—</span>';
  html += '</div>';
  // Entry/exit signal count
  html += '<div style="margin-top:8px; font-size:0.85em; color:var(--text2)">';
  html += 'Entry signals: ' + (log.entry_signals||[]).length + ' | ';
  html += 'Exit signals: ' + (log.exit_signals||[]).length + ' | ';
  html += 'Positions: ' + (log.position_count||0) + '/6';
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
    const fish_head = positions.fish_head_production || [];
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

  // Trade history
  const trades = adaptive.closed_trades || [];
  if (trades.length > 0) {
    html = '<table><tr><th>Symbol</th><th>Entry</th><th>Exit</th><th>P/L %</th><th></th></tr>';
    for (const t of [...trades].reverse()) {
      const rawPct = t.raw_pct != null ? (t.raw_pct * 100) : null;
      const barWidth = rawPct != null ? Math.min(Math.abs(rawPct) * 8, 120) : 0;
      const barType = rawPct != null && rawPct >= 0 ? 'win' : 'loss';
      html += `<tr>
        <td><strong>${t.symbol}</strong></td>
        <td>${t.entry_date||'—'}</td>
        <td>${t.exit_date||'—'}</td>
        <td class="${plClass(rawPct)}">${rawPct != null ? (rawPct >= 0 ? '+' : '') + rawPct.toFixed(2) + '%' : '—'}</td>
        <td><div class="bar-container"><div class="bar ${barType}" style="width:${barWidth}px"></div></div></td>
      </tr>`;
    }
    html += '</table>';
  } else {
    html = '<div style="color:var(--text2)">No trade history</div>';
  }
  $('#trades-content').innerHTML = html;

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
  html += '<div class="signal-row"><strong style="color:var(--text2)">BUY:</strong> ';
  html += (log.buy_actions && log.buy_actions.length) ? log.buy_actions.map(s => tag(s, 'buy')).join('') : '<span style="color:var(--text2)">—</span>';
  html += '</div>';
  html += '<div class="signal-row"><strong style="color:var(--text2)">SELL:</strong> ';
  html += (log.sell_actions && log.sell_actions.length) ? log.sell_actions.map(s => tag(s, 'sell')).join('') : '<span style="color:var(--text2)">—</span>';
  html += '</div>';
  html += '<div class="signal-row"><strong style="color:var(--text2)">HOLD (min_hold):</strong> ';
  html += (log.hold_blocked && log.hold_blocked.length) ? log.hold_blocked.map(s => tag(s, 'hold')).join('') : '<span style="color:var(--text2)">—</span>';
  html += '</div>';
  html += '<div style="margin-top:8px; font-size:0.85em; color:var(--text2)">';
  html += 'Entry signals: ' + (log.entry_signals||[]).length + ' | ';
  html += 'Exit signals: ' + (log.exit_signals||[]).length + ' | ';
  html += 'Positions: ' + (log.position_count||0) + '/6';
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

    let html = '<div style="font-size:0.8em; color:var(--text2); margin-bottom:8px">';
    html += 'Assets: HK$' + (data.total_assets_hkd||0).toLocaleString(undefined,{minimumFractionDigits:0});
    html += ' | Margin: ' + data.margin + 'x';
    html += ' | Held: ' + data.held_count + '/' + data.max_positions;
    html += ' | Signal: ' + (data.signal_date||'—');
    html += '</div>';

    html += '<div class="order-row order-header"><span>Side</span><span>Symbol</span><span>Qty</span><span>Type</span><span>Ref Price</span><span></span></div>';
    for (const o of data.orders) {
      const sideClass = o.side === 'BUY' ? 'positive' : 'negative';
      html += `<div class="order-row">
        <span class="${sideClass}"><strong>${o.side}</strong></span>
        <span><strong>${o.symbol}</strong></span>
        <span>${o.qty}</span>
        <span>MARKET</span>
        <span style="color:var(--text2)">${o.ref_price ? '$'+o.ref_price.toFixed(2) : '—'}</span>
        <span></span>
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
