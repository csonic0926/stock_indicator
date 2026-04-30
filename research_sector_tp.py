"""Sector-specific TP research.

Goal: compute per-sector rolling MP+σ from a wide universe (Top3000),
then replay trades with sector-specific TP to compare against uniform TP.

Isolation: does NOT modify strategy.py, manage.py, or any production code.
Uses existing simulation infrastructure read-only.

Usage:
    venv/bin/python research_sector_tp.py [--data-source 1989|2014]
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from statistics import stdev

import pandas

sys.path.insert(0, "src")
from stock_indicator.strategy import AdaptiveTPSLConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = REPO_ROOT / "data"

# --- Config ---
ADAPTIVE_CONFIG = AdaptiveTPSLConfig(
    window=20,
    sigma_multiplier=0.5,
    fixed_sl=0.03,
    override_min_hold_tp_only=True,
    min_hold_tp=1,
    delayed_rolling_update=True,
)
MIN_HOLD = 5


def load_sector_map() -> dict[str, int]:
    """Load symbol -> FF12 group mapping."""
    csv_path = DATA_DIRECTORY / "symbols_with_sector.csv"
    if not csv_path.exists():
        LOGGER.error("symbols_with_sector.csv not found")
        return {}
    df = pandas.read_csv(csv_path)
    return dict(zip(df["ticker"].str.upper(), df["ff12"].astype(int)))


FF12_LABELS = {
    1: "NoDur", 2: "Durbl", 3: "Manuf", 4: "Enrgy", 5: "Chems",
    6: "BusEq", 7: "Telcm", 8: "Utils", 9: "Shops", 10: "Hlth",
    11: "Money", 12: "Other",
}


def compute_tp_sl(
    profits: list[float],
    config: AdaptiveTPSLConfig,
) -> tuple[float, float]:
    """Compute adaptive TP/SL from a list of trade profit percentages."""
    tp_pct = config.min_tp
    sl_pct = config.min_sl

    positive = [p for p in profits if p > 0]
    if len(positive) >= 3:
        mp = sum(positive) / len(positive)
        sp = stdev(positive) if len(positive) >= 2 else 0.0
        tp_pct = max(config.min_tp, mp + config.sigma_multiplier * sp)
        sl_pct = max(config.min_sl, tp_pct / config.target_r)

    if config.fixed_sl is not None:
        sl_pct = min(sl_pct, config.fixed_sl)

    return tp_pct, sl_pct


def replay_with_sector_tp(
    trades: list[Trade],
    trade_symbols: dict[Trade, str],
    sector_map: dict[str, int],
    config: AdaptiveTPSLConfig,
) -> tuple[list[Trade], dict[str, list[float]]]:
    """Replay trades using sector-specific rolling TP/SL.

    Returns adjusted trades and per-sector TP history.
    """
    # Sort trades by entry date
    sorted_trades = sorted(trades, key=lambda t: t.entry_date)

    # Rolling window per sector
    sector_rolling: dict[int, list[float]] = defaultdict(list)
    # Global rolling (for comparison)
    global_rolling: list[float] = []

    adjusted_trades = []
    sector_tp_history: dict[str, list[float]] = defaultdict(list)

    for trade in sorted_trades:
        symbol = trade_symbols.get(trade, "")
        ff12 = sector_map.get(symbol.upper(), 12)
        sector_label = FF12_LABELS.get(ff12, "Other")

        # Compute sector-specific TP
        sector_profits = sector_rolling[ff12]
        if len(sector_profits) >= config.min_samples:
            tp_pct, sl_pct = compute_tp_sl(sector_profits, config)
        else:
            # Fallback to global until sector has enough data
            if len(global_rolling) >= config.min_samples:
                tp_pct, sl_pct = compute_tp_sl(global_rolling, config)
            else:
                tp_pct = config.min_tp
                sl_pct = config.min_sl

        sector_tp_history[sector_label].append(tp_pct)

        # Replay
        adjusted = _replay_trade_with_adaptive_tp_sl(
            trade, tp_pct, sl_pct,
            minimum_holding_bars=MIN_HOLD,
            minimum_holding_bars_tp=config.min_hold_tp,
        )
        adjusted_trades.append(adjusted)

        # Update rolling windows with RAW trade profit (not adjusted)
        raw_pct = (
            (trade.exit_price - trade.entry_price) / trade.entry_price
            if trade.entry_price > 0 else 0.0
        )
        sector_rolling[ff12].append(raw_pct)
        if len(sector_rolling[ff12]) > config.window:
            sector_rolling[ff12].pop(0)

        global_rolling.append(raw_pct)
        if len(global_rolling) > config.window:
            global_rolling.pop(0)

    return adjusted_trades, dict(sector_tp_history)


def replay_with_uniform_tp(
    trades: list[Trade],
    config: AdaptiveTPSLConfig,
) -> list[Trade]:
    """Replay trades using uniform (global) rolling TP/SL — baseline."""
    sorted_trades = sorted(trades, key=lambda t: t.entry_date)
    rolling: list[float] = []
    adjusted_trades = []

    for trade in sorted_trades:
        if len(rolling) >= config.min_samples:
            tp_pct, sl_pct = compute_tp_sl(rolling, config)
        else:
            tp_pct = config.min_tp
            sl_pct = config.min_sl

        adjusted = _replay_trade_with_adaptive_tp_sl(
            trade, tp_pct, sl_pct,
            minimum_holding_bars=MIN_HOLD,
            minimum_holding_bars_tp=config.min_hold_tp,
        )
        adjusted_trades.append(adjusted)

        raw_pct = (
            (trade.exit_price - trade.entry_price) / trade.entry_price
            if trade.entry_price > 0 else 0.0
        )
        rolling.append(raw_pct)
        if len(rolling) > config.window:
            rolling.pop(0)

    return adjusted_trades


def summarize_trades(trades: list[Trade], label: str) -> dict:
    """Compute summary stats for a list of trades."""
    if not trades:
        return {"label": label, "n": 0}

    profits = [(t.exit_price - t.entry_price) / t.entry_price for t in trades]
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p <= 0]

    return {
        "label": label,
        "n": len(trades),
        "wr": len(wins) / len(trades) * 100,
        "mean_pct": sum(profits) / len(profits) * 100,
        "mp": sum(wins) / len(wins) * 100 if wins else 0,
        "ml": sum(losses) / len(losses) * 100 if losses else 0,
        "pl": abs((sum(wins) / len(wins)) / (sum(losses) / len(losses)))
        if wins and losses else float("inf"),
        "mean_hold": sum(t.holding_period for t in trades) / len(trades),
    }


RAW_CSV = REPO_ROOT / "logs/multi_bucket_simulation_result/multi_bucket_simulation_20260430_120230.csv"


def main():
    LOGGER.info("Loading raw (no TP/SL) trades from %s", RAW_CSV)
    if not RAW_CSV.exists():
        LOGGER.error("Raw CSV not found: %s", RAW_CSV)
        return

    trade_df = pandas.read_csv(RAW_CSV)
    LOGGER.info("Loaded %d trades", len(trade_df))

    # Load sector map
    sector_map = load_sector_map()
    LOGGER.info("Sector map loaded: %d symbols", len(sector_map))

    # Analyze sector distribution in trades
    trade_df["ff12"] = trade_df["symbol"].str.upper().map(sector_map).fillna(12).astype(int)
    trade_df["sector"] = trade_df["ff12"].map(FF12_LABELS).fillna("Other")

    LOGGER.info("\n--- Trade distribution by sector ---")
    for sector, grp in trade_df.groupby("sector"):
        wins = (grp["percentage_change"] > 0).sum()
        wr = wins / len(grp) * 100
        mp = grp.loc[grp["percentage_change"] > 0, "percentage_change"].mean() * 100 if wins > 0 else 0
        losses = grp[grp["percentage_change"] <= 0]
        ml = losses["percentage_change"].mean() * 100 if len(losses) > 0 else 0
        pl = abs(mp / ml) if ml != 0 else float("inf")
        LOGGER.info(
            "  %-8s  n=%4d  WR=%.0f%%  MP=%.2f%%  ML=%.2f%%  P/L=%.2f",
            sector, len(grp), wr, mp, ml, pl,
        )

    # Per-sector rolling MP computation (from CSV data)
    LOGGER.info("\n--- Per-sector rolling MP (window=%d) ---", ADAPTIVE_CONFIG.window)
    for ff12 in sorted(trade_df["ff12"].unique()):
        sector_trades = trade_df[trade_df["ff12"] == ff12].sort_values("entry_date")
        label = FF12_LABELS.get(ff12, "Other")
        profits = sector_trades["percentage_change"].tolist()

        # Compute final rolling window stats
        if len(profits) >= ADAPTIVE_CONFIG.window:
            recent = profits[-ADAPTIVE_CONFIG.window:]
        else:
            recent = profits

        positive = [p for p in recent if p > 0]
        if len(positive) >= 3:
            mp = sum(positive) / len(positive) * 100
            sp = stdev(positive) * 100 if len(positive) >= 2 else 0
            tp = max(ADAPTIVE_CONFIG.min_tp * 100, mp + ADAPTIVE_CONFIG.sigma_multiplier * sp)
        else:
            mp = 0
            sp = 0
            tp = ADAPTIVE_CONFIG.min_tp * 100

        LOGGER.info(
            "  FF12=%2d %-8s  trades=%4d  MP=%.2f%%  σ=%.2f%%  TP=%.2f%%",
            ff12, label, len(sector_trades), mp, sp, tp,
        )

    # Summary: compare uniform vs sector-specific TP potential
    LOGGER.info("\n--- Uniform TP (all sectors pooled) ---")
    all_profits = trade_df["percentage_change"].tolist()
    recent_all = all_profits[-ADAPTIVE_CONFIG.window:]
    pos_all = [p for p in recent_all if p > 0]
    if len(pos_all) >= 3:
        mp_all = sum(pos_all) / len(pos_all) * 100
        sp_all = stdev(pos_all) * 100
        tp_all = max(ADAPTIVE_CONFIG.min_tp * 100, mp_all + ADAPTIVE_CONFIG.sigma_multiplier * sp_all)
    else:
        mp_all = 0
        sp_all = 0
        tp_all = ADAPTIVE_CONFIG.min_tp * 100
    LOGGER.info("  MP=%.2f%%  σ=%.2f%%  TP=%.2f%%", mp_all, sp_all, tp_all)

    # Save sector distribution
    csv_path = DATA_DIRECTORY / "sector_tp_analysis.csv"
    trade_df[["symbol", "sector", "ff12", "entry_date", "exit_date",
              "percentage_change", "holding_bars", "exit_reason"]].to_csv(csv_path, index=False)
    LOGGER.info("Sector analysis saved to %s", csv_path)


if __name__ == "__main__":
    main()
