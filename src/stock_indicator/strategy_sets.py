"""Utilities for loading named strategy pairs from a config file.

The configuration lives at ``data/strategy_sets.csv`` and should contain
rows with columns: ``strategy_id,buy,sell``. Each row defines a reusable
strategy identifier that expands to a concrete buy/sell pair.

Optional columns for entry signal filtering:

    d_sma_min, d_sma_max    — inclusive range for d(sma_angle)
    ema_min, ema_max        — inclusive range for ema_angle
    d_ema_min, d_ema_max    — inclusive range for d(ema_angle)
    price_score_min         — lower bound for price_concentration_score
    price_score_max         — upper bound for price_concentration_score

When a column is absent or empty the corresponding filter is disabled.

Example CSV contents:

    strategy_id,buy,sell,d_sma_min,ema_min,price_score_max
    s4,"ema_sma_cross_testing_4_...","ema_sma_cross_testing_5_...",0.4,0.15,0.035

"""

# TODO: review

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


DATA_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "data"
STRATEGY_SET_PATH = DATA_DIRECTORY / "strategy_sets.csv"


@dataclass
class StrategyEntryFilters:
    """Optional entry-signal filters loaded from strategy_sets.csv."""

    d_sma_min: float | None = None
    d_sma_max: float | None = None
    ema_min: float | None = None
    ema_max: float | None = None
    d_ema_min: float | None = None
    d_ema_max: float | None = None
    price_score_min: float | None = None
    price_score_max: float | None = None
    near_delta_min: float | None = None
    near_delta_max: float | None = None
    price_tightness_min: float | None = None
    price_tightness_max: float | None = None
    # Fish-body shape + BSV filters (third instance: trend join after vacuum
    # turn). When all three are set, entry signal is gated by:
    # 1) 60-bar shape descriptor's overall slope >= shape_slope_min
    # 2) 60-bar shape descriptor's midpoint deviation <= shape_dev_50_max
    # 3) BSV footprint observed within last shape_bsv_lookback bars
    # When any is None, fish-body gating is disabled.
    shape_slope_min: float | None = None
    shape_dev_50_max: float | None = None
    shape_bsv_lookback: int | None = None


def load_strategy_entry_filters(
    path: Path | None = None,
) -> Dict[str, StrategyEntryFilters]:
    """Load per-strategy entry filters from strategy_sets.csv.

    Returns an empty mapping when the file does not exist or a strategy
    has no extra filter columns set.
    """
    file_path = path or STRATEGY_SET_PATH
    if not file_path.exists():
        return {}
    result: Dict[str, StrategyEntryFilters] = {}
    with file_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            strategy_id = (row.get("strategy_id") or "").strip()
            if not strategy_id:
                continue

            def _float_or_none(key: str) -> float | None:
                raw = (row.get(key) or "").strip()
                if not raw:
                    return None
                return float(raw)

            def _int_or_none(key: str) -> int | None:
                raw = (row.get(key) or "").strip()
                if not raw:
                    return None
                return int(raw)

            filters = StrategyEntryFilters(
                d_sma_min=_float_or_none("d_sma_min"),
                d_sma_max=_float_or_none("d_sma_max"),
                ema_min=_float_or_none("ema_min"),
                ema_max=_float_or_none("ema_max"),
                d_ema_min=_float_or_none("d_ema_min"),
                d_ema_max=_float_or_none("d_ema_max"),
                price_score_min=_float_or_none("price_score_min"),
                price_score_max=_float_or_none("price_score_max"),
                near_delta_min=_float_or_none("near_delta_min"),
                near_delta_max=_float_or_none("near_delta_max"),
                price_tightness_min=_float_or_none("price_tightness_min"),
                price_tightness_max=_float_or_none("price_tightness_max"),
                shape_slope_min=_float_or_none("shape_slope_min"),
                shape_dev_50_max=_float_or_none("shape_dev_50_max"),
                shape_bsv_lookback=_int_or_none("shape_bsv_lookback"),
            )
            result[strategy_id] = filters
    return result


def load_strategy_set_mapping(path: Path | None = None) -> Dict[str, Tuple[str, str]]:
    """Load a mapping of strategy_id to (buy, sell) strategy names.

    Returns an empty mapping when the file does not exist.
    """
    file_path = path or STRATEGY_SET_PATH
    if not file_path.exists():
        return {}
    mapping: Dict[str, Tuple[str, str]] = {}
    with file_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            strategy_id = (row.get("strategy_id") or "").strip()
            buy_name = (row.get("buy") or "").strip()
            sell_name = (row.get("sell") or "").strip()
            if not strategy_id or not buy_name or not sell_name:
                continue
            mapping[strategy_id] = (buy_name, sell_name)
    return mapping

