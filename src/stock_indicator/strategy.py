"""Strategy evaluation utilities."""
# TODO: review

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
import heapq
import logging
import math
from math import ceil
from pathlib import Path
from statistics import mean, median, stdev
from typing import Callable, Dict, Iterable, List, Tuple
import re

import numpy
import pandas

from .indicators import bsv, ema, kalman_filter, sma
from .chip_filter import calculate_chip_concentration_metrics
from .simulator import (
    SimulationResult,
    Trade,
    calculate_annual_returns,
    calculate_annual_trade_counts,
    calculate_maximum_concurrent_positions,
    calculate_max_drawdown,
    simulate_portfolio_balance,
    simulate_trades,
)
from .symbols import SP500_SYMBOL

LOGGER = logging.getLogger(__name__)


DEFAULT_SMA_ANGLE_RANGE: tuple[float, float] = (
    math.degrees(math.atan(-0.3)),
    math.degrees(math.atan(2.14)),
)

DEFAULT_SHIFTED_EMA_ANGLE_RANGE: tuple[float, float] = (
    math.degrees(math.atan(-1.0)),
    math.degrees(math.atan(1.0)),
)

# Two-layer signal architecture for ema_sma_cross_testing:
#   A-layer (T, signal-day):  filters evaluate signal-day quality.
#     - buy_name's ``angle_range``           → sma_angle[T]
#     - buy_name's ``near_range``            → near_price_volume_ratio[T]
#     - buy_name's ``above_range``           → above_price_volume_ratio[T]
#     - CSV ``d_sma_min/max``                → d_sma_angle[T]
#     - CSV ``ema_min/max``                  → ema_angle[T]
#     - CSV ``d_ema_min/max``                → d_ema_angle[T]
#     - CSV ``price_score_min/max``          → price_concentration_score[T]
#
#   B-layer (T+1, confirmation-day):  applied only when
#   ``use_confirmation_angle=True`` and is a single hardcoded check on
#   ``sma_angle`` at the execution-confirmation bar. This is the
#   "is the trend still supportive at execution time" gate; it is
#   independent of the bucket-specific A-layer filters.
CONFIRMATION_SMA_ANGLE_RANGE: tuple[float, float] = (-0.01, 65.0)


def _split_strategy_choices(strategy_name: str) -> list[str]:
    """Split a strategy expression by recognized OR separators.

    The function separates an expression like ``"a or b"`` or ``"a|b"`` into
    individual strategy tokens. Commas are intentionally **not** treated as
    separators so that parameter lists such as ``"0.05,0.10"`` remain intact.
    Whitespace around the separators is ignored. When the expression contains
    no separators, the original name is returned as the sole list element.
    """
    parts = re.split(r"\s*(?:\bor\b|\||/)\s*", strategy_name.strip())
    return [token for token in parts if token]


def _extract_sma_factor(strategy_name: str) -> float | None:
    """Extract an optional SMA window factor from a strategy name.

    Supports two formats appended to the strategy identifier:
    - Explicit suffix: "..._sma1.2" (preferred)
    - Legacy numeric:  "..._40_-0.3_10.0_1.2" (fourth numeric segment)

    Returns None when the extra factor is not present.
    """
    # Preferred explicit suffix format: ..._sma1.2
    suffix_match = re.search(r"_sma([0-9]+(?:\.[0-9]+)?)$", strategy_name)
    if suffix_match:
        try:
            return float(suffix_match.group(1))
        except ValueError:  # noqa: PERF203
            return None

    # Legacy: trailing fourth numeric segment treated as factor
    parts = strategy_name.split("_")
    numeric_segments: list[str] = []
    while parts:
        token = parts[-1]
        try:
            float(token)
        except ValueError:
            break
        numeric_segments.append(token)
        parts.pop()
    numeric_segments.reverse()
    # Expect pattern: window(int), lower(float), upper(float), factor(float)
    if len(numeric_segments) >= 4 and numeric_segments[0].isdigit():
        try:
            return float(numeric_segments[-1])
        except ValueError:  # noqa: PERF203
            return None
    return None


def rename_signal_columns(
    price_data_frame: pandas.DataFrame, base_name: str, full_name: str
) -> None:
    """Rename strategy signal columns emitted with a base identifier.

    Many strategy helpers emit standard column names such as
    ``"ema_sma_cross_entry_signal"`` that reflect the base implementation name.
    When strategies are parameterized, the caller expects the columns to use the
    fully qualified strategy name, for example ``"ema_sma_cross_10_entry_signal"``.

    This helper updates the signal columns in ``price_data_frame`` in-place when
    ``full_name`` differs from ``base_name``. Columns that are not present are
    ignored so the operation is safe to call for every strategy invocation.
    """

    if base_name == full_name:
        return

    rename_mapping: dict[str, str] = {}
    signal_suffixes = [
        "entry_signal",
        "exit_signal",
        "raw_entry_signal",
        "raw_exit_signal",
    ]
    for suffix in signal_suffixes:
        original_column_name = f"{base_name}_{suffix}"
        renamed_column_name = f"{full_name}_{suffix}"
        if (
            original_column_name in price_data_frame.columns
            and original_column_name != renamed_column_name
        ):
            rename_mapping[original_column_name] = renamed_column_name

    if rename_mapping:
        price_data_frame.rename(columns=rename_mapping, inplace=True)


def _extract_short_long_windows_for_20_50(
    strategy_name: str,
) -> tuple[int, int] | None:
    """Extract short/long SMA windows from a "20_50_sma_cross" style name.

    Supports names like ``"20_50_sma_cross_15_30"`` in which the trailing
    two integer segments override the default 20/50 windows. Returns
    ``(short, long)`` when present and valid, otherwise ``None``.
    """
    parts = strategy_name.split("_")
    if len(parts) < 2:
        return None
    try:
        long_candidate = int(parts[-1])
        short_candidate = int(parts[-2])
    except ValueError:
        return None
    if short_candidate <= 0 or long_candidate <= 0:
        return None
    if short_candidate >= long_candidate:
        return None
    return short_candidate, long_candidate


def _symbol_lookup_aliases(symbol_name: str) -> set[str]:
    """Return ticker aliases used by different data vendors for lookup only.

    Yahoo Finance stores share-class separators as dashes, while SEC-derived
    sector data commonly stores the same separator as a dot.  The raw symbol
    from each source should remain unchanged; only lookup maps should include
    both spellings.
    """

    normalized_symbol = str(symbol_name or "").strip().upper()
    if not normalized_symbol:
        return set()
    aliases = {normalized_symbol}
    if "." in normalized_symbol:
        aliases.add(normalized_symbol.replace(".", "-"))
    if "-" in normalized_symbol:
        aliases.add(normalized_symbol.replace("-", "."))
    return aliases


def _add_symbol_aliases_to_group_lookup(
    symbol_to_group_lookup: dict[str, int],
    symbol_name: str,
    group_identifier: int,
) -> None:
    """Add exact and vendor-alias ticker spellings to an FF12 lookup."""

    for symbol_alias in _symbol_lookup_aliases(symbol_name):
        symbol_to_group_lookup.setdefault(symbol_alias, group_identifier)


def _expand_symbols_with_lookup_aliases(symbol_names: set[str]) -> set[str]:
    """Return symbols plus vendor-specific dot/dash aliases for lookup sets."""

    expanded_symbols: set[str] = set()
    for symbol_name in symbol_names:
        expanded_symbols.update(_symbol_lookup_aliases(symbol_name))
    return expanded_symbols


def load_symbols_excluded_by_industry() -> set[str]:
    """Return symbols that should be excluded as non-stock instruments.

    When the sector classification dataset is available (``data/symbols_with_sector``),
    exclude known non-common-stock SIC codes.  FF12 group 12 ("Other") is not
    excluded by itself because many valid common stocks map outside the 11
    Fama–French operating-industry buckets.
    """
    excluded_symbols: set[str] = set()
    try:
        # Import lazily to avoid hard dependency at import time if unused.
        from stock_indicator.sector_pipeline.config import (
            DEFAULT_OUTPUT_PARQUET_PATH,
            DEFAULT_OUTPUT_CSV_PATH,
        )
        from stock_indicator.sector_pipeline.overrides import (
            SECTOR_OVERRIDES_CSV_PATH,
        )
    except Exception:  # noqa: BLE001
        return excluded_symbols
    # Prefer Parquet for speed, fall back to CSV if needed.
    try:
        if DEFAULT_OUTPUT_PARQUET_PATH.exists():
            sector_frame = pandas.read_parquet(DEFAULT_OUTPUT_PARQUET_PATH)
        elif DEFAULT_OUTPUT_CSV_PATH is not None and DEFAULT_OUTPUT_CSV_PATH.exists():
            sector_frame = pandas.read_csv(DEFAULT_OUTPUT_CSV_PATH)
        else:
            return excluded_symbols
    except Exception:  # noqa: BLE001
        return excluded_symbols
    # Normalize expected columns and filter non-stock SIC codes.
    sector_frame.columns = [str(c).strip().lower() for c in sector_frame.columns]
    if "ticker" not in sector_frame.columns:
        excluded_symbols = set()
    else:
        # Exclude SIC codes that represent non-stock instruments:
        # 6221 = Commodity/crypto ETFs (GLD, SLV, USO, IBIT, GBTC, ...)
        # 6770 = SPACs / blank-check companies
        _excluded_sic_codes = {6221, 6770}
        if "sic" in sector_frame.columns:
            mask_non_stock = sector_frame["sic"].isin(_excluded_sic_codes)
            tickers_series = (
                sector_frame.loc[mask_non_stock, "ticker"].dropna().astype(str)
            )
            excluded_symbols = _expand_symbols_with_lookup_aliases(
                set(tickers_series.str.upper().tolist())
            )
        else:
            excluded_symbols = set()

    # Merge in any manual overrides that mark symbols as FF12=12
    try:
        import pandas as pd

        if SECTOR_OVERRIDES_CSV_PATH.exists():
            overrides = pd.read_csv(SECTOR_OVERRIDES_CSV_PATH)
            overrides.columns = [str(c).strip().lower() for c in overrides.columns]
            if "ticker" in overrides.columns and "ff12" in overrides.columns:
                other_overrides = overrides[overrides["ff12"] == 12]
                override_symbols = (
                    other_overrides["ticker"].dropna().astype(str).str.upper().tolist()
                )
                excluded_symbols.update(
                    _expand_symbols_with_lookup_aliases(set(override_symbols))
                )
    except Exception:  # noqa: BLE001
        # If overrides cannot be read, proceed with what we have
        pass
    return excluded_symbols


def load_ff12_groups_by_symbol() -> dict[str, int]:
    """Return a lookup mapping ticker symbol (uppercased) to FF12 group id.

    Only returns mappings for symbols explicitly tagged with an FF12 group in
    the sector classification output. Symbols labeled as ``Other`` (12) are
    excluded from the mapping since they are not considered for trading.

    If the sector dataset is unavailable or lacks expected columns, an empty
    mapping is returned and the caller should fall back to non-grouped logic.
    """
    try:
        from stock_indicator.sector_pipeline.config import (
            DEFAULT_OUTPUT_PARQUET_PATH,
            DEFAULT_OUTPUT_CSV_PATH,
        )
    except Exception:  # noqa: BLE001
        return {}
    try:
        if DEFAULT_OUTPUT_PARQUET_PATH.exists():
            sector_frame = pandas.read_parquet(DEFAULT_OUTPUT_PARQUET_PATH)
        elif DEFAULT_OUTPUT_CSV_PATH is not None and DEFAULT_OUTPUT_CSV_PATH.exists():
            sector_frame = pandas.read_csv(DEFAULT_OUTPUT_CSV_PATH)
        else:
            return {}
    except Exception:  # noqa: BLE001
        return {}
    sector_frame.columns = [str(c).strip().lower() for c in sector_frame.columns]
    if "ticker" not in sector_frame.columns or "ff12" not in sector_frame.columns:
        return {}
    sector_frame = sector_frame.dropna(subset=["ticker", "ff12"])  # type: ignore[arg-type]
    sector_frame = sector_frame[sector_frame["ff12"] != 12]
    symbol_to_group: dict[str, int] = {}
    for sector_row in sector_frame.itertuples(index=False):
        _add_symbol_aliases_to_group_lookup(
            symbol_to_group,
            str(sector_row.ticker),
            int(sector_row.ff12),
        )
    return symbol_to_group


def _build_eligibility_mask(
    merged_volume_frame: pandas.DataFrame,
    *,
    minimum_average_dollar_volume: float | None,
    top_dollar_volume_rank: int | None,
    minimum_average_dollar_volume_ratio: float | None,
    maximum_symbols_per_group: int = 1,
) -> pandas.DataFrame:
    """Return a mask of symbols eligible for trading.

    Parameters
    ----------
    merged_volume_frame:
        DataFrame of dollar-volume averages with dates as index and symbols
        as columns.
    minimum_average_dollar_volume:
        Minimum 50-day average dollar volume threshold in millions.
    top_dollar_volume_rank:
        Global Top-N rank applied after other filters.
    minimum_average_dollar_volume_ratio:
        Minimum ratio of the total market dollar volume. When Fama–French
        groups are available, this ratio is applied within each group.
    maximum_symbols_per_group:
        Maximum number of symbols to select per Fama–French group when
        ``top_dollar_volume_rank`` is specified. Defaults to one to preserve
        the previous behavior.

    Returns
    -------
    pandas.DataFrame
        Boolean mask aligned to ``merged_volume_frame``.
    """
    # TODO: review

    if merged_volume_frame.empty:
        return pandas.DataFrame()

    if (
        minimum_average_dollar_volume is None
        and top_dollar_volume_rank is None
        and minimum_average_dollar_volume_ratio is None
    ):
        return pandas.DataFrame(
            True,
            index=merged_volume_frame.index,
            columns=merged_volume_frame.columns,
        )

    eligibility_mask = ~merged_volume_frame.isna()
    if minimum_average_dollar_volume is not None:
        eligibility_mask &= (
            merged_volume_frame / 1_000_000 >= minimum_average_dollar_volume
        )

    symbol_to_fama_french_group_id = load_ff12_groups_by_symbol()
    group_id_to_symbol_columns: dict[int, List[str]] = {}
    if symbol_to_fama_french_group_id:
        for column_name in merged_volume_frame.columns:
            group_identifier = symbol_to_fama_french_group_id.get(
                column_name.upper()
            )
            if group_identifier is None:
                continue
            group_id_to_symbol_columns.setdefault(group_identifier, []).append(
                column_name
            )

    if minimum_average_dollar_volume_ratio is not None:
        if group_id_to_symbol_columns:
            ratio_eligibility_mask = pandas.DataFrame(
                False,
                index=merged_volume_frame.index,
                columns=merged_volume_frame.columns,
            )
            market_total_series = merged_volume_frame.sum(axis=1)
            for group_identifier, column_list in group_id_to_symbol_columns.items():
                group_frame = merged_volume_frame[column_list]
                group_total_series = group_frame.sum(axis=1)
                safe_group_total_series = group_total_series.where(
                    group_total_series > 0
                )
                safe_market_total_series = market_total_series.where(
                    market_total_series > 0
                )
                group_share_series = safe_group_total_series.divide(
                    safe_market_total_series
                )
                dynamic_threshold_series = (
                    minimum_average_dollar_volume_ratio / group_share_series
                )
                ratio_frame = group_frame.divide(
                    safe_group_total_series, axis=0
                )
                ratio_condition = ratio_frame.ge(dynamic_threshold_series, axis=0)
                ratio_eligibility_mask.loc[:, column_list] = ratio_condition
            eligibility_mask &= ratio_eligibility_mask
        else:
            total_volume_series = merged_volume_frame.sum(axis=1)
            ratio_frame = merged_volume_frame.divide(
                total_volume_series.where(total_volume_series > 0), axis=0
            )
            eligibility_mask &= ratio_frame >= minimum_average_dollar_volume_ratio

    if top_dollar_volume_rank is not None:
        if group_id_to_symbol_columns:
            selected_mask = pandas.DataFrame(
                False,
                index=merged_volume_frame.index,
                columns=merged_volume_frame.columns,
            )
            symbol_to_group_lookup = {
                symbol: symbol_to_fama_french_group_id.get(symbol.upper())
                for symbol in merged_volume_frame.columns
            }
            for current_date in merged_volume_frame.index:
                candidate_values = merged_volume_frame.loc[current_date].where(
                    eligibility_mask.loc[current_date], other=pandas.NA
                ).dropna()
                if candidate_values.empty:
                    continue
                sorted_symbols = candidate_values.sort_values(
                    ascending=False
                ).index.tolist()
                chosen_symbols: list[str] = []
                group_counts: dict[int, int] = {}
                for symbol_name in sorted_symbols:
                    group_identifier = symbol_to_group_lookup.get(symbol_name)
                    if group_identifier is None:
                        continue
                    current_count = group_counts.get(group_identifier, 0)
                    if current_count >= maximum_symbols_per_group:
                        continue
                    chosen_symbols.append(symbol_name)
                    group_counts[group_identifier] = current_count + 1
                    if len(chosen_symbols) >= int(top_dollar_volume_rank):
                        break
                if chosen_symbols:
                    selected_mask.loc[current_date, chosen_symbols] = True
            eligibility_mask &= selected_mask
        else:
            rank_frame = merged_volume_frame.rank(
                axis=1, method="min", ascending=False
            )
            eligibility_mask &= rank_frame <= top_dollar_volume_rank

    return eligibility_mask


# Number of days used for moving averages.
LONG_TERM_SMA_WINDOW: int = 150
DOLLAR_VOLUME_SMA_WINDOW: int = 50


@dataclass
class TradeDetail:
    """Represent a single trade event for reporting purposes.

    The dollar volume fields record the latest 50-day simple moving average
    dollar volume used when selecting symbols. The ratio expresses this
    symbol's share of the summed average dollar volume across the entire
    market, not just the eligible subset.

    Chip concentration metrics record characteristics of the volume
    distribution around the entry price. ``price_concentration_score`` is a
    normalized Herfindahl index of the price-volume histogram. The
    ``near_price_volume_ratio`` and ``above_price_volume_ratio`` values capture
    the fractions of volume near and above the entry price. ``histogram_node_count``
    approximates the number of significant volume clusters. The ``sma_angle``
    value stores the simple moving average slope, expressed in degrees, that was
    present on the signal date that triggered the entry. The ``result`` field
    marks whether a closed trade ended in a win or a loss. For closing trades,
    ``percentage_change`` records the fractional price change between entry and
    exit. The ``exit_reason`` field captures why a trade closed, such as
    ``"signal"``, ``"stop_loss"``, ``"take_profit"``, or ``"end_of_data"``.
    """
    # TODO: review
    date: pandas.Timestamp
    symbol: str
    action: str
    price: float
    simple_moving_average_dollar_volume: float
    total_simple_moving_average_dollar_volume: float
    simple_moving_average_dollar_volume_ratio: float
    # Group-aware metrics: totals and ratios computed within the symbol's FF12 group
    group_total_simple_moving_average_dollar_volume: float = 0.0
    group_simple_moving_average_dollar_volume_ratio: float = 0.0
    # Chip concentration metrics calculated at trade entry
    price_concentration_score: float | None = None
    near_price_volume_ratio: float | None = None
    above_price_volume_ratio: float | None = None
    below_price_volume_ratio: float | None = None
    near_delta: float | None = None
    price_tightness: float | None = None
    histogram_node_count: int | None = None
    sma_angle: float | None = None
    d_sma_angle: float | None = None
    ema_angle: float | None = None
    d_ema_angle: float | None = None
    # 60-bar slope at signal date (T): (close[T] - close[T-59]) / close[T-59].
    # Captures the trade's macro slope context — fish_head trades fire on
    # negative slope (post-vacuum), fish_tail on extreme positive (post-rally),
    # fish_body across the middle. Useful for slope-based regime filtering.
    slope_60: float | None = None
    # B-layer confirmation-day (T+1) sma_angle value. Recorded alongside the
    # signal-date (T) ``sma_angle`` so you can inspect what the confirmation
    # gate actually saw for each trade.
    sma_angle_confirmation: float | None = None
    # Signal bar open price (T+1 open) used as risk reference for SL/TP.
    # Only populated when pending market entry is used.
    signal_bar_open: float | None = None
    # Number of concurrent open positions at this event.
    # For an "open" event, includes this position. For a "close" event,
    # excludes the position being closed.
    concurrent_position_count: int = 0
    # Number of concurrent open positions across all strategy sets at the time of
    # this trade detail event. This value is populated during complex
    # simulations where multiple sets share a global position limit.
    global_concurrent_position_count: int | None = None
    strategy_set_label: str | None = None
    result: str | None = None  # TODO: review
    percentage_change: float | None = None  # TODO: review
    exit_reason: str = "signal"
    total_commission: float | None = None
    share_count: int | None = None
    # Intra-trade excursion statistics, populated on the exit detail only.
    # Expressed as a fraction of the trade's entry_price (e.g. +0.05 = bar
    # high reached 5% above entry; -0.03 = bar low reached 3% below entry).
    max_favorable_excursion_pct: float | None = None
    max_adverse_excursion_pct: float | None = None
    max_favorable_excursion_date: pandas.Timestamp | None = None
    max_adverse_excursion_date: pandas.Timestamp | None = None
    # Adaptive TP/SL percentages applied to this trade (exit detail only).
    adaptive_tp_pct: float | None = None
    adaptive_sl_pct: float | None = None


@dataclass
class StrategyMetrics:
    """Aggregate metrics describing strategy performance."""
    # TODO: review

    total_trades: int
    win_rate: float
    mean_profit_percentage: float
    profit_percentage_standard_deviation: float
    mean_loss_percentage: float
    loss_percentage_standard_deviation: float
    mean_holding_period: float
    holding_period_standard_deviation: float
    maximum_concurrent_positions: int
    maximum_drawdown: float
    final_balance: float
    compound_annual_growth_rate: float
    annual_returns: Dict[int, float]
    annual_trade_counts: Dict[int, int]
    trade_details_by_year: Dict[int, List[TradeDetail]] = field(default_factory=dict)


@dataclass
class ComplexStrategySetDefinition:
    """Configuration for a strategy set used in complex simulations."""

    label: str
    buy_strategy_name: str
    sell_strategy_name: str
    strategy_identifier: str | None = None
    stop_loss_percentage: float = 1.0
    take_profit_percentage: float = 0.0
    minimum_average_dollar_volume: float | None = None
    minimum_average_dollar_volume_ratio: float | None = None
    top_dollar_volume_rank: int | None = None
    maximum_symbols_per_group: int = 1
    d_sma_range: tuple[float, float] | None = None
    ema_range: tuple[float, float] | None = None
    d_ema_range: tuple[float, float] | None = None
    near_delta_range: tuple[float, float] | None = None
    price_tightness_range: tuple[float, float] | None = None
    sma_150_angle_min: float | None = None
    use_ftd_confirmation: bool = False
    trailing_stop_percentage: float = 0.0
    price_score_min: float | None = None
    price_score_max: float | None = None
    exit_alpha_factor: float | None = None
    # Fish-body shape + BSV gate (third instance — trend join after vacuum
    # turn). All three set → entry signal additionally requires:
    # 1) 60-bar shape slope >= shape_slope_min
    # 2) ALL middle samples (25%, 50%, 75%) deviation <= shape_dev_50_max
    #    (full concave U-shape: every interior point at-or-below baseline)
    # 3) BSV footprint observed in last shape_bsv_lookback bars
    shape_slope_min: float | None = None
    shape_dev_50_max: float | None = None
    shape_bsv_lookback: int | None = None
    # Per-bucket override for adaptive_tp_sl.tp_regime_adjust. When None, the
    # bucket inherits the top-level adaptive setting; when True/False it
    # overrides it for this bucket only. Used to enable TP regime scaling on
    # one bucket (e.g. break_high momentum) while keeping another static
    # (e.g. buy3 V-bottom needs wide TP for fat-tail rebounds).
    tp_regime_adjust: bool | None = None
    # Per-bucket override for adaptive_tp_sl.fixed_tp / fixed_sl. When None,
    # bucket inherits the top-level value (which may itself be None). Allows
    # per-bucket tight static TP/SL when each strategy has its own optimal
    # tight stops.
    fixed_tp: float | None = None
    fixed_sl: float | None = None
    # Per-bucket override for adaptive_tp_sl.min_sl. fixed_sl alone caps SL
    # from above; without raising min_sl, a bucket cannot widen SL beyond the
    # shared floor. Setting min_sl per bucket lets each instance configure its
    # own SL floor independently.
    min_sl: float | None = None
    # Per-bucket bounds on entry slope_60 (60-bar slope at signal date).
    # When set, candidates outside the bounds are skipped at the trade-
    # decision step. slope_max excludes blow-off territory (e.g. slope >
    # 50%); slope_min excludes deep declines.
    slope_max: float | None = None
    slope_min: float | None = None
    # Compound free-fall filter for fish_head: skip when BOTH
    # slope_60 < free_fall_slope AND near_delta < free_fall_near_delta.
    # Targets the diagnostic toxic cell where deep crash combines with
    # absent nearby volume floor (no-bid territory) — distinct from
    # slope_min which is unconditional. Both fields must be set together
    # to activate; if either is None, filter does not fire.
    free_fall_slope: float | None = None
    free_fall_near_delta: float | None = None
    # Per-bucket override for min_hold gates on TP and SL. None = inherit
    # top-level. Needed because different bucket narratives demand opposite
    # SL gate behavior — buy3 V-bottom needs SL to wait for min_hold (give
    # recovery time), break_high momentum needs SL to fire fast (no recovery
    # to wait for, gap-down at bar 5 makes SL exit much worse than intended).
    override_min_hold_tp_only: bool | None = None
    min_hold_tp: int | None = None
    override_min_hold_sl_only: bool | None = None
    min_hold_sl: int | None = None
    # Multi-bucket extensions (ignored by run_complex_simulation A/B path):
    # - entry_priority: lower number = wins entry contention first (used by
    #   run_multi_bucket_simulation as a tiebreaker in the event sort key).
    # - maximum_positions: per-bucket position cap; when None, the bucket is
    #   only limited by the global maximum_position_count.
    # - fill_remaining: when True, this bucket can only open when the TOTAL
    #   open position count (across all buckets) is below its maximum_positions.
    #   This mimics the complex simulation B-set behaviour where B fills
    #   positions that A didn't use, rather than having its own independent cap.
    entry_priority: int = 0
    maximum_positions: int | None = None
    fill_remaining: bool = False


@dataclass
class AdaptiveTPSLConfig:
    """Configuration for adaptive (rolling) take-profit and stop-loss.

    TP = rolling_MP + sigma_multiplier * rolling_σ_profit.
    SL sensor = median rolling raw signal loss, capped by fixed_sl when set.

    TP/SL rolling uses accepted signal history only.  Sector-specific rolling
    was removed because repeated tests showed signal-based rolling is the
    useful mechanism.
    """

    window: int = 20
    sigma_multiplier: float = 0.5
    # Backward-compatible config field. Rolling SL is no longer derived from
    # TP / target_r; it is kept so older JSON files still parse.
    target_r: float = 2.0
    # Backward-compatible config field. Rolling SL now uses median raw signal
    # loss as a robust regime sensor, so this field is no longer applied.
    sl_sigma_multiplier: float | None = None
    min_tp: float = 0.02
    min_sl: float = 0.01
    min_samples: int = 5
    # When set, SL is fixed at this value (TP remains adaptive).
    fixed_sl: float | None = None
    # When set, TP is fixed at this value (overrides rolling TP and regime
    # adjustment). Mirror of fixed_sl. Use both to run a fully static
    # TP/SL configuration without removing the adaptive_tp_sl block.
    fixed_tp: float | None = None
    # When True, adaptive TP/SL can trigger before minimum_holding_bars.
    override_min_hold: bool = False
    # When True, TP uses min_hold_tp instead of the global min_hold.
    override_min_hold_tp_only: bool = False
    # Minimum bars before TP can trigger (when override_min_hold_tp_only=True).
    # 0 = immediate, 1 = realistic (T+2 morning TP known, place order intraday).
    min_hold_tp: int = 0
    # When True, SL uses min_hold_sl instead of the global min_hold.
    # SL is risk control, not STATE confirmation, so it should not inherit the
    # signal-exit min_hold latency.
    override_min_hold_sl_only: bool = False
    # Minimum bars before SL can trigger (when override_min_hold_sl_only=True).
    # 0 = immediate, 1 = realistic (T+2 morning fill).
    min_hold_sl: int = 0
    # When True, SL is computed (rolling sl_pct as regime indicator) but never
    # fires as an exit. Used for dynamic min_hold throttle: rolling sl_pct/tp_pct
    # ratio drives slot lock duration, but trades exit only via TP or signal.
    # Avoids recovery-kill while preserving SL as regime sensor.
    disable_sl_trigger: bool = False
    # When True, scale the rolling TP target by (tp_pct / sl_pct) capped to
    # [tp_regime_ratio_min, tp_regime_ratio_max]. Calm regime (tp >> sl) widens
    # TP; stress regime (sl approaches or exceeds tp) shrinks TP so the smaller
    # stress-regime winners lock in before mean-reversion eats them.
    tp_regime_adjust: bool = False
    tp_regime_ratio_min: float = 0.5
    tp_regime_ratio_max: float = 1.5
    # When True, rolling stats only include trades that closed BEFORE the
    # current entry date (not same-day closes). This lets TP% be computed
    # on entry signal night (T), so TP orders can be placed at T+1 open.
    delayed_rolling_update: bool = False
    # When True, SL is tightened to break-even (entry price) once unrealized
    # profit reaches the rolling mean profit (MP).
    breakeven_at_mp: bool = False
    # When True, no TP or signal exit. New entry signals evict the oldest
    # open position (past min_hold) to free a slot. Exit price = eviction
    # day open of the evicted symbol.
    evict_oldest: bool = False


@dataclass
class ComplexSimulationMetrics:
    """Aggregate metrics for multiple strategy sets."""

    overall_metrics: StrategyMetrics
    metrics_by_set: Dict[str, StrategyMetrics]


@dataclass
class StrategyEvaluationArtifacts:
    """Container for intermediate results produced during strategy evaluation."""

    trades: List[Trade]
    simulation_results: List[SimulationResult]
    trade_symbol_lookup: Dict[Trade, str]
    closing_price_series_by_symbol: Dict[str, pandas.Series]
    trade_detail_pairs: Dict[Trade, tuple[TradeDetail, TradeDetail]]
    simulation_start_date: pandas.Timestamp | None


def _resolve_slot_release_date(
    trade: Trade,
    target_holding_bars: int,
    symbol: str,
    closing_price_series_by_symbol: Dict[str, pandas.Series],
) -> tuple[
    pandas.Timestamp,
    int,
    list[tuple[pandas.Timestamp, float, float, float]] | None,
]:
    """Return the date for a target holding period, extending past raw exit.

    ``Trade.bar_excursions`` only covers the raw trade lifetime.  Dynamic slot
    throttling can intentionally lock a slot after a signal/SL exit, so it must
    fall back to the symbol's trading calendar when the target bar is beyond
    the raw trade's excursion path.
    """

    if trade.bar_excursions and target_holding_bars <= len(trade.bar_excursions):
        target_index = target_holding_bars - 1
        return (
            trade.bar_excursions[target_index][0],
            target_holding_bars,
            trade.bar_excursions[:target_holding_bars],
        )

    price_series = closing_price_series_by_symbol.get(symbol)
    if price_series is not None and not price_series.empty:
        future_dates = [
            pandas.Timestamp(price_date)
            for price_date in price_series.index
            if pandas.Timestamp(price_date) > trade.entry_date
        ]
        future_dates.sort()
        if target_holding_bars <= len(future_dates):
            return (
                future_dates[target_holding_bars - 1],
                target_holding_bars,
                trade.bar_excursions,
            )

    if target_holding_bars <= trade.holding_period:
        return trade.exit_date, trade.holding_period, trade.bar_excursions

    return trade.exit_date, trade.holding_period, trade.bar_excursions


def _replay_trade_with_adaptive_tp_sl(
    trade: Trade,
    tp_pct: float,
    sl_pct: float,
    minimum_holding_bars: int = 0,
    minimum_holding_bars_tp: int | None = None,
    minimum_holding_bars_sl: int | None = None,
    breakeven_trigger_pct: float = 0.0,
    tp_pct_late: float | None = None,
    sl_only: bool = False,
    disable_sl_trigger: bool = False,
) -> Trade:
    """Replay a raw trade using adaptive TP/SL levels.

    Walks the trade's bar_excursions to find if TP or SL triggers before
    the signal-based exit.  Returns a new Trade with adjusted exit if
    triggered, or the original trade unchanged.

    *minimum_holding_bars* is the signal-exit min_hold (used as the default
    for TP and SL checks, and for the tp_pct_late switch).

    When *minimum_holding_bars_tp* is provided it overrides
    *minimum_holding_bars* for TP checks only, allowing TP to trigger
    earlier than the signal-exit min_hold.

    When *minimum_holding_bars_sl* is provided it overrides
    *minimum_holding_bars* for SL checks only, allowing SL to trigger
    earlier than the signal-exit min_hold (risk control should not be
    delayed by STATE-confirmation latency).

    When *breakeven_trigger_pct* > 0, once the bar high reaches that level
    the SL is tightened to break-even (entry price).

    When *tp_pct_late* is provided, TP switches from *tp_pct* to
    *tp_pct_late* once holding >= *minimum_holding_bars*.  This allows
    a wider (sector) TP during the catalyst window and a tighter
    (uniform) TP after min_hold.
    """
    if trade.bar_excursions is None or not trade.bar_excursions:
        return trade
    effective_min_hold_tp = (
        minimum_holding_bars_tp if minimum_holding_bars_tp is not None
        else minimum_holding_bars
    )
    effective_min_hold_sl = (
        minimum_holding_bars_sl if minimum_holding_bars_sl is not None
        else minimum_holding_bars
    )
    active_sl_pct = sl_pct
    for bar_index, excursion in enumerate(trade.bar_excursions):
        # Support both 3-tuple (legacy) and 4-tuple (with open_pct).
        if len(excursion) == 4:
            bar_date, bar_high_pct, bar_low_pct, bar_open_pct = excursion
        else:
            bar_date, bar_high_pct, bar_low_pct = excursion[:3]
            bar_open_pct = 0.0
        holding = bar_index + 1
        # Upgrade SL to break-even once profit reaches trigger level
        if (
            breakeven_trigger_pct > 0
            and bar_high_pct >= breakeven_trigger_pct
        ):
            active_sl_pct = 0.0  # break-even = entry price
        # Check SL (respects effective_min_hold_sl, decoupled from signal min_hold).
        # When disable_sl_trigger=True, SL value used as regime indicator only,
        # never fires as exit — skip the check entirely.
        if (
            not disable_sl_trigger
            and holding >= effective_min_hold_sl
            and bar_low_pct <= -active_sl_pct
        ):
            effective_sl_pct = active_sl_pct
            if bar_open_pct <= -active_sl_pct:
                effective_sl_pct = -bar_open_pct
            adjusted_exit_price = trade.entry_price * (1 - effective_sl_pct)
            adjusted_profit = adjusted_exit_price - trade.entry_price
            return replace(
                trade,
                exit_date=bar_date,
                exit_price=adjusted_exit_price,
                profit=adjusted_profit,
                holding_period=holding,
                exit_reason="adaptive_stop_loss",
                bar_excursions=trade.bar_excursions[: holding],
            )
        # Check TP (may use different min_hold).
        # If open gaps above TP, limit order fills at open (price improvement).
        # Switch to tp_pct_late after minimum_holding_bars if provided.
        if not sl_only:
            active_tp_pct = tp_pct
            if tp_pct_late is not None and holding >= minimum_holding_bars:
                active_tp_pct = tp_pct_late
            if (
                holding >= effective_min_hold_tp
                and active_tp_pct > 0
                and bar_high_pct >= active_tp_pct
            ):
                effective_tp_pct = max(active_tp_pct, bar_open_pct)
                adjusted_exit_price = trade.entry_price * (1 + effective_tp_pct)
                adjusted_profit = adjusted_exit_price - trade.entry_price
                return replace(
                    trade,
                    exit_date=bar_date,
                    exit_price=adjusted_exit_price,
                    profit=adjusted_profit,
                    holding_period=holding,
                    exit_reason="adaptive_take_profit",
                    bar_excursions=trade.bar_excursions[: holding],
                )
    if sl_only:
        # No SL triggered — hold indefinitely until evicted by a new signal.
        # Set exit_date far into the future so the trade stays open in the
        # event loop. Eviction or true end-of-data will override.
        far_future = pandas.Timestamp("2099-12-31")
        last = trade.bar_excursions[-1]
        # Use last bar's close (signal exit price) as fallback exit price;
        # eviction will replace this with the eviction-day open.
        adjusted_exit_price = trade.exit_price
        adjusted_profit = adjusted_exit_price - trade.entry_price
        return replace(
            trade,
            exit_date=far_future,
            exit_price=adjusted_exit_price,
            profit=adjusted_profit,
            holding_period=len(trade.bar_excursions),
            exit_reason="end_of_data",
            bar_excursions=trade.bar_excursions,
        )
    # Neither triggered — use signal exit as-is.
    return trade


def run_complex_simulation(
    data_directory: Path,
    set_definitions: Dict[str, ComplexStrategySetDefinition],
    *,
    maximum_position_count: int,
    starting_cash: float = 3000.0,
    withdraw_amount: float = 0.0,
    start_date: pandas.Timestamp | None = None,
    margin_multiplier: float = 1.0,
    margin_interest_annual_rate: float = 0.048,
    use_confirmation_angle: bool = False,
    confirmation_entry_mode: str = "limit",
    minimum_holding_bars: int = 0,
    multi_bucket_mode: bool = False,
    confirmation_sma_angle_range: tuple[float, float] | None = None,
    adaptive_tp_sl: AdaptiveTPSLConfig | None = None,
    max_same_symbol: int = 1,
    allowed_symbols: set[str] | None = None,
) -> ComplexSimulationMetrics:
    """Evaluate multiple strategy sets under a shared configuration.

    When ``multi_bucket_mode`` is ``False`` (default), the A/B legacy
    semantics apply: set B is automatically capped at half of the global
    ``maximum_position_count`` and is blocked from opening when the total
    open positions already meet B's cap. This matches the original
    complex_simulation behaviour.

    When ``multi_bucket_mode`` is ``True``, each set may declare its own
    ``maximum_positions`` (via ``ComplexStrategySetDefinition``) and an
    ``entry_priority`` integer (lower = higher priority) that breaks ties in
    the event sort order. The B-specific hardcoding is disabled so any number
    of buckets can share a global slot pool on a first-come-first-served
    basis, with priority as the secondary ordering key.
    """

    if maximum_position_count <= 0:
        raise ValueError("maximum_position_count must be positive")
    if not set_definitions:
        raise ValueError("set_definitions must not be empty")

    effective_interest_rate = (
        margin_interest_annual_rate if margin_multiplier != 1.0 else 0.0
    )
    artifacts_by_set: Dict[str, StrategyEvaluationArtifacts] = {}
    position_limits_by_set: Dict[str, int] = {}
    priority_mode_by_set: Dict[str, str] = {}
    for label, definition in set_definitions.items():
        if multi_bucket_mode:
            maximum_positions_for_set = (
                definition.maximum_positions
                if definition.maximum_positions is not None
                else maximum_position_count
            )
        else:
            maximum_positions_for_set = maximum_position_count
            if label.upper() == "B":
                maximum_positions_for_set = max(
                    1, math.ceil(maximum_position_count / 2)
                )
        position_limits_by_set[label] = maximum_positions_for_set
        strategy_identifier = (
            definition.strategy_identifier.lower()
            if definition.strategy_identifier
            else None
        )
        if strategy_identifier in {"s4", "s6"}:
            priority_mode_by_set[label] = strategy_identifier
        artifacts_by_set[label] = _generate_strategy_evaluation_artifacts(
            data_directory,
            definition.buy_strategy_name,
            definition.sell_strategy_name,
            minimum_average_dollar_volume=definition.minimum_average_dollar_volume,
            top_dollar_volume_rank=definition.top_dollar_volume_rank,
            maximum_symbols_per_group=definition.maximum_symbols_per_group,
            minimum_average_dollar_volume_ratio=
                definition.minimum_average_dollar_volume_ratio,
            start_date=start_date,
            maximum_position_count=maximum_positions_for_set,
            allowed_fama_french_groups=None,
            allowed_symbols=allowed_symbols,
            exclude_other_ff12=True,
            stop_loss_percentage=(
                1.0 if adaptive_tp_sl is not None
                else definition.stop_loss_percentage
            ),
            take_profit_percentage=(
                0.0 if adaptive_tp_sl is not None
                else definition.take_profit_percentage
            ),
            minimum_holding_bars=minimum_holding_bars,
            use_confirmation_angle=use_confirmation_angle,
            confirmation_entry_mode=confirmation_entry_mode,
            margin_multiplier=margin_multiplier,
            margin_interest_annual_rate=effective_interest_rate,
            d_sma_range=definition.d_sma_range,
            ema_range=definition.ema_range,
            d_ema_range=definition.d_ema_range,
            near_delta_range=definition.near_delta_range,
            price_tightness_range=definition.price_tightness_range,
            sma_150_angle_min=definition.sma_150_angle_min,
            use_ftd_confirmation=definition.use_ftd_confirmation,
            trailing_stop_percentage=(
                0.0 if adaptive_tp_sl is not None
                else definition.trailing_stop_percentage
            ),
            price_score_min=definition.price_score_min,
            price_score_max=definition.price_score_max,
            confirmation_sma_angle_range=confirmation_sma_angle_range,
            exit_alpha_factor=definition.exit_alpha_factor,
            shape_slope_min=definition.shape_slope_min,
            shape_dev_50_max=definition.shape_dev_50_max,
            shape_bsv_lookback=definition.shape_bsv_lookback,
            reentry_on_signal=(
                adaptive_tp_sl.evict_oldest if adaptive_tp_sl is not None else False
            ),
        )

    accepted_trades_by_set: Dict[str, List[Trade]] = {
        label: [] for label in set_definitions
    }
    open_position_counts_by_set: Dict[str, int] = {label: 0 for label in set_definitions}
    open_trade_keys: Dict[Tuple[str, int], str] = {}
    accepted_trade_keys: set[Tuple[str, int]] = set()
    # Track open positions per symbol for max_same_symbol enforcement.
    open_symbol_counts: Dict[str, int] = {}
    # Map trade_id -> symbol for decrementing on close.
    open_trade_symbols: Dict[int, str] = {}
    # Event tuple layout:
    #   (date, event_type, bucket_priority, entry_priority, insertion_counter,
    #    label, trade)
    # Sort key uses the first five fields in order, so bucket_priority wins
    # over within-bucket quality (entry_priority) and insertion order.
    events: List[
        tuple[pandas.Timestamp, int, int, float, int, str, Trade]
    ] = []
    event_insertion_counter = 0
    for label, artifacts in artifacts_by_set.items():
        priority_mode = priority_mode_by_set.get(label)
        bucket_priority_value = (
            set_definitions[label].entry_priority if multi_bucket_mode else 0
        )
        for trade in artifacts.trades:
            entry_priority = 0.0
            trade_detail_pair = artifacts.trade_detail_pairs.get(trade)
            if trade_detail_pair is not None and priority_mode is not None:
                entry_detail = trade_detail_pair[0]
                if priority_mode == "s4":
                    ratio_value = (
                        entry_detail.above_price_volume_ratio
                        if entry_detail.above_price_volume_ratio is not None
                        else float("inf")
                    )
                    entry_priority = ratio_value
                elif priority_mode == "s6":
                    ratio_value = (
                        entry_detail.near_price_volume_ratio
                        if entry_detail.near_price_volume_ratio is not None
                        else float("inf")
                    )
                    entry_priority = ratio_value
            events.append(
                (
                    trade.entry_date,
                    1,
                    bucket_priority_value,
                    entry_priority,
                    event_insertion_counter,
                    label,
                    trade,
                )
            )
            event_insertion_counter += 1
            if adaptive_tp_sl is None:
                # Non-adaptive: close events are pre-scheduled.
                events.append(
                    (
                        trade.exit_date,
                        0,
                        0,
                        0.0,
                        event_insertion_counter,
                        label,
                        trade,
                    )
                )
                event_insertion_counter += 1
    events.sort(
        key=lambda event: (event[0], event[1], event[2], event[3], event[4])
    )

    # Adaptive TP/SL state: rolling window of recently closed trades, split
    # by outcome so TP and SL each draw from their own last-N sample.
    # Each deque holds signed pct values: winners > 0 only, losers < 0 only.
    adaptive_closed_winners: deque[float] = deque()
    adaptive_closed_losers: deque[float] = deque()
    # Map from original trade id -> adjusted trade for adaptive mode.
    adaptive_trade_map: Dict[int, Trade] = {}
    # Reverse map: adjusted trade -> original trade (for detail pair lookups).
    adaptive_original_trade: Dict[int, Trade] = {}
    # TP/SL pcts applied per adjusted trade id.
    adaptive_tp_sl_applied: Dict[int, tuple[float, float]] = {}
    # Far-future adaptive trades need a real end-of-data fallback that can move
    # forward after same-symbol signal refreshes.
    adaptive_fallback_exits: Dict[int, tuple[pandas.Timestamp, float, int]] = {}
    # Slot occupancy can differ from accounting exit when SL is allowed earlier
    # than the outer min_hold.  The accounting trade records the real SL fill;
    # this map keeps the capital slot on the old min_hold-gated replay path.
    adaptive_slot_close_dates: Dict[int, pandas.Timestamp] = {}
    adaptive_slot_trade_map: Dict[int, Trade] = {}
    # Pending close events for adaptive mode (heap of
    # (close_date, counter, label, original_trade_id)).
    adaptive_close_heap: list[tuple[pandas.Timestamp, int, str, int]] = []
    adaptive_close_counter = 0
    # When delayed_rolling_update is True, closed trade pcts are buffered
    # here and only flushed into adaptive_closed_winners / adaptive_closed_losers
    # when an entry event on a LATER date is processed.
    pending_rolling_updates: list[tuple[pandas.Timestamp, float]] = []
    # For evict_oldest: track the latest active signal date for each open
    # trade.  This is intentionally separate from Trade.entry_date: accounting
    # and holding-period statistics keep the original entry date, while
    # eviction/min-hold decisions use the most recent same-symbol signal.
    open_trade_entry_dates: Dict[int, pandas.Timestamp] = {}

    if adaptive_tp_sl is not None:
        use_evict_oldest = adaptive_tp_sl.evict_oldest
        # Use heap-based event processing for adaptive mode.
        # Convert sorted entry events into a deque for efficient popping.
        entry_events = deque(events)
        events = []  # free memory
        # Track same-day close releases so entries cannot use slots freed
        # on the same date (no lookahead — you can't know at entry time
        # whether a TP/SL will trigger later that day).
        same_day_close_count = 0
        last_close_date: pandas.Timestamp | None = None

        _far_future = pandas.Timestamp("2099-12-31")

        if use_evict_oldest:

            def _convert_bar_excursions_to_entry_basis(
                segment_trade: Trade,
                position_entry_price: float,
            ) -> list[tuple[pandas.Timestamp, float, float, float]]:
                """Convert segment excursions to the original position basis."""

                converted_excursions: list[
                    tuple[pandas.Timestamp, float, float, float]
                ] = []
                if not segment_trade.bar_excursions or position_entry_price <= 0:
                    return converted_excursions
                for excursion in segment_trade.bar_excursions:
                    bar_date = excursion[0]
                    bar_high_percentage = excursion[1]
                    bar_low_percentage = excursion[2]
                    bar_open_percentage = (
                        excursion[3] if len(excursion) == 4 else 0.0
                    )
                    bar_high_price = segment_trade.entry_price * (
                        1 + bar_high_percentage
                    )
                    bar_low_price = segment_trade.entry_price * (
                        1 + bar_low_percentage
                    )
                    bar_open_price = segment_trade.entry_price * (
                        1 + bar_open_percentage
                    )
                    converted_excursions.append(
                        (
                            bar_date,
                            (bar_high_price - position_entry_price)
                            / position_entry_price,
                            (bar_low_price - position_entry_price)
                            / position_entry_price,
                            (bar_open_price - position_entry_price)
                            / position_entry_price,
                        )
                    )
                return converted_excursions

            def _excursion_extremes(
                bar_excursions: list[
                    tuple[pandas.Timestamp, float, float, float]
                ],
            ) -> tuple[
                float | None,
                float | None,
                pandas.Timestamp | None,
                pandas.Timestamp | None,
            ]:
                """Return MFE/MAE percentages and dates from entry-basis excursions."""

                favorable_percentage: float | None = None
                adverse_percentage: float | None = None
                favorable_date: pandas.Timestamp | None = None
                adverse_date: pandas.Timestamp | None = None
                for (
                    bar_date,
                    high_percentage,
                    low_percentage,
                    _open_percentage,
                ) in bar_excursions:
                    if (
                        favorable_percentage is None
                        or high_percentage > favorable_percentage
                    ):
                        favorable_percentage = high_percentage
                        favorable_date = bar_date
                    if (
                        adverse_percentage is None
                        or low_percentage < adverse_percentage
                    ):
                        adverse_percentage = low_percentage
                        adverse_date = bar_date
                return (
                    favorable_percentage,
                    adverse_percentage,
                    favorable_date,
                    adverse_date,
                )

            def _refresh_open_trade_from_same_symbol_signal(
                *,
                open_trade_identifier: int,
                refresh_trade: Trade,
            ) -> None:
                """Refresh an open same-symbol position without closing/re-opening."""

                nonlocal adaptive_close_counter
                current_adjusted_trade = adaptive_trade_map.get(
                    open_trade_identifier
                )
                if current_adjusted_trade is None:
                    return
                current_original_trade = adaptive_original_trade.get(
                    id(current_adjusted_trade),
                    current_adjusted_trade,
                )
                existing_excursions = list(
                    current_adjusted_trade.bar_excursions or []
                )
                converted_refresh_excursions = (
                    _convert_bar_excursions_to_entry_basis(
                        refresh_trade,
                        current_adjusted_trade.entry_price,
                    )
                )
                if existing_excursions:
                    last_existing_date = existing_excursions[-1][0]
                    converted_refresh_excursions = [
                        converted_excursion
                        for converted_excursion in converted_refresh_excursions
                        if converted_excursion[0] > last_existing_date
                    ]
                combined_excursions = (
                    existing_excursions + converted_refresh_excursions
                )
                applied_percentages = adaptive_tp_sl_applied.get(
                    id(current_adjusted_trade),
                    (0.0, adaptive_tp_sl.min_sl),
                )
                active_stop_loss_percentage = applied_percentages[1]

                replacement_trade = current_adjusted_trade
                for refresh_bar_index, refresh_excursion in enumerate(
                    converted_refresh_excursions,
                    start=1,
                ):
                    bar_date = refresh_excursion[0]
                    bar_low_percentage = refresh_excursion[2]
                    if (
                        refresh_bar_index >= minimum_holding_bars
                        and bar_low_percentage <= -active_stop_loss_percentage
                    ):
                        stop_loss_exit_price = (
                            current_adjusted_trade.entry_price
                            * (1 - active_stop_loss_percentage)
                        )
                        excursions_until_stop = combined_excursions[
                            : len(existing_excursions) + refresh_bar_index
                        ]
                        (
                            favorable_percentage,
                            adverse_percentage,
                            favorable_date,
                            adverse_date,
                        ) = _excursion_extremes(excursions_until_stop)
                        replacement_trade = replace(
                            current_adjusted_trade,
                            exit_date=bar_date,
                            exit_price=stop_loss_exit_price,
                            profit=stop_loss_exit_price
                            - current_adjusted_trade.entry_price,
                            holding_period=len(excursions_until_stop),
                            exit_reason="adaptive_stop_loss",
                            bar_excursions=excursions_until_stop,
                            max_favorable_excursion_pct=favorable_percentage,
                            max_adverse_excursion_pct=adverse_percentage,
                            max_favorable_excursion_date=favorable_date,
                            max_adverse_excursion_date=adverse_date,
                        )
                        break
                else:
                    (
                        favorable_percentage,
                        adverse_percentage,
                        favorable_date,
                        adverse_date,
                    ) = _excursion_extremes(combined_excursions)
                    replacement_trade = replace(
                        current_adjusted_trade,
                        exit_date=_far_future,
                        exit_price=refresh_trade.exit_price,
                        profit=refresh_trade.exit_price
                        - current_adjusted_trade.entry_price,
                        holding_period=len(combined_excursions),
                        exit_reason="end_of_data",
                        bar_excursions=combined_excursions,
                        max_favorable_excursion_pct=favorable_percentage,
                        max_adverse_excursion_pct=adverse_percentage,
                        max_favorable_excursion_date=favorable_date,
                        max_adverse_excursion_date=adverse_date,
                    )
                    adaptive_fallback_exits[open_trade_identifier] = (
                        refresh_trade.exit_date,
                        refresh_trade.exit_price,
                        len(combined_excursions),
                    )

                trades_list = accepted_trades_by_set[label]
                for trade_index, accepted_trade in enumerate(trades_list):
                    if id(accepted_trade) == id(current_adjusted_trade):
                        trades_list[trade_index] = replacement_trade
                        break
                adaptive_trade_map[open_trade_identifier] = replacement_trade
                adaptive_original_trade[id(replacement_trade)] = (
                    current_original_trade
                )
                adaptive_tp_sl_applied[id(replacement_trade)] = (
                    adaptive_tp_sl_applied.pop(
                        id(current_adjusted_trade),
                        applied_percentages,
                    )
                )
                if replacement_trade.exit_date != _far_future:
                    adaptive_close_counter += 1
                    heapq.heappush(
                        adaptive_close_heap,
                        (
                            replacement_trade.exit_date,
                            adaptive_close_counter,
                            label,
                            open_trade_identifier,
                        ),
                    )

        while entry_events or adaptive_close_heap:
            # Determine next event: either from entry_events or close_heap.
            next_entry = entry_events[0] if entry_events else None
            next_close = adaptive_close_heap[0] if adaptive_close_heap else None

            process_close = False
            if next_entry is None and next_close is not None:
                if use_evict_oldest and next_close[0] >= _far_future:
                    # Evict-mode end-of-data: settle remaining open trades
                    # at their original signal exit price/date.
                    while adaptive_close_heap:
                        _, _cnt2, cl, tid = heapq.heappop(adaptive_close_heap)
                        if (cl, tid) not in open_trade_keys:
                            continue
                        adj = adaptive_trade_map.get(tid)
                        if adj is None:
                            continue
                        if adj.exit_date != _far_future:
                            continue
                        orig = adaptive_original_trade.get(id(adj), adj)
                        fallback_exit = adaptive_fallback_exits.get(
                            tid,
                            (orig.exit_date, orig.exit_price, orig.holding_period),
                        )
                        fallback_exit_date = fallback_exit[0]
                        fallback_exit_price = fallback_exit[1]
                        fallback_holding_period = fallback_exit[2]
                        settled = replace(
                            adj,
                            exit_date=fallback_exit_date,
                            exit_price=fallback_exit_price,
                            profit=fallback_exit_price - adj.entry_price,
                            holding_period=fallback_holding_period,
                            exit_reason="end_of_data",
                        )
                        trades_list = accepted_trades_by_set[cl]
                        for ti, t in enumerate(trades_list):
                            if id(t) == id(adj):
                                trades_list[ti] = settled
                                break
                        adaptive_trade_map[tid] = settled
                        adaptive_original_trade[id(settled)] = orig
                        adaptive_tp_sl_applied[id(settled)] = adaptive_tp_sl_applied.pop(
                            id(adj), (0.0, 0.0)
                        )
                    break
                process_close = True
            elif next_close is not None and next_entry is not None:
                # Close events (type=0) before entry events (type=1) on same
                # date, matching the original sort convention.
                if next_close[0] < next_entry[0]:
                    process_close = True
                elif next_close[0] == next_entry[0]:
                    process_close = True  # close before open on same day

            if process_close:
                close_date, _cnt, close_label, orig_trade_id = heapq.heappop(
                    adaptive_close_heap
                )
                # Track same-day releases for lookahead prevention.
                if last_close_date != close_date:
                    same_day_close_count = 0
                    last_close_date = close_date
                close_key = (close_label, orig_trade_id)
                if use_evict_oldest:
                    adjusted_close_trade = adaptive_trade_map.get(orig_trade_id)
                    expected_close_date = adaptive_slot_close_dates.get(
                        orig_trade_id,
                        adjusted_close_trade.exit_date
                        if adjusted_close_trade is not None
                        else close_date,
                    )
                    if (
                        adjusted_close_trade is not None
                        and expected_close_date != close_date
                    ):
                        # A same-symbol refresh or eviction can replace the open
                        # trade after an older close event has already been queued.
                        # Ignore that stale heap event so it cannot close or settle
                        # the refreshed/evicted position.
                        continue
                if close_key in open_trade_keys:
                    open_trade_keys.pop(close_key, None)
                    open_position_counts_by_set[close_label] = max(
                        0, open_position_counts_by_set[close_label] - 1
                    )
                    same_day_close_count += 1
                    if use_evict_oldest:
                        open_trade_entry_dates.pop(orig_trade_id, None)
                    # Decrement same-symbol counter.
                    closed_sym = open_trade_symbols.pop(orig_trade_id, None)
                    if closed_sym and closed_sym in open_symbol_counts:
                        open_symbol_counts[closed_sym] = max(
                            0, open_symbol_counts[closed_sym] - 1
                        )
                    # Update rolling stats using the RAW (original) trade's
                    # result, not the adaptive-adjusted one.  This ensures
                    # the rolling window reflects true signal quality, not
                    # the effect of the adaptive TP/SL itself.
                    adjusted_trade = adaptive_trade_map.get(orig_trade_id)
                    if adjusted_trade is not None:
                        raw_trade = adaptive_original_trade.get(
                            id(adjusted_trade), adjusted_trade
                        )
                        pct = (
                            (raw_trade.exit_price - raw_trade.entry_price)
                            / raw_trade.entry_price
                            if raw_trade.entry_price > 0
                            else 0.0
                        )
                        rolling_update_date = max(
                            close_date,
                            raw_trade.exit_date,
                        )
                        if (
                            adaptive_tp_sl.delayed_rolling_update
                            or rolling_update_date > close_date
                        ):
                            pending_rolling_updates.append(
                                (rolling_update_date, pct)
                            )
                        else:
                            if pct > 0:
                                adaptive_closed_winners.append(pct)
                                if len(adaptive_closed_winners) > adaptive_tp_sl.window:
                                    adaptive_closed_winners.popleft()
                            elif pct < 0:
                                adaptive_closed_losers.append(pct)
                                if len(adaptive_closed_losers) > adaptive_tp_sl.window:
                                    adaptive_closed_losers.popleft()
            else:
                (
                    event_date,
                    event_type,
                    _bucket_priority,
                    _event_priority,
                    _insertion_counter,
                    label,
                    trade,
                ) = entry_events.popleft()
                # Flush pending rolling updates from trades that closed
                # strictly before this entry date (delayed_rolling_update).
                if pending_rolling_updates:
                    remaining: list[tuple[pandas.Timestamp, float]] = []
                    for closed_date, closed_pct in pending_rolling_updates:
                        if closed_date < event_date:
                            if closed_pct > 0:
                                adaptive_closed_winners.append(closed_pct)
                                if len(adaptive_closed_winners) > adaptive_tp_sl.window:
                                    adaptive_closed_winners.popleft()
                            elif closed_pct < 0:
                                adaptive_closed_losers.append(closed_pct)
                                if len(adaptive_closed_losers) > adaptive_tp_sl.window:
                                    adaptive_closed_losers.popleft()
                        else:
                            remaining.append((closed_date, closed_pct))
                    pending_rolling_updates[:] = remaining
                # Reset same-day close counter when we move to a new date.
                if last_close_date is not None and event_date > last_close_date:
                    same_day_close_count = 0
                trade_identifier = id(trade)
                trade_key = (label, trade_identifier)
                if trade_key in accepted_trade_keys:
                    continue
                trade_sym = artifacts_by_set[label].trade_symbol_lookup.get(
                    trade, "",
                )
                # Per-bucket entry filters reading from trade detail
                # (slope_60 / near_delta computed upstream).
                _bucket_def_slope = set_definitions[label]
                _need_detail = (
                    _bucket_def_slope.slope_max is not None
                    or _bucket_def_slope.slope_min is not None
                    or (
                        _bucket_def_slope.free_fall_slope is not None
                        and _bucket_def_slope.free_fall_near_delta is not None
                    )
                )
                if _need_detail:
                    _detail_pair = artifacts_by_set[label].trade_detail_pairs.get(
                        trade
                    )
                    if _detail_pair is not None:
                        _entry_slope = _detail_pair[0].slope_60
                        _entry_near_delta = _detail_pair[0].near_delta
                        # Independent slope bounds (unconditional).
                        if _entry_slope is not None:
                            if (
                                _bucket_def_slope.slope_max is not None
                                and _entry_slope > _bucket_def_slope.slope_max
                            ):
                                continue
                            if (
                                _bucket_def_slope.slope_min is not None
                                and _entry_slope < _bucket_def_slope.slope_min
                            ):
                                continue
                        # Compound free-fall filter (AND): both must trigger.
                        if (
                            _bucket_def_slope.free_fall_slope is not None
                            and _bucket_def_slope.free_fall_near_delta is not None
                            and _entry_slope is not None
                            and _entry_near_delta is not None
                            and _entry_slope < _bucket_def_slope.free_fall_slope
                            and _entry_near_delta
                            < _bucket_def_slope.free_fall_near_delta
                        ):
                            continue
                if use_evict_oldest and trade_sym:
                    refreshed_trade_identifier: int | None = None
                    for open_identifier, open_symbol in open_trade_symbols.items():
                        if open_symbol == trade_sym:
                            refreshed_trade_identifier = open_identifier
                            break
                    if refreshed_trade_identifier is not None:
                        # Same-symbol signal refreshes the operating clock but
                        # does not close/re-open the real position.  Accounting
                        # entry date, entry price, and holding-period basis stay
                        # on the original trade.
                        _refresh_open_trade_from_same_symbol_signal(
                            open_trade_identifier=refreshed_trade_identifier,
                            refresh_trade=trade,
                        )
                        open_trade_entry_dates[refreshed_trade_identifier] = event_date
                        accepted_trade_keys.add(trade_key)
                        continue
                # Slot check: add back same-day closes to prevent lookahead.
                # Entries cannot use slots freed by closes on the same date
                # because you don't know at entry time whether TP/SL will
                # trigger later that day.
                current_open_total = len(open_trade_keys) + same_day_close_count
                if current_open_total >= maximum_position_count:
                    if not use_evict_oldest:
                        continue
                    # Evict oldest open position past min_hold.
                    evict_candidate: tuple[pandas.Timestamp, int, str] | None = None
                    for (ek_label, ek_tid), ek_lbl in open_trade_keys.items():
                        e_date = open_trade_entry_dates.get(ek_tid)
                        if e_date is None:
                            continue
                        # Count business days held.
                        held_days = len(pandas.bdate_range(e_date, event_date)) - 1
                        if held_days < minimum_holding_bars:
                            continue
                        if evict_candidate is None or e_date < evict_candidate[0]:
                            evict_candidate = (e_date, ek_tid, ek_label)
                    if evict_candidate is None:
                        continue  # all open positions within min_hold
                    # Perform eviction.
                    evict_entry_date, evict_tid, evict_label = evict_candidate
                    evict_key = (evict_label, evict_tid)
                    evicted_adjusted = adaptive_trade_map.get(evict_tid)
                    if evicted_adjusted is None:
                        continue
                    evicted_original = adaptive_original_trade.get(
                        id(evicted_adjusted), evicted_adjusted
                    )
                    # Find eviction day open price from bar_excursions.
                    evict_exit_price = evicted_adjusted.entry_price  # fallback
                    evict_holding = 0
                    if evicted_original.bar_excursions:
                        for bi, exc in enumerate(evicted_original.bar_excursions):
                            exc_date = exc[0]
                            exc_open_pct = exc[3] if len(exc) == 4 else 0.0
                            if exc_date >= event_date:
                                evict_exit_price = evicted_adjusted.entry_price * (1 + exc_open_pct)
                                evict_holding = bi + 1
                                break
                        else:
                            # event_date beyond last bar — use last bar close
                            evict_exit_price = evicted_original.exit_price
                            evict_holding = len(evicted_original.bar_excursions)
                    evict_profit = evict_exit_price - evicted_adjusted.entry_price
                    evicted_new = replace(
                        evicted_adjusted,
                        exit_date=event_date,
                        exit_price=evict_exit_price,
                        profit=evict_profit,
                        holding_period=evict_holding,
                        exit_reason="evicted",
                        bar_excursions=(
                            evicted_original.bar_excursions[:evict_holding]
                            if evicted_original.bar_excursions else None
                        ),
                    )
                    # Replace in accepted trades list.
                    trades_list = accepted_trades_by_set[evict_label]
                    for ti, t in enumerate(trades_list):
                        if id(t) == id(evicted_adjusted):
                            trades_list[ti] = evicted_new
                            break
                    adaptive_trade_map[evict_tid] = evicted_new
                    adaptive_original_trade[id(evicted_new)] = evicted_original
                    adaptive_tp_sl_applied[id(evicted_new)] = adaptive_tp_sl_applied.pop(
                        id(evicted_adjusted), (0.0, 0.0)
                    )
                    # Clean up open tracking.
                    open_trade_keys.pop(evict_key, None)
                    open_position_counts_by_set[evict_label] = max(
                        0, open_position_counts_by_set[evict_label] - 1
                    )
                    open_trade_entry_dates.pop(evict_tid, None)
                    evict_sym = open_trade_symbols.pop(evict_tid, None)
                    if evict_sym and evict_sym in open_symbol_counts:
                        open_symbol_counts[evict_sym] = max(
                            0, open_symbol_counts[evict_sym] - 1
                        )
                    # Update rolling stats with evicted trade's result.
                    evict_pct = (
                        evict_profit / evicted_adjusted.entry_price
                        if evicted_adjusted.entry_price > 0 else 0.0
                    )
                    if adaptive_tp_sl.delayed_rolling_update:
                        pending_rolling_updates.append((event_date, evict_pct))
                    else:
                        if evict_pct > 0:
                            adaptive_closed_winners.append(evict_pct)
                            if len(adaptive_closed_winners) > adaptive_tp_sl.window:
                                adaptive_closed_winners.popleft()
                        elif evict_pct < 0:
                            adaptive_closed_losers.append(evict_pct)
                            if len(adaptive_closed_losers) > adaptive_tp_sl.window:
                                adaptive_closed_losers.popleft()
                    # Eviction is deliberate (not subject to lookahead like
                    # TP/SL closes), so do NOT increment same_day_close_count.
                    current_open_total = len(open_trade_keys) + same_day_close_count
                if (
                    not multi_bucket_mode
                    and label.upper() == "B"
                    and current_open_total >= position_limits_by_set[label]
                ):
                    continue
                if multi_bucket_mode and set_definitions[label].fill_remaining:
                    if current_open_total >= position_limits_by_set[label]:
                        continue
                elif open_position_counts_by_set[label] >= position_limits_by_set[label]:
                    continue

                # Check max_same_symbol limit.
                if max_same_symbol < 999:
                    if open_symbol_counts.get(trade_sym, 0) >= max_same_symbol:
                        continue

                # Resolve per-bucket overrides up front. min_sl is needed
                # before rolling computation since it is the SL floor; the
                # rest are applied after rolling stats are computed.
                bucket_def = set_definitions[label]
                effective_min_sl = (
                    bucket_def.min_sl
                    if bucket_def.min_sl is not None
                    else adaptive_tp_sl.min_sl
                )
                effective_fixed_sl = (
                    bucket_def.fixed_sl
                    if bucket_def.fixed_sl is not None
                    else adaptive_tp_sl.fixed_sl
                )
                effective_fixed_tp = (
                    bucket_def.fixed_tp
                    if bucket_def.fixed_tp is not None
                    else adaptive_tp_sl.fixed_tp
                )
                effective_tp_regime_adjust = (
                    bucket_def.tp_regime_adjust
                    if bucket_def.tp_regime_adjust is not None
                    else adaptive_tp_sl.tp_regime_adjust
                )

                # Compute adaptive TP/SL from rolling stats.
                tp_pct = adaptive_tp_sl.min_tp
                sl_pct = effective_min_sl
                rolling_mp = 0.0

                _tp_pct_late: float | None = None

                if (
                    len(adaptive_closed_winners) + len(adaptive_closed_losers)
                    >= adaptive_tp_sl.min_samples
                ):
                    profits = list(adaptive_closed_winners)
                    if len(profits) >= 3:
                        mean_profit_percentage = sum(profits) / len(profits)
                        rolling_mp = mean_profit_percentage
                        if len(profits) >= 2:
                            profit_standard_deviation = stdev(profits)
                        else:
                            profit_standard_deviation = 0.0
                        tp_pct = max(
                            adaptive_tp_sl.min_tp,
                            mean_profit_percentage
                            + adaptive_tp_sl.sigma_multiplier
                            * profit_standard_deviation,
                        )

                    losses = [
                        abs(loss_percentage)
                        for loss_percentage in adaptive_closed_losers
                    ]
                    if len(losses) >= 3:
                        sl_pct = max(
                            effective_min_sl,
                            median(losses),
                        )

                # Apply fixed_sl as ceiling (cap): SL never exceeds this.
                if effective_fixed_sl is not None:
                    sl_pct = min(sl_pct, effective_fixed_sl)

                # Regime-adaptive TP: multiply TP target by capped tp/sl ratio.
                # Skipped when fixed_tp is in effect (fixed value below wins).
                if (
                    effective_tp_regime_adjust
                    and sl_pct > 0
                    and effective_fixed_tp is None
                ):
                    raw_ratio = tp_pct / sl_pct
                    capped_ratio = max(
                        adaptive_tp_sl.tp_regime_ratio_min,
                        min(adaptive_tp_sl.tp_regime_ratio_max, raw_ratio),
                    )
                    tp_pct = max(
                        adaptive_tp_sl.min_tp,
                        tp_pct * capped_ratio,
                    )

                # Apply fixed_tp as override: forces TP to this exact value,
                # bypassing rolling stats and regime adjustment. Last word.
                if effective_fixed_tp is not None:
                    tp_pct = effective_fixed_tp

                # Break-even trigger: rolling MP (before sigma adjustment).
                be_trigger = rolling_mp if adaptive_tp_sl.breakeven_at_mp else 0.0

                # Replay trade with adaptive levels.
                effective_min_hold = (
                    0 if adaptive_tp_sl.override_min_hold
                    else minimum_holding_bars
                )
                # Per-bucket override for min_hold gates (None = inherit top-level).
                effective_override_tp = (
                    bucket_def.override_min_hold_tp_only
                    if bucket_def.override_min_hold_tp_only is not None
                    else adaptive_tp_sl.override_min_hold_tp_only
                )
                effective_min_hold_tp_value = (
                    bucket_def.min_hold_tp
                    if bucket_def.min_hold_tp is not None
                    else adaptive_tp_sl.min_hold_tp
                )
                effective_override_sl = (
                    bucket_def.override_min_hold_sl_only
                    if bucket_def.override_min_hold_sl_only is not None
                    else adaptive_tp_sl.override_min_hold_sl_only
                )
                effective_min_hold_sl_value = (
                    bucket_def.min_hold_sl
                    if bucket_def.min_hold_sl is not None
                    else adaptive_tp_sl.min_hold_sl
                )
                effective_min_hold_tp: int | None = None
                if effective_override_tp:
                    effective_min_hold_tp = effective_min_hold_tp_value
                effective_min_hold_sl: int | None = None
                if effective_override_sl:
                    effective_min_hold_sl = effective_min_hold_sl_value
                adjusted = _replay_trade_with_adaptive_tp_sl(
                    trade, tp_pct, sl_pct, effective_min_hold,
                    minimum_holding_bars_tp=effective_min_hold_tp,
                    minimum_holding_bars_sl=effective_min_hold_sl,
                    breakeven_trigger_pct=be_trigger,
                    tp_pct_late=_tp_pct_late,
                    sl_only=use_evict_oldest,
                    disable_sl_trigger=adaptive_tp_sl.disable_sl_trigger,
                )
                # Dynamic min_hold throttle based on R-multiple (rolling SL/TP).
                # TP exits release immediately.  Non-TP exits keep the slot
                # locked until max(min_hold, round(min_hold * SL / TP)).
                if tp_pct > 0 and effective_min_hold > 0:
                    dynamic_min_hold = max(
                        effective_min_hold,
                        round(effective_min_hold * sl_pct / tp_pct),
                    )
                else:
                    dynamic_min_hold = effective_min_hold

                slot_release_date = adjusted.exit_date
                slot_release_holding = adjusted.holding_period
                slot_release_bar_excursions = adjusted.bar_excursions
                if (
                    adjusted.exit_reason != "adaptive_take_profit"
                    and dynamic_min_hold > adjusted.holding_period
                ):
                    trade_symbol = artifacts_by_set[label].trade_symbol_lookup.get(
                        trade,
                        "",
                    )
                    (
                        slot_release_date,
                        slot_release_holding,
                        slot_release_bar_excursions,
                    ) = _resolve_slot_release_date(
                        trade,
                        dynamic_min_hold,
                        trade_symbol,
                        artifacts_by_set[label].closing_price_series_by_symbol,
                    )

                slot_trade = replace(
                    adjusted,
                    exit_date=slot_release_date,
                    holding_period=slot_release_holding,
                    bar_excursions=slot_release_bar_excursions,
                )
                adaptive_slot_close_dates[trade_identifier] = slot_trade.exit_date
                adaptive_slot_trade_map[id(adjusted)] = slot_trade
                adaptive_trade_map[trade_identifier] = adjusted
                adaptive_original_trade[id(adjusted)] = trade
                adaptive_tp_sl_applied[id(adjusted)] = (tp_pct, sl_pct)
                if use_evict_oldest and adjusted.exit_date == _far_future:
                    adaptive_fallback_exits[trade_identifier] = (
                        trade.exit_date,
                        trade.exit_price,
                        trade.holding_period,
                    )

                accepted_trade_keys.add(trade_key)
                open_trade_keys[trade_key] = label
                open_trade_entry_dates[trade_identifier] = event_date
                open_position_counts_by_set[label] += 1
                # Track same-symbol count.
                if trade_sym:
                    open_trade_symbols[trade_identifier] = trade_sym
                if max_same_symbol < 999:
                    open_symbol_counts[trade_sym] = open_symbol_counts.get(trade_sym, 0) + 1
                accepted_trades_by_set[label].append(adjusted)

                # Schedule close event.
                adaptive_close_counter += 1
                heapq.heappush(
                    adaptive_close_heap,
                    (
                        adaptive_slot_close_dates[trade_identifier],
                        adaptive_close_counter,
                        label,
                        trade_identifier,
                    ),
                )
    else:
        # Original non-adaptive event processing.
        for (
            event_date,
            event_type,
            _bucket_priority,
            _event_priority,
            _insertion_counter,
            label,
            trade,
        ) in events:
            trade_identifier = id(trade)
            trade_key = (label, trade_identifier)
            normalized_label = label.upper()
            if event_type == 0:
                if trade_key in open_trade_keys:
                    open_trade_keys.pop(trade_key, None)
                    open_position_counts_by_set[label] = max(
                        0, open_position_counts_by_set[label] - 1
                    )
            else:
                if trade_key in accepted_trade_keys:
                    continue
                current_open_total = len(open_trade_keys)
                if current_open_total >= maximum_position_count:
                    continue
                if (
                    not multi_bucket_mode
                    and normalized_label == "B"
                    and current_open_total >= position_limits_by_set[label]
                ):
                    continue
                if multi_bucket_mode and set_definitions[label].fill_remaining:
                    # fill_remaining bucket: can only open when total open
                    # positions (across all buckets) is below this bucket's max.
                    if current_open_total >= position_limits_by_set[label]:
                        continue
                elif open_position_counts_by_set[label] >= position_limits_by_set[label]:
                    continue
                accepted_trade_keys.add(trade_key)
                open_trade_keys[trade_key] = label
                open_position_counts_by_set[label] += 1
                accepted_trades_by_set[label].append(trade)

    metrics_by_set: Dict[str, StrategyMetrics] = {}
    aggregated_trades: List[Trade] = []
    aggregated_slot_trades: List[Trade] = []
    aggregated_trade_profit_list: List[float] = []
    aggregated_profit_percentage_list: List[float] = []
    aggregated_loss_percentage_list: List[float] = []
    aggregated_holding_period_list: List[int] = []
    aggregated_detail_pairs_with_label: List[
        Tuple[TradeDetail, TradeDetail]
    ] = []
    aggregated_trade_symbol_lookup: Dict[Trade, str] = {}
    aggregated_slot_trade_symbol_lookup: Dict[Trade, str] = {}
    aggregated_simulation_results: List[SimulationResult] = []
    aggregated_closing_price_series_by_symbol: Dict[str, pandas.Series] = {}

    for label, artifacts in artifacts_by_set.items():
        trades_for_set = accepted_trades_by_set[label]
        trade_profit_list: List[float] = []
        profit_percentage_list: List[float] = []
        loss_percentage_list: List[float] = []
        holding_period_list: List[int] = []
        detail_pairs_with_label: List[Tuple[TradeDetail, TradeDetail]] = []
        filtered_trade_symbol_lookup: Dict[Trade, str] = {}

        for trade in trades_for_set:
            trade_profit_list.append(trade.profit)
            holding_period_list.append(trade.holding_period)
            percentage_change = trade.profit / trade.entry_price
            if percentage_change > 0:
                profit_percentage_list.append(percentage_change)
            elif percentage_change < 0:
                loss_percentage_list.append(abs(percentage_change))
            # For adaptive TP/SL, the accepted trade may be a replayed copy.
            # Look up the original trade for symbol and detail pair mapping.
            original_trade = adaptive_original_trade.get(id(trade), trade)
            filtered_trade_symbol_lookup[trade] = artifacts.trade_symbol_lookup.get(
                original_trade, ""
            )
            entry_detail, exit_detail = artifacts.trade_detail_pairs[original_trade]
            # When adaptive TP/SL modified the trade, update the exit detail
            # to reflect the adjusted exit date, price, reason, and holding.
            _applied = adaptive_tp_sl_applied.get(id(trade))
            if trade is not original_trade:
                pct_change = (
                    (trade.exit_price - trade.entry_price) / trade.entry_price
                    if trade.entry_price > 0
                    else 0.0
                )
                exit_detail = replace(
                    exit_detail,
                    date=trade.exit_date,
                    price=trade.exit_price,
                    exit_reason=trade.exit_reason,
                    percentage_change=pct_change,
                    result="win" if pct_change > 0 else "lose",
                    max_favorable_excursion_pct=trade.max_favorable_excursion_pct,
                    max_adverse_excursion_pct=trade.max_adverse_excursion_pct,
                    max_favorable_excursion_date=trade.max_favorable_excursion_date,
                    max_adverse_excursion_date=trade.max_adverse_excursion_date,
                    adaptive_tp_pct=_applied[0] if _applied else None,
                    adaptive_sl_pct=_applied[1] if _applied else None,
                )
            if _applied and exit_detail.adaptive_tp_pct is None:
                exit_detail = replace(
                    exit_detail,
                    adaptive_tp_pct=_applied[0],
                    adaptive_sl_pct=_applied[1],
                )
            detail_pairs_with_label.append(
                (
                    replace(entry_detail, strategy_set_label=label),
                    replace(exit_detail, strategy_set_label=label),
                )
            )

        slot_trades_for_set = [
            adaptive_slot_trade_map.get(id(trade), trade)
            for trade in trades_for_set
        ]
        filtered_slot_trade_symbol_lookup = {
            adaptive_slot_trade_map.get(id(trade), trade): filtered_trade_symbol_lookup[
                trade
            ]
            for trade in trades_for_set
            if trade in filtered_trade_symbol_lookup
        }

        trade_details_by_year = _organize_trade_details_by_year(detail_pairs_with_label)
        filtered_simulation_results: List[SimulationResult] = []
        if slot_trades_for_set:
            filtered_simulation_results.append(
                SimulationResult(
                    trades=slot_trades_for_set,
                    total_profit=sum(trade.profit for trade in trades_for_set),
                )
            )
        maximum_concurrent_positions = calculate_maximum_concurrent_positions(
            filtered_simulation_results
        )

        if not trades_for_set:
            metrics_by_set[label] = calculate_metrics(
                [],
                [],
                [],
                [],
                maximum_concurrent_positions,
                0.0,
                0.0,
                0.0,
                {},
                {},
                trade_details_by_year,
            )
            continue

        simulation_start_date = artifacts.simulation_start_date
        if simulation_start_date is None:
            simulation_start_date = pandas.Timestamp.now()
        annual_returns = calculate_annual_returns(
            slot_trades_for_set,
            starting_cash,
            position_limits_by_set[label],
            simulation_start_date,
            withdraw_amount,
            margin_multiplier=margin_multiplier,
            margin_interest_annual_rate=effective_interest_rate,
            trade_symbol_lookup=filtered_slot_trade_symbol_lookup,
            closing_price_series_by_symbol=artifacts.closing_price_series_by_symbol,
            settlement_lag_days=1,
        )
        annual_trade_counts = calculate_annual_trade_counts(trades_for_set)
        final_balance = simulate_portfolio_balance(
            slot_trades_for_set,
            starting_cash,
            position_limits_by_set[label],
            withdraw_amount,
            margin_multiplier=margin_multiplier,
            margin_interest_annual_rate=effective_interest_rate,
        )
        # Propagate commission data from Trade objects (set by
        # simulate_portfolio_balance) back to the corresponding TradeDetails.
        for trade in trades_for_set:
            slot_trade = adaptive_slot_trade_map.get(id(trade), trade)
            trade.total_commission = slot_trade.total_commission
            trade.share_count = slot_trade.share_count
            orig = adaptive_original_trade.get(id(trade), trade)
            if orig in artifacts.trade_detail_pairs:
                _, exit_detail = artifacts.trade_detail_pairs[orig]
                exit_detail.total_commission = trade.total_commission
                exit_detail.share_count = trade.share_count

        maximum_drawdown = calculate_max_drawdown(
            slot_trades_for_set,
            starting_cash,
            position_limits_by_set[label],
            filtered_slot_trade_symbol_lookup,
            artifacts.closing_price_series_by_symbol,
            withdraw_amount,
            margin_multiplier=margin_multiplier,
            margin_interest_annual_rate=effective_interest_rate,
        )
        if slot_trades_for_set:
            last_trade_exit_date = max(
                trade.exit_date for trade in slot_trades_for_set
            )
        else:
            last_trade_exit_date = simulation_start_date
        compound_annual_growth_rate_value = 0.0
        if (
            simulation_start_date is not None
            and last_trade_exit_date is not None
            and starting_cash > 0
        ):
            duration_days = (last_trade_exit_date - simulation_start_date).days
            if duration_days > 0:
                duration_years = duration_days / 365.25
                compound_annual_growth_rate_value = (
                    final_balance / starting_cash
                ) ** (1 / duration_years) - 1

        metrics_by_set[label] = calculate_metrics(
            trade_profit_list,
            profit_percentage_list,
            loss_percentage_list,
            holding_period_list,
            maximum_concurrent_positions,
            maximum_drawdown,
            final_balance,
            compound_annual_growth_rate_value,
            annual_returns,
            annual_trade_counts,
            trade_details_by_year,
        )

        aggregated_trades.extend(trades_for_set)
        aggregated_slot_trades.extend(slot_trades_for_set)
        aggregated_trade_profit_list.extend(trade_profit_list)
        aggregated_profit_percentage_list.extend(profit_percentage_list)
        aggregated_loss_percentage_list.extend(loss_percentage_list)
        aggregated_holding_period_list.extend(holding_period_list)
        aggregated_detail_pairs_with_label.extend(detail_pairs_with_label)
        aggregated_trade_symbol_lookup.update(filtered_trade_symbol_lookup)
        aggregated_slot_trade_symbol_lookup.update(filtered_slot_trade_symbol_lookup)
        aggregated_simulation_results.extend(filtered_simulation_results)
        for symbol_name, closing_series in (
            artifacts.closing_price_series_by_symbol.items()
        ):
            if symbol_name not in aggregated_closing_price_series_by_symbol:
                aggregated_closing_price_series_by_symbol[
                    symbol_name
                ] = closing_series

    aggregated_trade_details_by_year = _organize_trade_details_by_year(
        aggregated_detail_pairs_with_label
    )
    _assign_global_concurrent_position_counts(aggregated_detail_pairs_with_label)
    aggregated_maximum_concurrent_positions = calculate_maximum_concurrent_positions(
        aggregated_simulation_results
    )

    if aggregated_trades:
        start_dates = [
            artifacts.simulation_start_date
            for artifacts in artifacts_by_set.values()
            if artifacts.simulation_start_date is not None
        ]
        if start_dates:
            aggregated_simulation_start_date = min(start_dates)
        else:
            aggregated_simulation_start_date = pandas.Timestamp.now()
        aggregated_annual_returns = calculate_annual_returns(
            aggregated_slot_trades,
            starting_cash,
            maximum_position_count,
            aggregated_simulation_start_date,
            withdraw_amount,
            margin_multiplier=margin_multiplier,
            margin_interest_annual_rate=effective_interest_rate,
            trade_symbol_lookup=aggregated_slot_trade_symbol_lookup,
            closing_price_series_by_symbol=(
                aggregated_closing_price_series_by_symbol
            ),
            settlement_lag_days=1,
        )
        aggregated_annual_trade_counts = calculate_annual_trade_counts(
            aggregated_trades
        )
        aggregated_final_balance = simulate_portfolio_balance(
            aggregated_slot_trades,
            starting_cash,
            maximum_position_count,
            withdraw_amount,
            margin_multiplier=margin_multiplier,
            margin_interest_annual_rate=effective_interest_rate,
        )
        aggregated_maximum_drawdown = calculate_max_drawdown(
            aggregated_slot_trades,
            starting_cash,
            maximum_position_count,
            aggregated_slot_trade_symbol_lookup,
            aggregated_closing_price_series_by_symbol,
            withdraw_amount,
            margin_multiplier=margin_multiplier,
            margin_interest_annual_rate=effective_interest_rate,
        )
        last_exit_date = max(trade.exit_date for trade in aggregated_slot_trades)
        aggregated_compound_annual_growth_rate = 0.0
        if starting_cash > 0 and aggregated_simulation_start_date is not None:
            duration_days = (last_exit_date - aggregated_simulation_start_date).days
            if duration_days > 0:
                duration_years = duration_days / 365.25
                aggregated_compound_annual_growth_rate = (
                    aggregated_final_balance / starting_cash
                ) ** (1 / duration_years) - 1
    else:
        aggregated_annual_returns = {}
        aggregated_annual_trade_counts = {}
        aggregated_final_balance = 0.0
        aggregated_maximum_drawdown = 0.0
        aggregated_compound_annual_growth_rate = 0.0

    overall_metrics = calculate_metrics(
        aggregated_trade_profit_list,
        aggregated_profit_percentage_list,
        aggregated_loss_percentage_list,
        aggregated_holding_period_list,
        aggregated_maximum_concurrent_positions,
        aggregated_maximum_drawdown,
        aggregated_final_balance,
        aggregated_compound_annual_growth_rate,
        aggregated_annual_returns,
        aggregated_annual_trade_counts,
        aggregated_trade_details_by_year,
    )

    return ComplexSimulationMetrics(
        overall_metrics=overall_metrics, metrics_by_set=metrics_by_set
    )


def compute_signals_for_date(
    data_directory: Path,
    evaluation_date: pandas.Timestamp,
    buy_strategy_name: str,
    sell_strategy_name: str,
    *,
    minimum_average_dollar_volume: float | None = None,
    top_dollar_volume_rank: int | None = None,
    minimum_average_dollar_volume_ratio: float | None = None,
    allowed_fama_french_groups: set[int] | None = None,
    allowed_symbols: set[str] | None = None,
    exclude_other_ff12: bool = True,
    maximum_symbols_per_group: int = 1,
    use_unshifted_signals: bool = False,
    near_delta_range: tuple[float, float] | None = None,
    price_tightness_range: tuple[float, float] | None = None,
) -> Dict[str, List[str]]:
    """Compute entry/exit signals on ``evaluation_date`` using simulation filters.

    This helper reproduces the symbol-universe preparation and selection logic
    used by :func:`evaluate_combined_strategy` (group-aware ratio thresholds,
    Top-N with one-per-group cap, entry gated by eligibility) but returns only
    the symbols that have buy or sell signals on the last available bar at or
    before ``evaluation_date``. The result does not depend on position sizing or
    portfolio capacity and does not require running the trade simulator.

    Parameters
    ----------
    data_directory:
        Directory containing price CSV files.
    evaluation_date:
        Date at which signals are sampled. If a symbol has no row exactly on
        this day, the most recent row before this day is used.
    buy_strategy_name:
        Strategy identifier for entry signals (may include parameters or
        composite expressions like "A or B").
    sell_strategy_name:
        Strategy identifier for exit signals (same conventions as
        ``buy_strategy_name``).
    minimum_average_dollar_volume:
        Absolute 50-day average dollar volume threshold in millions.
    top_dollar_volume_rank:
        Global Top-N ranking. When sector data is available, a per-group cap
        of ``maximum_symbols_per_group`` symbols is enforced.
    minimum_average_dollar_volume_ratio:
        Minimum ratio of total market 50-day average dollar volume. When
        sector data is available, a dynamic per-group threshold is applied to
        avoid bias toward larger groups.
    allowed_fama_french_groups:
        Restrict the tradable universe to the specified FF12 group identifiers
        (1–11). Group 12 ("Other") is not used for group-aware sector
        selection.
    allowed_symbols:
        Optional whitelist of symbols (CSV stems) to consider.
    exclude_other_ff12:
        When True, known non-stock instruments are skipped using SIC/override
        data. FF12 group 12 common stocks are not excluded by this flag.
    maximum_symbols_per_group:
        Maximum number of symbols to select per group when
        ``top_dollar_volume_rank`` is provided.
    use_unshifted_signals:
        When ``True``, strategy helpers may emit ``*_raw_entry_signal`` and
        ``*_raw_exit_signal`` columns. Those unshifted columns are evaluated on
        the same day they are generated.

    Returns
    -------
    Dict[str, List[str] | List[tuple[str, int | None]]]
        Mapping with keys ``"filtered_symbols"``, ``"entry_signals"`` and
        ``"exit_signals"``. ``filtered_symbols`` contains pairs of symbol and
        Fama–French group identifier for symbols that passed the dollar-volume
        filter on the evaluation day. The other lists contain symbols that
        triggered the respective signals on the sampled row.
    """
    # TODO: review

    # Validate strategies first (supports composite expressions)
    buy_choice_names = _split_strategy_choices(buy_strategy_name)
    sell_choice_names = _split_strategy_choices(sell_strategy_name)

    def _has_supported(tokens: list[str], table: dict) -> bool:
        for token in tokens:
            try:
                base_name, _, _, _, _ = parse_strategy_name(token)
            except Exception:  # noqa: BLE001
                continue
            if base_name in table:
                return True
        return False

    if not _has_supported(buy_choice_names, BUY_STRATEGIES):
        raise ValueError(f"Unsupported strategy: {buy_strategy_name}")
    if not _has_supported(sell_choice_names, SELL_STRATEGIES):
        raise ValueError(f"Unsupported strategy: {sell_strategy_name}")

    # Load and normalize per-symbol frames, compute 50-day dollar-volume SMA
    symbol_frames: List[tuple[Path, pandas.DataFrame]] = []
    symbols_excluded_by_industry = (
        load_symbols_excluded_by_industry() if exclude_other_ff12 else set()
    )
    symbol_to_group_map = load_ff12_groups_by_symbol()
    symbol_to_group_map_for_filtering: dict[str, int] | None = None
    if allowed_fama_french_groups is not None:
        symbol_to_group_map_for_filtering = symbol_to_group_map
    for csv_file_path in data_directory.glob("*.csv"):
        if csv_file_path.stem == SP500_SYMBOL:
            continue
        if allowed_symbols is not None and csv_file_path.stem not in allowed_symbols:
            continue
        if csv_file_path.stem.upper() in symbols_excluded_by_industry:
            continue
        if symbol_to_group_map_for_filtering is not None:
            group_identifier = symbol_to_group_map_for_filtering.get(
                csv_file_path.stem.upper()
            )
            if (
                group_identifier is None
                or group_identifier not in allowed_fama_french_groups
            ):
                continue
        price_data_frame = load_price_data(csv_file_path)
        if price_data_frame.empty:
            continue
        if "volume" in price_data_frame.columns:
            dollar_volume_series_full = (
                price_data_frame["close"] * price_data_frame["volume"]
            )
            price_data_frame["simple_moving_average_dollar_volume"] = sma(
                dollar_volume_series_full, DOLLAR_VOLUME_SMA_WINDOW
            )
        else:
            if (
                minimum_average_dollar_volume is not None
                or top_dollar_volume_rank is not None
                or minimum_average_dollar_volume_ratio is not None
            ):
                # Selection requires dollar-volume metrics
                continue
            price_data_frame["simple_moving_average_dollar_volume"] = float("nan")
        symbol_frames.append((csv_file_path, price_data_frame))

    if not symbol_frames:
        return {"entry_signals": [], "exit_signals": []}

    merged_volume_frame = pandas.concat(
        {
            csv_path.stem: frame["simple_moving_average_dollar_volume"]
            for csv_path, frame in symbol_frames
        },
        axis=1,
    )

    # Build eligibility mask (group-aware when sector data is available)
    eligibility_mask = _build_eligibility_mask(
        merged_volume_frame,
        minimum_average_dollar_volume=minimum_average_dollar_volume,
        top_dollar_volume_rank=top_dollar_volume_rank,
        minimum_average_dollar_volume_ratio=minimum_average_dollar_volume_ratio,
        maximum_symbols_per_group=maximum_symbols_per_group,
    )

    filtered_symbols_with_groups: List[tuple[str, int | None]] = []
    last_eligible_date: pandas.Timestamp | None = None
    try:
        eligible_dates = eligibility_mask.index[eligibility_mask.index <= evaluation_date]
        if len(eligible_dates) > 0:
            last_eligible_date = eligible_dates[-1]
            eligibility_row = eligibility_mask.loc[last_eligible_date]
            for symbol_name, is_eligible in eligibility_row.items():
                if bool(is_eligible):
                    group_identifier = symbol_to_group_map.get(symbol_name.upper())
                    filtered_symbols_with_groups.append((symbol_name, group_identifier))
    except Exception:  # noqa: BLE001
        filtered_symbols_with_groups = []
        last_eligible_date = None

    if filtered_symbols_with_groups and last_eligible_date is not None:
        try:
            latest_average_dollar_volume_series = merged_volume_frame.loc[
                last_eligible_date
            ]
        except KeyError:
            latest_average_dollar_volume_series = None
        if isinstance(latest_average_dollar_volume_series, pandas.Series):
            # TODO: review
            symbol_to_average_dollar_volume: dict[str, float] = {}
            for symbol_name, _ in filtered_symbols_with_groups:
                average_dollar_volume_value = latest_average_dollar_volume_series.get(
                    symbol_name,
                    float("nan"),
                )
                if pandas.isna(average_dollar_volume_value):
                    symbol_to_average_dollar_volume[symbol_name] = float("-inf")
                else:
                    symbol_to_average_dollar_volume[symbol_name] = float(
                        average_dollar_volume_value
                    )
            filtered_symbols_with_groups.sort(
                key=lambda symbol_with_group: symbol_to_average_dollar_volume.get(
                    symbol_with_group[0],
                    float("-inf"),
                ),
                reverse=True,
            )

    # Prepare per-symbol masks aligned to each frame
    selected_symbol_data: List[tuple[Path, pandas.DataFrame, pandas.Series]] = []
    for csv_file_path, price_data_frame in symbol_frames:
        symbol_name = csv_file_path.stem
        if symbol_name not in eligibility_mask.columns:
            continue
        symbol_mask = eligibility_mask[symbol_name]
        symbol_mask = symbol_mask.reindex(price_data_frame.index, fill_value=False)
        if not symbol_mask.any():
            continue
        selected_symbol_data.append((csv_file_path, price_data_frame, symbol_mask))

    entry_signal_symbols: List[str] = []
    exit_signal_symbols: List[str] = []

    def _apply_parsed_strategy(
        full_name: str,
        base_name: str,
        window_size: int | None,
        angle_range: tuple[float, float] | None,
        near_range: tuple[float, float] | None,
        above_range: tuple[float, float] | None,
        table: Dict[str, Callable[..., None]],
        frame: pandas.DataFrame,
        include_raw_signals: bool,
    ) -> None:
        """Apply a named strategy function to ``frame`` with parsed parameters."""

        kwargs: dict = {}
        if base_name == "20_50_sma_cross":
            maybe_windows = _extract_short_long_windows_for_20_50(full_name)
            if maybe_windows is not None:
                kwargs["short_window_size"], kwargs["long_window_size"] = maybe_windows
        else:
            if window_size is not None:
                kwargs["window_size"] = window_size
            if angle_range is not None:
                kwargs["angle_range"] = angle_range
            sma_factor_value = _extract_sma_factor(full_name)
            if (
                sma_factor_value is not None
                and base_name in {"ema_sma_cross", "ema_sma_cross_with_slope"}
            ):
                kwargs["sma_window_factor"] = sma_factor_value
            if (
                base_name == "ema_sma_cross_testing"
                and near_range is not None
                and above_range is not None
            ):
                kwargs["near_range"] = near_range
                kwargs["above_range"] = above_range
            if base_name == "ema_sma_cross_testing":
                if near_delta_range is not None:
                    kwargs["near_delta_range"] = near_delta_range
                if price_tightness_range is not None:
                    kwargs["price_tightness_range"] = price_tightness_range
        table[base_name](frame, include_raw_signals=include_raw_signals, **kwargs)
        if base_name != full_name:
            rename_mapping = {
                f"{base_name}_entry_signal": f"{full_name}_entry_signal",
                f"{base_name}_exit_signal": f"{full_name}_exit_signal",
            }
            if include_raw_signals:
                rename_mapping.update(
                    {
                        f"{base_name}_raw_entry_signal": f"{full_name}_raw_entry_signal",
                        f"{base_name}_raw_exit_signal": f"{full_name}_raw_exit_signal",
                    }
                )
            frame.rename(columns=rename_mapping, inplace=True)

    # Build signals and sample the most recent bar at or before evaluation_date
    for csv_file_path, price_data_frame, symbol_mask in selected_symbol_data:
        # Build buy-side signals (support composite expressions)
        raw_buy_signal_columns: list[str] = []
        shifted_buy_signal_columns: list[str] = []
        for buy_name in buy_choice_names:
            try:
                (
                    base_name,
                    window_size,
                    angle_range,
                    near_range,
                    above_range,
                ) = parse_strategy_name(buy_name)
            except Exception:  # noqa: BLE001
                continue
            if base_name not in BUY_STRATEGIES:
                continue
            _apply_parsed_strategy(
                buy_name,
                base_name,
                window_size,
                angle_range,
                near_range,
                above_range,
                BUY_STRATEGIES,
                price_data_frame,
                include_raw_signals=use_unshifted_signals,
            )
            entry_column_name = f"{buy_name}_entry_signal"
            if use_unshifted_signals:
                raw_column_name = f"{buy_name}_raw_entry_signal"
                if raw_column_name in price_data_frame.columns:
                    raw_buy_signal_columns.append(raw_column_name)
                if entry_column_name in price_data_frame.columns:
                    shifted_buy_signal_columns.append(entry_column_name)
            else:
                if entry_column_name in price_data_frame.columns:
                    shifted_buy_signal_columns.append(entry_column_name)

        sell_signal_columns: list[str] = []
        for sell_name in sell_choice_names:
            try:
                (
                    base_name,
                    window_size,
                    angle_range,
                    near_range,
                    above_range,
                ) = parse_strategy_name(sell_name)
            except Exception:  # noqa: BLE001
                continue
            if base_name not in SELL_STRATEGIES:
                continue
            _apply_parsed_strategy(
                sell_name,
                base_name,
                window_size,
                angle_range,
                near_range,
                above_range,
                SELL_STRATEGIES,
                price_data_frame,
                include_raw_signals=use_unshifted_signals,
            )
            if use_unshifted_signals:
                column_name = f"{sell_name}_raw_exit_signal"
                if column_name in price_data_frame.columns:
                    sell_signal_columns.append(column_name)
                elif f"{sell_name}_exit_signal" in price_data_frame.columns:
                    sell_signal_columns.append(f"{sell_name}_exit_signal")
            else:
                column_name = f"{sell_name}_exit_signal"
                if column_name in price_data_frame.columns:
                    sell_signal_columns.append(column_name)

        # Combined columns (OR across choices)
        raw_buy_signal_columns = list(dict.fromkeys(raw_buy_signal_columns))
        shifted_buy_signal_columns = list(dict.fromkeys(shifted_buy_signal_columns))
        sell_signal_columns = list(dict.fromkeys(sell_signal_columns))
        if use_unshifted_signals:
            # Use raw (unshifted) signals directly.  The raw signal on day T
            # captures the same condition as the shifted signal on T+1, so
            # there is no need to realign via shift(-1).  The previous
            # shift(-1, fill_value=False) approach silently dropped the signal
            # when the evaluation date was the last bar in the data frame.
            if raw_buy_signal_columns:
                price_data_frame["_combined_buy_entry"] = (
                    price_data_frame[raw_buy_signal_columns]
                    .any(axis=1)
                    .fillna(False)
                    .astype(bool)
                )
            else:
                price_data_frame["_combined_buy_entry"] = False
        else:
            if shifted_buy_signal_columns:
                price_data_frame["_combined_buy_entry"] = (
                    price_data_frame[shifted_buy_signal_columns]
                    .any(axis=1)
                    .fillna(False)
                    .astype(bool)
                )
            else:
                price_data_frame["_combined_buy_entry"] = False
        if sell_signal_columns:
            price_data_frame["_combined_sell_exit"] = (
                price_data_frame[sell_signal_columns].any(axis=1).fillna(False)
            )
        else:
            price_data_frame["_combined_sell_exit"] = False

        # Sample the last available bar at or before evaluation_date
        eligible_index = price_data_frame.index[price_data_frame.index <= evaluation_date]
        if len(eligible_index) == 0:
            continue
        last_bar_timestamp = eligible_index[-1]
        current_row = price_data_frame.loc[last_bar_timestamp]

        # Entry requires signal AND eligibility on that bar
        if bool(current_row["_combined_buy_entry"]) and bool(symbol_mask.loc[last_bar_timestamp]):
            entry_signal_symbols.append(csv_file_path.stem)
        # Exit ignores eligibility so existing positions can close
        if bool(current_row["_combined_sell_exit"]):
            exit_signal_symbols.append(csv_file_path.stem)

    return {
        "filtered_symbols": filtered_symbols_with_groups,
        "entry_signals": entry_signal_symbols,
        "exit_signals": exit_signal_symbols,
    }


def load_price_data(csv_file_path: Path) -> pandas.DataFrame:
    """Load price data from ``csv_file_path`` and normalize column names.

    Duplicate dates are removed and the index is sorted to ensure that the
    resulting frame has unique, chronologically ordered entries. Column labels
    are converted to lowercase ``snake_case`` and common suffixes such as
    ``_price`` are stripped so that names like ``Adj Close`` or ``Close Price``
    become ``adj_close`` and ``close``. When several columns normalize to the
    same label, the first occurrence is retained and later duplicates are
    discarded. When the CSV file is empty, an empty data frame is returned so
    the caller can skip the symbol gracefully.
    """
    # TODO: review

    try:
        price_data_frame = pandas.read_csv(
            csv_file_path, parse_dates=["Date"], index_col="Date"
        )
    except pandas.errors.EmptyDataError:
        return pandas.DataFrame()
    except ValueError as value_error:
        # Gracefully handle files that do not include a 'Date' column by
        # attempting to infer a suitable date column.
        message_text = str(value_error)
        if "Missing column provided to 'parse_dates': 'Date'" in message_text:
            temp_frame = pandas.read_csv(csv_file_path)
            original_columns = list(temp_frame.columns)
            candidate_map = {name.lower(): name for name in original_columns}
            candidate_name: str | None = None
            for possible in ("date", "datetime", "timestamp"):
                if possible in candidate_map:
                    candidate_name = candidate_map[possible]
                    break
            if candidate_name is None and original_columns:
                candidate_name = original_columns[0]
            try:
                temp_frame[candidate_name] = pandas.to_datetime(
                    temp_frame[candidate_name]
                )
                price_data_frame = temp_frame.set_index(candidate_name)
            except Exception as parse_error:  # noqa: BLE001
                raise ValueError(
                    (
                        f"Could not locate a date column in {csv_file_path.name}; "
                        "expected a 'Date' column."
                    )
                ) from parse_error
        else:
            raise
    price_data_frame = price_data_frame.loc[
        ~price_data_frame.index.duplicated(keep="first")
    ]
    price_data_frame.sort_index(inplace=True)
    if isinstance(price_data_frame.columns, pandas.MultiIndex):
        price_data_frame.columns = price_data_frame.columns.get_level_values(0)
    price_data_frame.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(column_name).strip().lower())
        for column_name in price_data_frame.columns
    ]
    price_data_frame.columns = [
        re.sub(
            r"^_+",
            "",
            re.sub(
                r"(?:^|_)(open|close|high|low|volume)_.*",
                r"\1",
                column_name,
            ),
        )
        for column_name in price_data_frame.columns
    ]
    duplicate_column_mask = price_data_frame.columns.duplicated()
    if duplicate_column_mask.any():
        duplicate_column_names = price_data_frame.columns[
            duplicate_column_mask
        ].tolist()
        LOGGER.warning(
            "Duplicate column names %s found in %s; keeping first occurrence",
            duplicate_column_names,
            csv_file_path.name,
        )
        price_data_frame = price_data_frame.loc[:, ~duplicate_column_mask]
    required_columns = {"open", "close"}
    missing_column_names = [
        required_column
        for required_column in required_columns
        if required_column not in price_data_frame.columns
    ]
    if missing_column_names:
        missing_columns_string = ", ".join(missing_column_names)
        raise ValueError(
            f"Missing required columns: {missing_columns_string} in file {csv_file_path.name}"
        )
    return price_data_frame


def attach_ema_sma_cross_signals(
    price_data_frame: pandas.DataFrame,
    window_size: int = 40,
    require_close_above_long_term_sma: bool = False,
    sma_window_factor: float | None = None,
    include_raw_signals: bool = False,
) -> None:
    """Attach EMA/SMA cross entry and exit signals to ``price_data_frame``.

    Parameters
    ----------
    price_data_frame:
        DataFrame containing ``open`` and ``close`` price columns.
    window_size:
        Number of periods for EMA calculations.
    require_close_above_long_term_sma:
        When ``True``, entry signals are only generated if the previous day's
        closing price is greater than the 150-day simple moving average.
    sma_window_factor:
        Optional multiplier applied to ``window_size`` to determine the SMA
        window as ``ceil(window_size * factor)``. When ``None``, uses
        ``window_size`` for SMA as well.
    include_raw_signals:
        When ``True``, attach unshifted ``*_raw_entry_signal`` and
        ``*_raw_exit_signal`` columns representing same-day signals.
    """
    # TODO: review

    # Round close to 3 decimals before EMA/SMA to stabilize signals
    _close_r3 = price_data_frame["close"].round(3)
    price_data_frame["ema_value"] = ema(_close_r3, window_size)
    # Allow SMA window to be a factor of EMA window, ceiling
    if sma_window_factor is not None and sma_window_factor > 0:
        adjusted_sma_window: int = int(ceil(window_size * float(sma_window_factor)))
    else:
        adjusted_sma_window = int(window_size)
    price_data_frame["sma_value"] = sma(_close_r3, adjusted_sma_window)
    price_data_frame["long_term_sma_value"] = sma(
        _close_r3, LONG_TERM_SMA_WINDOW
    )
    price_data_frame["ema_previous"] = price_data_frame["ema_value"].shift(1)
    price_data_frame["sma_previous"] = price_data_frame["sma_value"].shift(1)
    price_data_frame["long_term_sma_previous"] = price_data_frame[
        "long_term_sma_value"
    ].shift(1)
    price_data_frame["close_previous"] = price_data_frame["close"].shift(1)
    ema_cross_up = (
        (price_data_frame["ema_previous"] <= price_data_frame["sma_previous"])
        & (price_data_frame["ema_value"] > price_data_frame["sma_value"])
    )
    ema_cross_down = (
        (price_data_frame["ema_previous"] >= price_data_frame["sma_previous"])
        & (price_data_frame["ema_value"] < price_data_frame["sma_value"])
    )
    base_entry_signal = ema_cross_up.shift(1, fill_value=False)
    if require_close_above_long_term_sma:
        price_data_frame["ema_sma_cross_entry_signal"] = (
            base_entry_signal
            & (
                price_data_frame["close_previous"]
                > price_data_frame["long_term_sma_previous"]
            )
        )
        if include_raw_signals:
            price_data_frame["ema_sma_cross_raw_entry_signal"] = (
                ema_cross_up
                & (
                    price_data_frame["close"]
                    > price_data_frame["long_term_sma_value"]
                )
            )
    else:
        price_data_frame["ema_sma_cross_entry_signal"] = base_entry_signal
        if include_raw_signals:
            price_data_frame["ema_sma_cross_raw_entry_signal"] = ema_cross_up
    price_data_frame["ema_sma_cross_exit_signal"] = ema_cross_down.shift(
        1, fill_value=False
    )
    if include_raw_signals:
        price_data_frame["ema_sma_cross_raw_exit_signal"] = ema_cross_down


def attach_20_50_sma_cross_signals(
    price_data_frame: pandas.DataFrame,
    short_window_size: int = 20,
    long_window_size: int = 50,
    include_raw_signals: bool = False,
) -> None:
    """Attach SMA cross entry/exit signals using configurable windows.

    By default this reproduces the classic 20/50 SMA cross. When invoked with
    ``short_window_size`` and ``long_window_size``, it uses those windows
    instead (e.g., 15/30).

    Parameters
    ----------
    price_data_frame:
        DataFrame containing a ``close`` column.
    short_window_size:
        Number of periods for the short simple moving average (default ``20``).
    long_window_size:
        Number of periods for the long simple moving average (default ``50``).
    include_raw_signals:
        When ``True``, attach unshifted ``*_raw_entry_signal`` and
        ``*_raw_exit_signal`` columns representing same-day signals.
    """
    # TODO: review

    if short_window_size <= 0 or long_window_size <= 0:
        raise ValueError("SMA window sizes must be positive integers")
    if short_window_size >= long_window_size:
        raise ValueError(
            "short_window_size must be smaller than long_window_size for a cross"
        )

    _close_r3 = price_data_frame["close"].round(3)
    price_data_frame["sma_20_value"] = sma(_close_r3, short_window_size)
    price_data_frame["sma_50_value"] = sma(_close_r3, long_window_size)
    price_data_frame["sma_20_previous"] = price_data_frame["sma_20_value"].shift(1)
    price_data_frame["sma_50_previous"] = price_data_frame["sma_50_value"].shift(1)
    sma_20_crosses_above_sma_50 = (
        (price_data_frame["sma_20_previous"] <= price_data_frame["sma_50_previous"])
        & (price_data_frame["sma_20_value"] > price_data_frame["sma_50_value"])
    )
    sma_20_crosses_below_sma_50 = (
        (price_data_frame["sma_20_previous"] >= price_data_frame["sma_50_previous"])
        & (price_data_frame["sma_20_value"] < price_data_frame["sma_50_value"])
    )
    price_data_frame["20_50_sma_cross_entry_signal"] = (
        sma_20_crosses_above_sma_50.shift(1, fill_value=False)
    )
    price_data_frame["20_50_sma_cross_exit_signal"] = (
        sma_20_crosses_below_sma_50.shift(1, fill_value=False)
    )
    if include_raw_signals:
        price_data_frame["20_50_sma_cross_raw_entry_signal"] = (
            sma_20_crosses_above_sma_50
        )
        price_data_frame["20_50_sma_cross_raw_exit_signal"] = (
            sma_20_crosses_below_sma_50
        )


## Removed deprecated strategies: ema_sma_cross_and_rsi, ftd_ema_sma_cross


def attach_ema_sma_cross_with_slope_signals(
    price_data_frame: pandas.DataFrame,
    window_size: int = 40,
    angle_range: tuple[float, float] = DEFAULT_SMA_ANGLE_RANGE,
    sma_window_factor: float | None = None,
    bounds_as_tangent: bool = False,
    include_raw_signals: bool = False,
) -> None:
    """Attach EMA/SMA cross signals filtered by simple moving average angle.

    Entry signals require the prior-day EMA cross, the simple moving average
    angle to fall within ``angle_range``, and the closing price to remain above
    the long-term simple moving average. Unless an angle range is provided in
    the strategy name, this function uses the default range derived from the
    tangents ``(-0.3, 2.14)`` converted to degrees. The normalized slope scales
    with ``window_size``; larger windows produce smaller relative changes, so
    adjust ``angle_range`` accordingly when overriding the default.

    Parameters
    ----------
    price_data_frame:
        DataFrame containing ``open`` and ``close`` price columns.
    window_size:
        Number of periods for EMA calculations.
    angle_range:
        Inclusive range ``(lower_bound, upper_bound)`` for the SMA angle in
        degrees. When ``bounds_as_tangent`` is ``True``, interpret the bounds as
        tangents and convert them to degrees.
    sma_window_factor:
        Optional multiplier applied to ``window_size`` to determine the SMA
        window as ``ceil(window_size * factor)``. When ``None``, uses
        ``window_size`` for SMA as well.
    bounds_as_tangent:
        When ``True``, interpret ``angle_range`` as tangent values instead of
        degrees.
    include_raw_signals:
        When ``True``, attach unshifted ``*_raw_entry_signal`` and
        ``*_raw_exit_signal`` columns representing same-day signals.

    Raises
    ------
    ValueError
        If ``angle_range`` has a lower bound greater than its upper bound.
    """
    # TODO: review

    angle_lower_bound, angle_upper_bound = angle_range
    if bounds_as_tangent:
        angle_lower_bound = math.degrees(math.atan(angle_lower_bound))
        angle_upper_bound = math.degrees(math.atan(angle_upper_bound))
    if angle_lower_bound > angle_upper_bound:
        raise ValueError(
            "Invalid angle_range: lower bound cannot exceed upper bound"
        )

    attach_ema_sma_cross_signals(
        price_data_frame,
        window_size,
        require_close_above_long_term_sma=True,
        sma_window_factor=sma_window_factor,
        include_raw_signals=include_raw_signals,
    )
    relative_change = (
        price_data_frame["sma_value"] - price_data_frame["sma_previous"]
    ) / price_data_frame["sma_previous"]
    price_data_frame["sma_angle"] = numpy.degrees(numpy.arctan(relative_change))
    sma_angle_previous = price_data_frame["sma_angle"].shift(1)
    price_data_frame["ema_sma_cross_with_slope_entry_signal"] = (
        price_data_frame["ema_sma_cross_entry_signal"]
        & (sma_angle_previous >= angle_lower_bound)
        & (sma_angle_previous <= angle_upper_bound)
    )
    price_data_frame["ema_sma_cross_with_slope_exit_signal"] = price_data_frame[
        "ema_sma_cross_exit_signal"
    ]
    if include_raw_signals:
        price_data_frame["ema_sma_cross_with_slope_raw_entry_signal"] = (
            price_data_frame["ema_sma_cross_raw_entry_signal"]
            & (price_data_frame["sma_angle"] >= angle_lower_bound)
            & (price_data_frame["sma_angle"] <= angle_upper_bound)
        )
        price_data_frame["ema_sma_cross_with_slope_raw_exit_signal"] = (
            price_data_frame["ema_sma_cross_raw_exit_signal"]
        )


def attach_ema_sma_cross_testing_signals(
    price_data_frame: pandas.DataFrame,
    window_size: int = 40,
    angle_range: tuple[float, float] = DEFAULT_SMA_ANGLE_RANGE,
    near_range: tuple[float, float] = (0.0, 0.12),
    above_range: tuple[float, float] = (0.0, 0.10),
    sma_window_factor: float | None = None,
    bounds_as_tangent: bool = False,
    include_raw_signals: bool = False,
    use_confirmation_angle: bool = False,
    d_sma_range: tuple[float, float] | None = None,
    ema_range: tuple[float, float] | None = None,
    d_ema_range: tuple[float, float] | None = None,
    near_delta_range: tuple[float, float] | None = None,
    price_tightness_range: tuple[float, float] | None = None,
    sma_150_angle_min: float | None = None,
    use_ftd_confirmation: bool = False,
    price_score_min: float | None = None,
    price_score_max: float | None = None,
    confirmation_sma_angle_range: tuple[float, float] | None = None,
    exit_alpha_factor: float | None = None,
    shape_slope_min: float | None = None,
    shape_dev_50_max: float | None = None,
    shape_bsv_lookback: int | None = None,
) -> None:
    """Attach EMA/SMA cross testing signals with angle and chip filters.

    Entry signals mirror :func:`attach_ema_sma_cross_with_slope_signals` but do
    not require the closing price to remain above the long-term simple moving
    average. Instead, this variant recomputes chip concentration metrics and
    requires that both the near-price and the above-price volume ratios on the
    crossover date fall within the inclusive ``near_range`` and ``above_range``
    bounds. The unshifted ratios are retained for ``*_raw_entry_signal``
    evaluation so same-day raw signals remain consistent.

    Parameters
    ----------
    price_data_frame:
        DataFrame containing ``open``, ``high``, ``low``, ``close`` and
        ``volume`` columns.
    window_size:
        Number of periods for EMA calculations.
    angle_range:
        Inclusive range ``(lower_bound, upper_bound)`` for the SMA angle in
        degrees. When ``bounds_as_tangent`` is ``True``, interpret the bounds as
        tangents and convert them to degrees.
    near_range:
        Inclusive ``(lower, upper)`` bounds for the near-price volume ratio.
    above_range:
        Inclusive ``(lower, upper)`` bounds for the above-price volume ratio.
    sma_window_factor:
        Optional multiplier applied to ``window_size`` to determine the SMA
        window as ``ceil(window_size * factor)``. When ``None``, uses
        ``window_size`` for the SMA as well.
    bounds_as_tangent:
        When ``True``, interpret ``angle_range`` as tangent values instead of
        degrees.
    include_raw_signals:
        When ``True``, attach unshifted ``*_raw_entry_signal`` and
        ``*_raw_exit_signal`` columns representing same-day signals.

    Raises
    ------
    ValueError
        If ``angle_range`` has a lower bound greater than its upper bound.
    """
    # TODO: review

    angle_lower_bound, angle_upper_bound = angle_range
    near_lower_bound, near_upper_bound = near_range
    above_lower_bound, above_upper_bound = above_range
    if bounds_as_tangent:
        angle_lower_bound = math.degrees(math.atan(angle_lower_bound))
        angle_upper_bound = math.degrees(math.atan(angle_upper_bound))
    if angle_lower_bound > angle_upper_bound:
        raise ValueError(
            "Invalid angle_range: lower bound cannot exceed upper bound"
        )

    attach_ema_sma_cross_signals(
        price_data_frame,
        window_size,
        require_close_above_long_term_sma=False,
        sma_window_factor=sma_window_factor,
        include_raw_signals=include_raw_signals,
    )
    relative_change = (
        price_data_frame["sma_value"] - price_data_frame["sma_previous"]
    ) / price_data_frame["sma_previous"]
    price_data_frame["sma_angle"] = numpy.degrees(numpy.arctan(relative_change))

    near_ratios: List[float | None] = []
    above_ratios: List[float | None] = []
    below_ratios: List[float | None] = []
    price_scores: List[float | None] = []
    for row_index in range(len(price_data_frame)):
        chip_metrics = calculate_chip_concentration_metrics(
            price_data_frame.iloc[: row_index + 1],
            lookback_window_size=60,
            include_volume_profile=False,
        )
        near_ratios.append(chip_metrics["near_price_volume_ratio"])
        above_ratios.append(chip_metrics["above_price_volume_ratio"])
        below_ratios.append(chip_metrics["below_price_volume_ratio"])
        price_scores.append(chip_metrics["price_score"])
    price_data_frame["near_price_volume_ratio"] = pandas.Series(
        near_ratios, index=price_data_frame.index
    )
    price_data_frame["above_price_volume_ratio"] = pandas.Series(
        above_ratios, index=price_data_frame.index
    )
    price_data_frame["below_price_volume_ratio"] = pandas.Series(
        below_ratios, index=price_data_frame.index
    )

    # VCP footprint metrics: supply dry-up leaves two traces over 60 bars.
    # near_delta = near_now - near_60_bars_ago.  Negative = near declining
    #   (floating supply being absorbed while price stays).
    # price_tightness = (rolling_high - rolling_low) / close.  Small value =
    #   price confined to a narrow range (base tightening).
    near_series = price_data_frame["near_price_volume_ratio"]
    price_data_frame["near_delta"] = near_series - near_series.shift(60)
    price_data_frame["price_tightness"] = (
        price_data_frame["high"].rolling(window=60, min_periods=1).max()
        - price_data_frame["low"].rolling(window=60, min_periods=1).min()
    ) / price_data_frame["close"]

    # Stage 2 indicator: 150-day SMA angle.  Positive = uptrend.
    sma_150 = price_data_frame["close"].rolling(window=150, min_periods=150).mean()
    sma_150_prev = sma_150.shift(1)
    sma_150_rel_change = (sma_150 - sma_150_prev) / sma_150_prev
    price_data_frame["sma_150_angle"] = numpy.degrees(numpy.arctan(sma_150_rel_change))

    price_data_frame["near_price_volume_ratio_previous"] = price_data_frame[
        "near_price_volume_ratio"
    ].shift(1)
    price_data_frame["above_price_volume_ratio_previous"] = price_data_frame[
        "above_price_volume_ratio"
    ].shift(1)

    near_price_ratio_previous_ok = (
        price_data_frame["near_price_volume_ratio_previous"].ge(near_lower_bound)
        & price_data_frame["near_price_volume_ratio_previous"].le(near_upper_bound)
    )
    above_price_ratio_previous_ok = (
        price_data_frame["above_price_volume_ratio_previous"].ge(above_lower_bound)
        & price_data_frame["above_price_volume_ratio_previous"].le(above_upper_bound)
    )

    # A-layer (signal-day, T): sma_angle filter ALWAYS uses the shifted
    # (previous-row) value so that at row T+1 we check sma_angle[T]. This
    # is the signal-quality check decoupled from execution confirmation.
    sma_angle_previous = price_data_frame["sma_angle"].shift(1)
    entry_conditions = (
        price_data_frame["ema_sma_cross_entry_signal"]
        & (sma_angle_previous >= angle_lower_bound)
        & (sma_angle_previous <= angle_upper_bound)
        & (
            near_price_ratio_previous_ok.fillna(False)
            & above_price_ratio_previous_ok.fillna(False)
        )
    )
    # B-layer (confirmation-day, T+1): when confirmation mode is enabled,
    # apply an ADDITIONAL sma_angle check on the execution bar. Paired
    # with pending_limit_entry / pending_market_entry in the simulator so
    # that actual entry is delayed to T+2 via limit or market order,
    # avoiding lookahead. The range defaults to
    # CONFIRMATION_SMA_ANGLE_RANGE but can be overridden per-run via the
    # ``confirmation_sma_angle_range`` parameter.
    if use_confirmation_angle:
        confirmation_range = (
            confirmation_sma_angle_range
            if confirmation_sma_angle_range is not None
            else CONFIRMATION_SMA_ANGLE_RANGE
        )
        confirmation_lower, confirmation_upper = confirmation_range
        entry_conditions = entry_conditions & (
            price_data_frame["sma_angle"].ge(confirmation_lower).fillna(False)
            & price_data_frame["sma_angle"].le(confirmation_upper).fillna(False)
        )

    # Optional d_sma_angle filter — always uses T (the signal date) since
    # the derivative sma_angle[T] - sma_angle[T-1] is known at T close,
    # before the T+1 open decision point.
    if d_sma_range is not None:
        d_sma = price_data_frame["sma_angle"].diff()
        d_sma_series = d_sma.shift(1)
        entry_conditions = entry_conditions & (
            d_sma_series.ge(d_sma_range[0]).fillna(False)
            & d_sma_series.le(d_sma_range[1]).fillna(False)
        )

    # Optional ema_angle filter — always uses T (the signal date) to assess
    # signal quality.  The value ema_angle[T] is known at T close, before the
    # T+1 open decision point.
    if ema_range is not None:
        if "ema_value" in price_data_frame.columns:
            ema_prev = price_data_frame["ema_value"].shift(1)
            ema_rel_change = (price_data_frame["ema_value"] - ema_prev) / ema_prev
            ema_angle = numpy.degrees(numpy.arctan(ema_rel_change))
            ema_series = ema_angle.shift(1)
            entry_conditions = entry_conditions & (
                ema_series.ge(ema_range[0]).fillna(False)
                & ema_series.le(ema_range[1]).fillna(False)
            )

    # Optional d_ema_angle filter — always uses T (signal date). The
    # derivative d_ema[T] = ema_angle[T] - ema_angle[T-1] is known at T close.
    if d_ema_range is not None:
        if "ema_value" in price_data_frame.columns:
            ema_prev_for_d = price_data_frame["ema_value"].shift(1)
            ema_rel_change_for_d = (
                price_data_frame["ema_value"] - ema_prev_for_d
            ) / ema_prev_for_d
            ema_angle_for_d = numpy.degrees(numpy.arctan(ema_rel_change_for_d))
            d_ema_series = ema_angle_for_d.diff().shift(1)
            entry_conditions = entry_conditions & (
                d_ema_series.ge(d_ema_range[0]).fillna(False)
                & d_ema_series.le(d_ema_range[1]).fillna(False)
            )

    # Optional near_delta filter — uses T (signal date).  near_delta[T] is
    # known at T close (computed from near[T] - near[T-60]).
    if near_delta_range is not None:
        nd_series = price_data_frame["near_delta"].shift(1)
        entry_conditions = entry_conditions & (
            nd_series.ge(near_delta_range[0]).fillna(False)
            & nd_series.le(near_delta_range[1]).fillna(False)
        )

    # Optional price_tightness filter — uses T (signal date).
    # price_tightness[T] = (rolling_high - rolling_low) / close over 60 bars.
    if price_tightness_range is not None:
        pt_series = price_data_frame["price_tightness"].shift(1)
        entry_conditions = entry_conditions & (
            pt_series.ge(price_tightness_range[0]).fillna(False)
            & pt_series.le(price_tightness_range[1]).fillna(False)
        )

    # Optional Stage 2 gate: 150-day SMA must be trending up (angle > min).
    if sma_150_angle_min is not None:
        sma_150_series = price_data_frame["sma_150_angle"].shift(1)
        entry_conditions = entry_conditions & (
            sma_150_series.ge(sma_150_angle_min).fillna(False)
        )

    # Optional price_concentration_score filter (checked on T, the signal date)
    if price_score_min is not None or price_score_max is not None:
        price_score_series = pandas.Series(
            price_scores, index=price_data_frame.index
        ).shift(1)  # Use T (previous bar)
        if price_score_min is not None:
            entry_conditions = entry_conditions & (
                price_score_series.ge(price_score_min).fillna(False)
            )
        if price_score_max is not None:
            entry_conditions = entry_conditions & (
                price_score_series.le(price_score_max).fillna(False)
            )

    # Follow Through Day (FTD) confirmation gate.
    # When use_ftd_confirmation is True, entry signals require a FTD pattern
    # within a lookback window: bottom formed, ascending lows, rising volume.
    if use_ftd_confirmation:
        _low = price_data_frame["low"]
        _close = price_data_frame["close"]
        _open = price_data_frame["open"]
        _vol = price_data_frame["volume"]
        _ma50 = _close.rolling(window=50, min_periods=50).mean()

        # 4 bars ago was the lowest low in 23 bars
        _low_4ago = _low.shift(3)
        _lowest_23 = _low.rolling(window=23, min_periods=1).min()
        bottom_check = _low_4ago == _lowest_23.shift(3)

        # Ascending lows over 4 bars: low1 > low2 > low3 > low4
        low_check = (
            (_low > _low.shift(1))
            & (_low.shift(1) > _low.shift(2))
            & (_low.shift(2) > _low.shift(3))
        )

        # Volume: sum of last 4 bars > sum of previous 4 bars
        _vol_sum_4 = _vol.rolling(window=4, min_periods=4).sum()
        vol_check = _vol_sum_4 > _vol_sum_4.shift(4)

        # MA check: 4 bars ago close was below MA(50)
        ma_check = _close.shift(3) < _ma50.shift(3)

        ftd_signal = bottom_check & low_check & vol_check & ma_check
        ftd_signal = ftd_signal.fillna(False)

        # FTD must have occurred within the last ftd_lookback bars
        ftd_lookback = 10
        ftd_recent = ftd_signal.rolling(window=ftd_lookback, min_periods=1).max().astype(bool)
        entry_conditions = entry_conditions & ftd_recent

    # Fish-body shape + BSV gate (third instance: trend join after vacuum
    # turn). Active when all three params are set. Gates entry on:
    # 1) 60-bar shape descriptor's slope >= shape_slope_min
    # 2) ALL three middle samples (25%, 50%, 75%) deviation <=
    #    shape_dev_50_max (every interior point sits at-or-below the
    #    head-to-tail baseline — full concave U-shape, not just midpoint dip)
    # 3) BSV footprint observed in last shape_bsv_lookback bars
    if (
        shape_slope_min is not None
        and shape_dev_50_max is not None
        and shape_bsv_lookback is not None
    ):
        shape_window_size = 60
        sample_count = 5
        close_series = price_data_frame["close"].astype(float)
        slope_pass = pandas.Series(False, index=price_data_frame.index)
        dev_pass = pandas.Series(False, index=price_data_frame.index)
        for row_index in range(shape_window_size - 1, len(price_data_frame)):
            window_start = row_index - (shape_window_size - 1)
            window_slice = close_series.iloc[window_start : row_index + 1]
            head_value = float(window_slice.iloc[0])
            if head_value == 0 or pandas.isna(head_value):
                continue
            sample_indices = [
                int(round(position * (shape_window_size - 1) / (sample_count - 1)))
                for position in range(sample_count)
            ]
            samples = [float(window_slice.iloc[idx]) for idx in sample_indices]
            head = samples[0]
            tail = samples[-1]
            slope_value = (tail - head) / head
            if slope_value >= shape_slope_min:
                slope_pass.iloc[row_index] = True
            # Check all middle samples (positions 1 .. sample_count - 2)
            # against the head-to-tail linear baseline. Every interior
            # deviation must be <= shape_dev_50_max.
            all_below = True
            for sample_index in range(1, sample_count - 1):
                position_fraction = sample_index / (sample_count - 1)
                baseline_at_sample = head + (tail - head) * position_fraction
                sample_deviation = (
                    samples[sample_index] - baseline_at_sample
                ) / abs(head)
                if sample_deviation > shape_dev_50_max:
                    all_below = False
                    break
            if all_below:
                dev_pass.iloc[row_index] = True
        # Shape pass uses T (signal date) — shift by 1 so signal at T+1 reads T's shape.
        shape_pass_at_t = (slope_pass & dev_pass).shift(1, fill_value=False)

        # BSV footprint detection
        bsv_frame = bsv(
            high_series=price_data_frame["high"],
            low_series=price_data_frame["low"],
            close_series=price_data_frame["close"],
            volume_series=price_data_frame["volume"],
        )
        bsv_active = (bsv_frame["footprint_flag"] > 0).fillna(False)
        # BSV in last shape_bsv_lookback bars (inclusive of T)
        bsv_recent_at_t = (
            bsv_active.rolling(window=shape_bsv_lookback, min_periods=1)
            .max()
            .astype(bool)
            .shift(1, fill_value=False)
        )

        entry_conditions = entry_conditions & shape_pass_at_t & bsv_recent_at_t

    price_data_frame["ema_sma_cross_testing_entry_signal"] = entry_conditions

    # Exit signal: heavy-alpha EMA cross down SMA, or default cross exit
    if exit_alpha_factor is not None and exit_alpha_factor > 0:
        # CustomEMA(α_heavy) crosses below SMA — same structure as Level 1
        # entry (EMA up-cross SMA) but with heavier α for faster response
        _close_r3 = price_data_frame["close"].round(3)
        heavy_alpha = exit_alpha_factor / (window_size + 1)
        heavy_alpha = min(heavy_alpha, 1.0)  # clamp to valid range
        ema_heavy = _close_r3.ewm(alpha=heavy_alpha, adjust=False).mean()
        ema_heavy_previous = ema_heavy.shift(1)
        sma_current = price_data_frame["sma_value"]
        sma_previous = price_data_frame["sma_previous"]
        # Fire when heavy EMA crosses below SMA
        heavy_cross_down = (ema_heavy_previous >= sma_previous) & (ema_heavy < sma_current)
        price_data_frame["ema_sma_cross_testing_exit_signal"] = (
            heavy_cross_down.shift(1, fill_value=False)
        )
    else:
        price_data_frame["ema_sma_cross_testing_exit_signal"] = price_data_frame[
            "ema_sma_cross_exit_signal"
        ]

    if include_raw_signals:
        near_price_ratio_raw_ok = (
            price_data_frame["near_price_volume_ratio"].ge(near_lower_bound)
            & price_data_frame["near_price_volume_ratio"].le(near_upper_bound)
        )
        above_price_ratio_raw_ok = (
            price_data_frame["above_price_volume_ratio"].ge(above_lower_bound)
            & price_data_frame["above_price_volume_ratio"].le(above_upper_bound)
        )
        price_data_frame["ema_sma_cross_testing_raw_entry_signal"] = (
            price_data_frame["ema_sma_cross_raw_entry_signal"]
            & (price_data_frame["sma_angle"] >= angle_lower_bound)
            & (price_data_frame["sma_angle"] <= angle_upper_bound)
            & (
                near_price_ratio_raw_ok.fillna(False)
                & above_price_ratio_raw_ok.fillna(False)
            )
        )
        price_data_frame["ema_sma_cross_testing_raw_exit_signal"] = (
            price_data_frame["ema_sma_cross_raw_exit_signal"]
        )


def attach_ema_shift_cross_with_slope_signals(
    price_data_frame: pandas.DataFrame,
    window_size: int = 35,
    angle_range: tuple[float, float] = DEFAULT_SHIFTED_EMA_ANGLE_RANGE,
    bounds_as_tangent: bool = False,
    include_raw_signals: bool = False,
) -> None:
    """Attach EMA/EMA(shifted) cross signals filtered by EMA angle.

    This strategy mirrors :func:`attach_ema_sma_cross_with_slope_signals` but
    replaces the simple moving average with an exponential moving average
    computed from the closing prices shifted back by three trading days.

    Entry conditions:
    - Previous day's EMA crosses above the shifted EMA
    - The shifted EMA angle falls within ``angle_range``

    Exit condition:
    - Previous day's EMA crosses below the shifted EMA

    Parameters
    ----------
    price_data_frame:
        DataFrame containing at least ``open`` and ``close`` columns.
    window_size:
        Number of periods for both EMAs. Defaults to ``35`` when not specified
        in the strategy name.
    angle_range:
        Inclusive range ``(lower_bound, upper_bound)`` for the shifted EMA angle
        in degrees. When ``bounds_as_tangent`` is ``True``, interpret bounds as
        tangents and convert them to degrees.
    bounds_as_tangent:
        When ``True``, interpret ``angle_range`` as tangent bounds rather than
        degrees.
    include_raw_signals:
        When ``True``, attach unshifted ``*_raw_entry_signal`` and
        ``*_raw_exit_signal`` columns representing same-day signals.

    Raises
    ------
    ValueError
        If ``angle_range`` has a lower bound greater than its upper bound.
    """
    angle_lower_bound, angle_upper_bound = angle_range
    if bounds_as_tangent:
        angle_lower_bound = math.degrees(math.atan(angle_lower_bound))
        angle_upper_bound = math.degrees(math.atan(angle_upper_bound))
    if angle_lower_bound > angle_upper_bound:
        raise ValueError(
            "Invalid angle_range: lower bound cannot exceed upper bound",
        )

    _close_r3 = price_data_frame["close"].round(3)
    price_data_frame["ema_value"] = ema(_close_r3, window_size)
    price_data_frame["shifted_close"] = price_data_frame["close"].round(3).shift(3)
    price_data_frame["shifted_ema_value"] = ema(
        price_data_frame["shifted_close"], window_size
    )

    price_data_frame["ema_previous"] = price_data_frame["ema_value"].shift(1)
    price_data_frame["shifted_ema_previous"] = price_data_frame["shifted_ema_value"].shift(1)

    crosses_up = (
        (price_data_frame["ema_previous"] <= price_data_frame["shifted_ema_previous"])
        & (price_data_frame["ema_value"] > price_data_frame["shifted_ema_value"])
    )
    crosses_down = (
        (price_data_frame["ema_previous"] >= price_data_frame["shifted_ema_previous"])
        & (price_data_frame["ema_value"] < price_data_frame["shifted_ema_value"])
    )

    relative_change = (
        price_data_frame["shifted_ema_value"] - price_data_frame["shifted_ema_previous"]
    ) / price_data_frame["shifted_ema_previous"]
    price_data_frame["shifted_ema_angle"] = numpy.degrees(
        numpy.arctan(relative_change)
    )

    base_entry = crosses_up.shift(1, fill_value=False)
    base_exit = crosses_down.shift(1, fill_value=False)

    shifted_ema_angle_previous = price_data_frame["shifted_ema_angle"].shift(1)
    price_data_frame["ema_shift_cross_with_slope_entry_signal"] = (
        base_entry
        & (shifted_ema_angle_previous >= angle_lower_bound)
        & (shifted_ema_angle_previous <= angle_upper_bound)
    )
    price_data_frame["ema_shift_cross_with_slope_exit_signal"] = base_exit
    if include_raw_signals:
        price_data_frame["ema_shift_cross_with_slope_raw_entry_signal"] = (
            crosses_up
            & (price_data_frame["shifted_ema_angle"] >= angle_lower_bound)
            & (price_data_frame["shifted_ema_angle"] <= angle_upper_bound)
        )
        price_data_frame["ema_shift_cross_with_slope_raw_exit_signal"] = (
            crosses_down
        )

## Removed deprecated strategy: ema_sma_cross_with_slope_and_volume


## Removed deprecated strategy: ema_sma_double_cross


## Removed deprecated strategy: kalman_filtering

# TODO: review
BUY_STRATEGIES: Dict[str, Callable[..., None]] = {
    "ema_sma_cross": attach_ema_sma_cross_signals,
    "20_50_sma_cross": attach_20_50_sma_cross_signals,
    "ema_sma_cross_with_slope": attach_ema_sma_cross_with_slope_signals,
    "ema_sma_cross_testing": attach_ema_sma_cross_testing_signals,
    "ema_shift_cross_with_slope": attach_ema_shift_cross_with_slope_signals,
}

# TODO: review
SELL_STRATEGIES: Dict[str, Callable[..., None]] = {
    **BUY_STRATEGIES,
}

# TODO: review
SUPPORTED_STRATEGIES: Dict[str, Callable[..., None]] = {
    **SELL_STRATEGIES,
}


def parse_strategy_name(
    strategy_name: str,
) -> tuple[
    str,
    int | None,
    tuple[float, float] | None,
    tuple[float, float] | None,
    tuple[float, float] | None,
]:
    """Split ``strategy_name`` into base name and numeric suffix values.

    Strategy identifiers may include a numeric window size suffix, an angle
    range, and optional percentage thresholds. These numeric components appear
    after the base name separated by underscores. Supported patterns are:

    ``base``
        No numeric segments.
    ``base_40``
        Window size only.
    ``base_-1.0_2.0``
        Angle range only.
    ``base_40_-1.0_2.0``
        Window size and angle range.
    ``base_40_-1.0_2.0_0.5_1.5``
        Window size, angle range, percentage ``near`` and ``above`` thresholds.

    Any optional trailing ``_sma{factor}`` suffix (e.g., ``_sma1.2``) is ignored
    for base-name/window/angle parsing and should be obtained via
    :func:`_extract_sma_factor` if needed.

    Parameters
    ----------
    strategy_name:
        The full strategy name possibly containing a numeric suffix and an
        optional angle range with thresholds.

    Returns
    -------
    tuple[
        str,
        int | None,
        tuple[float, float] | None,
        tuple[float, float] | None,
        tuple[float, float] | None,
    ]
        ``(base_name, window_size, angle_range, near_range, above_range)``.
        ``angle_range`` is a ``(lower, upper)`` tuple in degrees. ``near_range``
        and ``above_range`` are inclusive ``(lower, upper)`` tuples parsed from
        ``"min,max"`` segments when present. Missing components yield ``None``.

    Raises
    ------
    ValueError
        If the strategy name ends with an underscore, specifies a non-positive
        window size, or contains an unexpected number of numeric segments.
    """
    # Strip optional trailing "_sma{factor}" before numeric parsing
    stripped_name = strategy_name
    sma_suffix_match = re.search(r"^(.*)_sma([0-9]+(?:\.[0-9]+)?)$", strategy_name)
    if sma_suffix_match:
        stripped_name = sma_suffix_match.group(1)

    name_parts = stripped_name.split("_")
    if "" in name_parts:
        raise ValueError(f"Malformed strategy name: {strategy_name}")

    numeric_segments: list[str] = []

    def _is_numeric_or_range(text: str) -> bool:
        try:
            float(text)
            return True
        except ValueError:
            if "," in text:
                lower_text, upper_text = text.split(",", 1)
                try:
                    float(lower_text)
                    float(upper_text)
                    return True
                except ValueError:
                    return False
            return False

    while name_parts:
        segment = name_parts[-1]
        if not _is_numeric_or_range(segment):
            break
        numeric_segments.append(segment)
        name_parts.pop()
    numeric_segments.reverse()

    base_name = "_".join(name_parts)
    segment_count = len(numeric_segments)
    if segment_count == 0:
        return base_name, None, None, None, None

    if segment_count == 1:
        numeric_value = numeric_segments[0]
        if numeric_value.isdigit():
            window_size = int(numeric_value)
            if window_size <= 0:
                raise ValueError(
                    "Window size must be a positive integer in strategy name: "
                    f"{strategy_name}"
                )
        return base_name, window_size, None, None, None
        raise ValueError(
            "Malformed strategy name: expected two numeric segments for angle range "
            f"but found {segment_count} in '{strategy_name}'"
        )

    if segment_count == 2:
        lower_bound, upper_bound = (
            float(numeric_segments[0]),
            float(numeric_segments[1]),
        )
        return base_name, None, (lower_bound, upper_bound), None, None

    if segment_count == 3:
        window_value = numeric_segments[0]
        if not window_value.isdigit():
            raise ValueError(
                "Malformed strategy name: expected two numeric segments for angle range "
                f"but found {segment_count} in '{strategy_name}'"
            )
        window_size = int(window_value)
        if window_size <= 0:
            raise ValueError(
                "Window size must be a positive integer in strategy name: "
                f"{strategy_name}"
            )
        lower_bound, upper_bound = (
            float(numeric_segments[1]),
            float(numeric_segments[2]),
        )
        return base_name, window_size, (lower_bound, upper_bound), None, None

    if segment_count == 5:
        window_value = numeric_segments[0]
        if not window_value.isdigit():
            raise ValueError(
                "Malformed strategy name: expected window size as first numeric segment "
                f"in '{strategy_name}'"
            )
        window_size = int(window_value)
        if window_size <= 0:
            raise ValueError(
                "Window size must be a positive integer in strategy name: "
                f"{strategy_name}"
            )
        lower_bound, upper_bound = (
            float(numeric_segments[1]),
            float(numeric_segments[2]),
        )

        def _parse_range(segment: str) -> tuple[float, float]:
            if "," in segment:
                lower_str, upper_str = segment.split(",", 1)
                return float(lower_str), float(upper_str)
            value = float(segment)
            return 0.0, value

        near_range = _parse_range(numeric_segments[3])
        above_range = _parse_range(numeric_segments[4])
        return (
            base_name,
            window_size,
            (lower_bound, upper_bound),
            near_range,
            above_range,
        )

    raise ValueError(
        "Malformed strategy name: expected up to five numeric segments but "
        f"found {segment_count} in '{strategy_name}'"
    )



def calculate_metrics(
    trade_profit_list: List[float],
    profit_percentage_list: List[float],
    loss_percentage_list: List[float],
    holding_period_list: List[int],
    maximum_concurrent_positions: int = 0,
    maximum_drawdown: float = 0.0,
    final_balance: float = 0.0,
    compound_annual_growth_rate: float = 0.0,
    annual_returns: Dict[int, float] | None = None,
    annual_trade_counts: Dict[int, int] | None = None,
    trade_details_by_year: Dict[int, List[TradeDetail]] | None = None,
) -> StrategyMetrics:
    """Compute summary metrics for a list of simulated trades, including CAGR."""
    # TODO: review

    total_trades = len(trade_profit_list)
    if total_trades == 0:
        return StrategyMetrics(
            total_trades=0,
            win_rate=0.0,
            mean_profit_percentage=0.0,
            profit_percentage_standard_deviation=0.0,
            mean_loss_percentage=0.0,
            loss_percentage_standard_deviation=0.0,
            mean_holding_period=0.0,
            holding_period_standard_deviation=0.0,
            maximum_concurrent_positions=maximum_concurrent_positions,
            maximum_drawdown=maximum_drawdown,
            final_balance=final_balance,
            compound_annual_growth_rate=compound_annual_growth_rate,
            annual_returns={} if annual_returns is None else annual_returns,
            annual_trade_counts={} if annual_trade_counts is None else annual_trade_counts,
            trade_details_by_year=
                {} if trade_details_by_year is None else trade_details_by_year,
        )

    winning_trade_count = sum(
        1 for profit_amount in trade_profit_list if profit_amount > 0
    )
    win_rate = winning_trade_count / total_trades

    def calculate_mean(values: List[float]) -> float:
        return mean(values) if values else 0.0

    def calculate_standard_deviation(values: List[float]) -> float:
        return stdev(values) if len(values) > 1 else 0.0

    return StrategyMetrics(
        total_trades=total_trades,
        win_rate=win_rate,
        mean_profit_percentage=calculate_mean(profit_percentage_list),
        profit_percentage_standard_deviation=calculate_standard_deviation(
            profit_percentage_list
        ),
        mean_loss_percentage=calculate_mean(loss_percentage_list),
        loss_percentage_standard_deviation=calculate_standard_deviation(
            loss_percentage_list
        ),
        mean_holding_period=calculate_mean(
            [float(value) for value in holding_period_list]
        ),
        holding_period_standard_deviation=calculate_standard_deviation(
            [float(value) for value in holding_period_list]
        ),
        maximum_concurrent_positions=maximum_concurrent_positions,
        maximum_drawdown=maximum_drawdown,
        final_balance=final_balance,
        compound_annual_growth_rate=compound_annual_growth_rate,
        annual_returns={} if annual_returns is None else annual_returns,
        annual_trade_counts={} if annual_trade_counts is None else annual_trade_counts,
        trade_details_by_year=
            {} if trade_details_by_year is None else trade_details_by_year,
    )


def _generate_strategy_evaluation_artifacts(
    data_directory: Path,
    buy_strategy_name: str,
    sell_strategy_name: str,
    minimum_average_dollar_volume: float | None = None,
    top_dollar_volume_rank: int | None = None,
    maximum_symbols_per_group: int = 1,
    minimum_average_dollar_volume_ratio: float | None = None,
    start_date: pandas.Timestamp | None = None,
    maximum_position_count: int = 3,
    allowed_fama_french_groups: set[int] | None = None,
    allowed_symbols: set[str] | None = None,
    exclude_other_ff12: bool = True,
    stop_loss_percentage: float = 1.0,
    take_profit_percentage: float = 0.0,
    minimum_holding_bars: int = 0,
    use_confirmation_angle: bool = False,
    confirmation_entry_mode: str = "limit",
    margin_multiplier: float = 1.0,
    margin_interest_annual_rate: float = 0.048,
    d_sma_range: tuple[float, float] | None = None,
    ema_range: tuple[float, float] | None = None,
    d_ema_range: tuple[float, float] | None = None,
    near_delta_range: tuple[float, float] | None = None,
    price_tightness_range: tuple[float, float] | None = None,
    sma_150_angle_min: float | None = None,
    use_ftd_confirmation: bool = False,
    trailing_stop_percentage: float = 0.0,
    price_score_min: float | None = None,
    price_score_max: float | None = None,
    confirmation_sma_angle_range: tuple[float, float] | None = None,
    exit_alpha_factor: float | None = None,
    shape_slope_min: float | None = None,
    shape_dev_50_max: float | None = None,
    shape_bsv_lookback: int | None = None,
    reentry_on_signal: bool = False,
) -> StrategyEvaluationArtifacts:
    """Build intermediate artifacts for strategy evaluation.

    The helper encapsulates the heavy lifting originally performed directly by
    :func:`evaluate_combined_strategy`. It prepares all simulation results,
    trade metadata, and trade detail records so callers can compute metrics
    tailored to their needs, such as enforcing custom position caps.
    """

    buy_choice_names = _split_strategy_choices(buy_strategy_name)
    sell_choice_names = _split_strategy_choices(sell_strategy_name)

    def _has_supported(tokens: list[str], table: dict) -> bool:
        for token in tokens:
            try:
                base, _, _, _, _ = parse_strategy_name(token)
            except Exception:  # noqa: BLE001
                continue
            if base in table:
                return True
        return False

    if not _has_supported(buy_choice_names, BUY_STRATEGIES):
        raise ValueError(f"Unsupported strategy: {buy_strategy_name}")
    if not _has_supported(sell_choice_names, SELL_STRATEGIES):
        raise ValueError(f"Unsupported strategy: {sell_strategy_name}")

    if (
        minimum_average_dollar_volume is not None
        and minimum_average_dollar_volume_ratio is not None
    ):
        raise ValueError(
            "Specify either minimum_average_dollar_volume or "
            "minimum_average_dollar_volume_ratio, not both",
        )

    simulation_results: List[SimulationResult] = []
    all_trades: List[Trade] = []
    simulation_start_date: pandas.Timestamp | None = None
    trade_symbol_lookup: Dict[Trade, str] = {}
    closing_price_series_by_symbol: Dict[str, pandas.Series] = {}
    trade_detail_pairs: Dict[Trade, Tuple[TradeDetail, TradeDetail]] = {}

    symbol_frames: List[tuple[Path, pandas.DataFrame]] = []
    symbols_excluded_by_industry = (
        load_symbols_excluded_by_industry() if exclude_other_ff12 else set()
    )
    symbol_to_group_map_for_filtering: dict[str, int] | None = None
    if allowed_fama_french_groups is not None:
        symbol_to_group_map_for_filtering = load_ff12_groups_by_symbol()
    for csv_file_path in data_directory.glob("*.csv"):
        if csv_file_path.stem == SP500_SYMBOL:
            continue
        if allowed_symbols is not None and csv_file_path.stem not in allowed_symbols:
            continue
        if csv_file_path.stem.upper() in symbols_excluded_by_industry:
            continue
        if symbol_to_group_map_for_filtering is not None:
            group_identifier = symbol_to_group_map_for_filtering.get(
                csv_file_path.stem.upper()
            )
            if group_identifier is None or group_identifier not in allowed_fama_french_groups:
                continue
        price_data_frame = load_price_data(csv_file_path)
        if price_data_frame.empty:
            continue
        if "volume" in price_data_frame.columns:
            dollar_volume_series_full = (
                price_data_frame["close"] * price_data_frame["volume"]
            )
            price_data_frame["simple_moving_average_dollar_volume"] = sma(
                dollar_volume_series_full, DOLLAR_VOLUME_SMA_WINDOW
            )
        else:
            if (
                minimum_average_dollar_volume is not None
                or top_dollar_volume_rank is not None
            ):
                raise ValueError(
                    "Volume column is required to compute dollar volume metrics"
                )
            price_data_frame["simple_moving_average_dollar_volume"] = float("nan")
        symbol_frames.append((csv_file_path, price_data_frame))

    if symbol_frames:
        merged_volume_frame = pandas.concat(
            {
                csv_path.stem: frame["simple_moving_average_dollar_volume"]
                for csv_path, frame in symbol_frames
            },
            axis=1,
        )
        eligibility_mask = _build_eligibility_mask(
            merged_volume_frame,
            minimum_average_dollar_volume=minimum_average_dollar_volume,
            top_dollar_volume_rank=top_dollar_volume_rank,
            minimum_average_dollar_volume_ratio=minimum_average_dollar_volume_ratio,
            maximum_symbols_per_group=maximum_symbols_per_group,
        )
    else:
        merged_volume_frame = pandas.DataFrame()
        eligibility_mask = pandas.DataFrame()

    market_total_dollar_volume_by_date = (
        merged_volume_frame.sum(axis=1).to_dict()
    )

    symbol_to_fama_french_group_id_for_details = load_ff12_groups_by_symbol()
    group_total_dollar_volume_by_group_and_date: dict[int, dict[pandas.Timestamp, float]] = {}
    if not merged_volume_frame.empty and symbol_to_fama_french_group_id_for_details:
        group_id_to_symbol_columns_for_details: dict[int, list[str]] = {}
        for column_name in merged_volume_frame.columns:
            group_identifier = symbol_to_fama_french_group_id_for_details.get(
                column_name.upper()
            )
            if group_identifier is None:
                continue
            group_id_to_symbol_columns_for_details.setdefault(
                group_identifier, []
            ).append(column_name)
        for group_identifier, column_names in group_id_to_symbol_columns_for_details.items():
            group_frame = merged_volume_frame[column_names]
            group_totals = group_frame.sum(axis=1)
            group_total_dollar_volume_by_group_and_date[group_identifier] = (
                group_totals.to_dict()
            )

    selected_symbol_data: List[tuple[Path, pandas.DataFrame, pandas.Series]] = []
    first_eligible_dates: List[pandas.Timestamp] = []
    simple_moving_average_dollar_volume_by_symbol_and_date: dict[str, dict[pandas.Timestamp, float]] = {}
    for csv_file_path, price_data_frame in symbol_frames:
        symbol_name = csv_file_path.stem
        if eligibility_mask.empty or symbol_name not in eligibility_mask.columns:
            symbol_mask = pandas.Series(False, index=price_data_frame.index)
        else:
            symbol_mask = eligibility_mask[symbol_name].reindex(
                price_data_frame.index, fill_value=False
            )
        if not symbol_mask.any():
            continue
        selected_symbol_data.append((csv_file_path, price_data_frame, symbol_mask))
        simple_moving_average_dollar_volume_by_symbol_and_date[symbol_name] = (
            price_data_frame["simple_moving_average_dollar_volume"].to_dict()
        )
        first_eligible_dates.append(symbol_mask[symbol_mask].index.min())

    if first_eligible_dates:
        earliest_eligible_date = min(first_eligible_dates)
        if start_date is not None:
            simulation_start_date = max(start_date, earliest_eligible_date)
        else:
            simulation_start_date = earliest_eligible_date
    else:
        simulation_start_date = start_date

    for csv_file_path, price_data_frame, symbol_mask in selected_symbol_data:
        buy_signal_columns: list[str] = []
        buy_bases_for_cooldown: set[str] = set()
        for buy_name in _split_strategy_choices(buy_strategy_name):
            try:
                (
                    base_name,
                    window_size,
                    angle_range,
                    near_range,
                    above_range,
                ) = parse_strategy_name(buy_name)
            except Exception:
                continue
            if base_name not in BUY_STRATEGIES:
                continue
            buy_bases_for_cooldown.add(base_name)
            buy_function = BUY_STRATEGIES[base_name]
            kwargs: dict = {}
            if base_name == "20_50_sma_cross":
                short_long = _extract_short_long_windows_for_20_50(buy_name)
                if short_long is not None:
                    kwargs["short_window_size"], kwargs["long_window_size"] = short_long
            else:
                if window_size is not None:
                    kwargs["window_size"] = window_size
                if angle_range is not None:
                    kwargs["angle_range"] = angle_range
                sma_factor_value = _extract_sma_factor(buy_name)
                if (
                    sma_factor_value is not None
                    and base_name in {"ema_sma_cross", "ema_sma_cross_with_slope"}
                ):
                    kwargs["sma_window_factor"] = sma_factor_value
                if (
                    base_name == "ema_sma_cross_testing"
                    and near_range is not None
                    and above_range is not None
                ):
                    kwargs["near_range"] = near_range
                    kwargs["above_range"] = above_range
                if (
                    base_name == "ema_sma_cross_testing"
                    and use_confirmation_angle
                ):
                    kwargs["use_confirmation_angle"] = True
                if base_name == "ema_sma_cross_testing":
                    if d_sma_range is not None:
                        kwargs["d_sma_range"] = d_sma_range
                    if ema_range is not None:
                        kwargs["ema_range"] = ema_range
                    if d_ema_range is not None:
                        kwargs["d_ema_range"] = d_ema_range
                    if near_delta_range is not None:
                        kwargs["near_delta_range"] = near_delta_range
                    if price_tightness_range is not None:
                        kwargs["price_tightness_range"] = price_tightness_range
                    if sma_150_angle_min is not None:
                        kwargs["sma_150_angle_min"] = sma_150_angle_min
                    if use_ftd_confirmation:
                        kwargs["use_ftd_confirmation"] = True
                    if price_score_min is not None:
                        kwargs["price_score_min"] = price_score_min
                    if price_score_max is not None:
                        kwargs["price_score_max"] = price_score_max
                    if confirmation_sma_angle_range is not None:
                        kwargs["confirmation_sma_angle_range"] = confirmation_sma_angle_range
                    if exit_alpha_factor is not None:
                        kwargs["exit_alpha_factor"] = exit_alpha_factor
                    if shape_slope_min is not None:
                        kwargs["shape_slope_min"] = shape_slope_min
                    if shape_dev_50_max is not None:
                        kwargs["shape_dev_50_max"] = shape_dev_50_max
                    if shape_bsv_lookback is not None:
                        kwargs["shape_bsv_lookback"] = shape_bsv_lookback
            buy_function(price_data_frame, **kwargs)
            rename_signal_columns(price_data_frame, base_name, buy_name)
            entry_column_name = f"{buy_name}_entry_signal"
            if entry_column_name in price_data_frame.columns:
                buy_signal_columns.append(entry_column_name)

        sell_price_data_frame = price_data_frame.copy()
        extraneous_columns = [
            column_name
            for column_name in sell_price_data_frame.columns
            if column_name in {"ema_value", "sma_value"}
            or column_name.startswith("ema_sma_cross_")
        ]
        sell_price_data_frame.drop(
            columns=extraneous_columns, inplace=True, errors="ignore"
        )

        sell_signal_columns: list[str] = []
        for sell_name in _split_strategy_choices(sell_strategy_name):
            try:
                (
                    base_name,
                    window_size,
                    angle_range,
                    near_range,
                    above_range,
                ) = parse_strategy_name(sell_name)
            except Exception:
                continue
            if base_name not in SELL_STRATEGIES:
                continue
            sell_function = SELL_STRATEGIES[base_name]
            kwargs: dict = {}
            if base_name == "20_50_sma_cross":
                short_long = _extract_short_long_windows_for_20_50(sell_name)
                if short_long is not None:
                    kwargs["short_window_size"], kwargs["long_window_size"] = short_long
            else:
                if window_size is not None:
                    kwargs["window_size"] = window_size
                if angle_range is not None:
                    kwargs["angle_range"] = angle_range
                sma_factor_value = _extract_sma_factor(sell_name)
                if (
                    sma_factor_value is not None
                    and base_name in {"ema_sma_cross", "ema_sma_cross_with_slope"}
                ):
                    kwargs["sma_window_factor"] = sma_factor_value
                if (
                    base_name == "ema_sma_cross_testing"
                    and near_range is not None
                    and above_range is not None
                ):
                    kwargs["near_range"] = near_range
                    kwargs["above_range"] = above_range
                if base_name == "ema_sma_cross_testing" and exit_alpha_factor is not None:
                    kwargs["exit_alpha_factor"] = exit_alpha_factor
            sell_function(sell_price_data_frame, **kwargs)
            rename_signal_columns(sell_price_data_frame, base_name, sell_name)
            entry_column_name = f"{sell_name}_entry_signal"
            exit_column_name = f"{sell_name}_exit_signal"
            if entry_column_name in sell_price_data_frame.columns:
                price_data_frame[entry_column_name] = sell_price_data_frame[entry_column_name]
            if exit_column_name in sell_price_data_frame.columns:
                price_data_frame[exit_column_name] = sell_price_data_frame[exit_column_name]
                sell_signal_columns.append(exit_column_name)

        buy_signal_columns = list(dict.fromkeys(buy_signal_columns))
        sell_signal_columns = list(dict.fromkeys(sell_signal_columns))
        if buy_signal_columns:
            price_data_frame["_combined_buy_entry"] = (
                price_data_frame[buy_signal_columns].any(axis=1).fillna(False)
            )
        else:
            price_data_frame["_combined_buy_entry"] = False
        if sell_signal_columns:
            price_data_frame["_combined_sell_exit"] = (
                sell_price_data_frame[sell_signal_columns].any(axis=1).fillna(False)
            )
        else:
            price_data_frame["_combined_sell_exit"] = False

        def entry_rule(current_row: pandas.Series) -> bool:
            symbol_is_eligible = bool(symbol_mask.shift(1, fill_value=False).loc[current_row.name])
            return bool(current_row["_combined_buy_entry"]) and symbol_is_eligible

        def exit_rule(current_row: pandas.Series, entry_row: pandas.Series) -> bool:
            return bool(current_row["_combined_sell_exit"])

        def cancel_rule(current_row: pandas.Series) -> bool:
            return bool(current_row["_combined_sell_exit"])

        cooldown_after_close = 5 if any(
            base in {"ema_sma_cross", "ema_sma_cross_with_slope", "ema_shift_cross_with_slope"}
            for base in buy_bases_for_cooldown
        ) else 0
        if simulation_start_date is not None:
            run_frame = price_data_frame.loc[
                price_data_frame.index >= simulation_start_date
            ]
        else:
            run_frame = price_data_frame

        if run_frame.empty:
            continue

        simulation_result = simulate_trades(
            data=run_frame,
            entry_rule=entry_rule,
            exit_rule=exit_rule,
            entry_price_column="open",
            exit_price_column="open",
            stop_loss_percentage=stop_loss_percentage,
            take_profit_percentage=take_profit_percentage,
            trailing_stop_percentage=trailing_stop_percentage,
            cooldown_bars=cooldown_after_close,
            minimum_holding_bars=minimum_holding_bars,
            pending_limit_entry=use_confirmation_angle and confirmation_entry_mode == "limit",
            pending_market_entry=use_confirmation_angle and confirmation_entry_mode == "market",
            cancel_pending_rule=cancel_rule if use_confirmation_angle else None,
            reentry_on_signal=reentry_on_signal,
        )
        simulation_results.append(simulation_result)
        all_trades.extend(simulation_result.trades)
        symbol_name = csv_file_path.stem
        closing_price_series_by_symbol[symbol_name] = price_data_frame["close"].copy()
        symbol_volume_lookup = simple_moving_average_dollar_volume_by_symbol_and_date.get(
            symbol_name, {}
        )
        for completed_trade in simulation_result.trades:
            trade_symbol_lookup[completed_trade] = symbol_name
            percentage_change = completed_trade.profit / completed_trade.entry_price

            entry_dollar_volume = float(
                symbol_volume_lookup.get(completed_trade.entry_date, 0.0)
            )
            market_total_entry_dollar_volume = (
                market_total_dollar_volume_by_date.get(
                    completed_trade.entry_date, 0.0
                )
            )
            if market_total_entry_dollar_volume == 0:
                entry_volume_ratio = 0.0
            else:
                entry_volume_ratio = (
                    entry_dollar_volume / market_total_entry_dollar_volume
                )

            exit_dollar_volume = float(
                symbol_volume_lookup.get(completed_trade.exit_date, 0.0)
            )
            market_total_exit_dollar_volume = (
                market_total_dollar_volume_by_date.get(
                    completed_trade.exit_date, 0.0
                )
            )
            if market_total_exit_dollar_volume == 0:
                exit_volume_ratio = 0.0
            else:
                exit_volume_ratio = (
                    exit_dollar_volume / market_total_exit_dollar_volume
                )

            symbol_group_id = symbol_to_fama_french_group_id_for_details.get(
                symbol_name.upper()
            )
            group_entry_total = 0.0
            if symbol_group_id is not None:
                group_entry_total = float(
                    group_total_dollar_volume_by_group_and_date
                    .get(symbol_group_id, {})
                    .get(completed_trade.entry_date, 0.0)
                )
            if group_entry_total == 0.0:
                group_entry_total = float(market_total_entry_dollar_volume)
                group_entry_ratio = float(entry_volume_ratio)
            else:
                if group_entry_total == 0:
                    group_entry_ratio = 0.0
                else:
                    group_entry_ratio = float(entry_dollar_volume) / group_entry_total

            entry_index_position = run_frame.index.get_indexer_for(
                [completed_trade.entry_date]
            )
            # In pending market entry mode the timeline is:
            #   T   = signal fires (chip metrics should reflect this bar)
            #   T+1 = confirmation bar (angle values checked here)
            #   T+2 = market fill (entry_date)
            # Without pending entry, entry_date = T+1 and signal_date = T.
            confirmation_date = completed_trade.entry_date
            signal_date = completed_trade.entry_date
            if entry_index_position.size == 1:
                entry_index_position = int(entry_index_position[0])
                if (
                    entry_index_position >= 0
                    and entry_index_position < len(run_frame.index)
                    and entry_index_position > 0
                ):
                    confirmation_date = run_frame.index[entry_index_position - 1]
                    if entry_index_position > 1:
                        signal_date = run_frame.index[entry_index_position - 2]
                    else:
                        signal_date = confirmation_date
                else:
                    confirmation_date = completed_trade.entry_date
                    signal_date = completed_trade.entry_date
            chip_metrics = calculate_chip_concentration_metrics(
                price_data_frame.loc[: signal_date],
                lookback_window_size=60,
                include_volume_profile=False,
            )
            sma_angle_for_signal: float | None = None
            d_sma_angle_for_signal: float | None = None
            ema_angle_for_signal: float | None = None
            d_ema_angle_for_signal: float | None = None

            def _lookup_signal_value(column_name: str) -> float | None:
                """Return the column's value at the filter decision time.

                All A-layer filters (sma_angle, ema_angle, d_sma_angle,
                d_ema_angle, chip metrics) evaluate at the signal date T.
                trade_detail records the SAME values so that post-hoc
                bucket analysis matches what the live filter sees.

                Lookup order: signal_date (T) first, then confirmation_date
                (T+1), then the trade's entry_date (T+2) as a final
                fallback. The fallbacks only take effect when T is not in
                the frame's index (edge cases at the very start of data).
                """
                if column_name not in price_data_frame.columns:
                    return None
                for candidate_date in (
                    signal_date,
                    confirmation_date,
                    completed_trade.entry_date,
                ):
                    if candidate_date in price_data_frame.index:
                        val = price_data_frame.at[candidate_date, column_name]
                        if pandas.notna(val):
                            return float(val)
                return None

            def _lookup_confirmation_value(column_name: str) -> float | None:
                """Return the column's value at the B-layer confirmation
                bar (T+1). Used to record what the confirmation gate
                actually saw for this trade, independently of the A-layer
                signal-date lookup above.
                """
                if column_name not in price_data_frame.columns:
                    return None
                if confirmation_date in price_data_frame.index:
                    val = price_data_frame.at[confirmation_date, column_name]
                    if pandas.notna(val):
                        return float(val)
                return None

            # Compute ema_angle and derivatives if not already in the frame
            if "ema_value" in price_data_frame.columns and "ema_angle" not in price_data_frame.columns:
                ema_prev = price_data_frame["ema_value"].shift(1)
                ema_rel_change = (price_data_frame["ema_value"] - ema_prev) / ema_prev
                price_data_frame["ema_angle"] = numpy.degrees(numpy.arctan(ema_rel_change))
            if "sma_angle" in price_data_frame.columns and "d_sma_angle" not in price_data_frame.columns:
                price_data_frame["d_sma_angle"] = price_data_frame["sma_angle"].diff()
            if "ema_angle" in price_data_frame.columns and "d_ema_angle" not in price_data_frame.columns:
                price_data_frame["d_ema_angle"] = price_data_frame["ema_angle"].diff()
            if "close" in price_data_frame.columns and "slope_60" not in price_data_frame.columns:
                close_60_bars_ago = price_data_frame["close"].shift(59)
                price_data_frame["slope_60"] = (
                    price_data_frame["close"] - close_60_bars_ago
                ) / close_60_bars_ago

            sma_angle_for_signal = _lookup_signal_value("sma_angle")
            d_sma_angle_for_signal = _lookup_signal_value("d_sma_angle")
            ema_angle_for_signal = _lookup_signal_value("ema_angle")
            d_ema_angle_for_signal = _lookup_signal_value("d_ema_angle")
            slope_60_for_signal = _lookup_signal_value("slope_60")
            sma_angle_confirmation_value = _lookup_confirmation_value("sma_angle")
            entry_detail = TradeDetail(
                date=completed_trade.entry_date,
                symbol=symbol_name,
                action="open",
                price=completed_trade.entry_price,
                simple_moving_average_dollar_volume=entry_dollar_volume,
                total_simple_moving_average_dollar_volume=market_total_entry_dollar_volume,
                simple_moving_average_dollar_volume_ratio=entry_volume_ratio,
                group_total_simple_moving_average_dollar_volume=group_entry_total,
                group_simple_moving_average_dollar_volume_ratio=group_entry_ratio,
                price_concentration_score=chip_metrics["price_score"],
                near_price_volume_ratio=chip_metrics["near_price_volume_ratio"],
                above_price_volume_ratio=chip_metrics["above_price_volume_ratio"],
                below_price_volume_ratio=chip_metrics["below_price_volume_ratio"],
                near_delta=_lookup_signal_value("near_delta"),
                price_tightness=_lookup_signal_value("price_tightness"),
                histogram_node_count=chip_metrics["histogram_node_count"],
                sma_angle=sma_angle_for_signal,
                d_sma_angle=d_sma_angle_for_signal,
                ema_angle=ema_angle_for_signal,
                d_ema_angle=d_ema_angle_for_signal,
                slope_60=slope_60_for_signal,
                sma_angle_confirmation=sma_angle_confirmation_value,
                signal_bar_open=completed_trade.signal_bar_open,
            )
            trade_result = "win" if completed_trade.profit > 0 else "lose"
            group_exit_total = 0.0
            if symbol_group_id is not None:
                group_exit_total = float(
                    group_total_dollar_volume_by_group_and_date
                    .get(symbol_group_id, {})
                    .get(completed_trade.exit_date, 0.0)
                )
            if group_exit_total == 0.0:
                group_exit_total = float(market_total_exit_dollar_volume)
                group_exit_ratio = float(exit_volume_ratio)
            else:
                if group_exit_total == 0:
                    group_exit_ratio = 0.0
                else:
                    group_exit_ratio = float(exit_dollar_volume) / group_exit_total

            exit_detail = TradeDetail(
                date=completed_trade.exit_date,
                symbol=symbol_name,
                action="close",
                price=completed_trade.exit_price,
                simple_moving_average_dollar_volume=exit_dollar_volume,
                total_simple_moving_average_dollar_volume=market_total_exit_dollar_volume,
                simple_moving_average_dollar_volume_ratio=exit_volume_ratio,
                result=trade_result,
                percentage_change=percentage_change,
                group_total_simple_moving_average_dollar_volume=group_exit_total,
                group_simple_moving_average_dollar_volume_ratio=group_exit_ratio,
                exit_reason=completed_trade.exit_reason,
                total_commission=completed_trade.total_commission,
                share_count=completed_trade.share_count,
                max_favorable_excursion_pct=completed_trade.max_favorable_excursion_pct,
                max_adverse_excursion_pct=completed_trade.max_adverse_excursion_pct,
                max_favorable_excursion_date=completed_trade.max_favorable_excursion_date,
                max_adverse_excursion_date=completed_trade.max_adverse_excursion_date,
            )
            trade_detail_pairs[completed_trade] = (entry_detail, exit_detail)

    return StrategyEvaluationArtifacts(
        trades=all_trades,
        simulation_results=simulation_results,
        trade_symbol_lookup=trade_symbol_lookup,
        closing_price_series_by_symbol=closing_price_series_by_symbol,
        trade_detail_pairs=trade_detail_pairs,
        simulation_start_date=simulation_start_date,
    )


def _organize_trade_details_by_year(
    detail_pairs: Iterable[Tuple[TradeDetail, TradeDetail]]
) -> Dict[int, List[TradeDetail]]:
    """Group trade detail records by calendar year sorted by date."""

    trade_details_by_year: Dict[int, List[TradeDetail]] = {}
    for entry_detail, exit_detail in detail_pairs:
        trade_details_by_year.setdefault(entry_detail.date.year, []).append(entry_detail)
        trade_details_by_year.setdefault(exit_detail.date.year, []).append(exit_detail)
    for year_details in trade_details_by_year.values():
        year_details.sort(key=lambda detail: detail.date)
    return trade_details_by_year


def _assign_global_concurrent_position_counts(
    detail_pairs: Iterable[Tuple[TradeDetail, TradeDetail]]
) -> None:
    """Populate global concurrent position counts for trade detail records."""

    trade_details: List[TradeDetail] = [
        detail for entry_detail, exit_detail in detail_pairs for detail in (entry_detail, exit_detail)
    ]
    if not trade_details:
        return
    trade_details.sort(
        key=lambda detail: (
            detail.date,
            0 if detail.action == "close" else 1,
            detail.symbol,
            id(detail),
        )
    )
    open_symbols: Dict[str, bool] = {}
    current_open_count = 0
    for trade_detail in trade_details:
        symbol_name = trade_detail.symbol
        if trade_detail.action == "close":
            was_open = open_symbols.get(symbol_name, False)
            if was_open:
                trade_detail.global_concurrent_position_count = max(0, current_open_count - 1)
                open_symbols[symbol_name] = False
                current_open_count = max(0, current_open_count - 1)
            else:
                trade_detail.global_concurrent_position_count = current_open_count
        else:
            trade_detail.global_concurrent_position_count = current_open_count + 1
            if not open_symbols.get(symbol_name, False):
                open_symbols[symbol_name] = True
                current_open_count += 1


def evaluate_combined_strategy(
    data_directory: Path,
    buy_strategy_name: str,
    sell_strategy_name: str,
    minimum_average_dollar_volume: float | None = None,
    top_dollar_volume_rank: int | None = None,  # TODO: review
    maximum_symbols_per_group: int = 1,
    minimum_average_dollar_volume_ratio: float | None = None,
    starting_cash: float = 3000.0,
    withdraw_amount: float = 0.0,
    stop_loss_percentage: float = 1.0,
    take_profit_percentage: float = 0.0,
    minimum_holding_bars: int = 0,
    use_confirmation_angle: bool = False,
    confirmation_entry_mode: str = "limit",
    start_date: pandas.Timestamp | None = None,
    maximum_position_count: int = 3,
    allowed_fama_french_groups: set[int] | None = None,
    allowed_symbols: set[str] | None = None,
    exclude_other_ff12: bool = True,
    margin_multiplier: float = 1.0,
    margin_interest_annual_rate: float = 0.048,
) -> StrategyMetrics:
    """Evaluate a combination of strategies for entry and exit signals."""

    artifacts = _generate_strategy_evaluation_artifacts(
        data_directory,
        buy_strategy_name,
        sell_strategy_name,
        minimum_average_dollar_volume=minimum_average_dollar_volume,
        top_dollar_volume_rank=top_dollar_volume_rank,
        maximum_symbols_per_group=maximum_symbols_per_group,
        minimum_average_dollar_volume_ratio=minimum_average_dollar_volume_ratio,
        start_date=start_date,
        maximum_position_count=maximum_position_count,
        allowed_fama_french_groups=allowed_fama_french_groups,
        allowed_symbols=allowed_symbols,
        exclude_other_ff12=exclude_other_ff12,
        stop_loss_percentage=stop_loss_percentage,
        take_profit_percentage=take_profit_percentage,
        minimum_holding_bars=minimum_holding_bars,
        use_confirmation_angle=use_confirmation_angle,
        confirmation_entry_mode=confirmation_entry_mode,
        margin_multiplier=margin_multiplier,
        margin_interest_annual_rate=margin_interest_annual_rate,
    )

    # Filter trades by portfolio-level position cap so that metrics, trade
    # details and simulation logs only reflect trades that would actually be
    # executed under the position limit.
    accepted_trade_ids: set[int] = set()
    open_trade_ids: set[int] = set()
    entry_exit_events: List[tuple[pandas.Timestamp, int, Trade]] = []
    for completed_trade in artifacts.trades:
        entry_exit_events.append((completed_trade.entry_date, 1, completed_trade))
        entry_exit_events.append((completed_trade.exit_date, 0, completed_trade))
    entry_exit_events.sort(key=lambda e: (e[0], e[1]))
    for _event_date, event_type, completed_trade in entry_exit_events:
        trade_id = id(completed_trade)
        if event_type == 0:
            open_trade_ids.discard(trade_id)
        else:
            if trade_id in accepted_trade_ids:
                continue
            if len(open_trade_ids) >= maximum_position_count:
                continue
            accepted_trade_ids.add(trade_id)
            open_trade_ids.add(trade_id)
    filtered_trades = [
        t for t in artifacts.trades if id(t) in accepted_trade_ids
    ]
    filtered_trade_detail_pairs = {
        t: artifacts.trade_detail_pairs[t]
        for t in filtered_trades
        if t in artifacts.trade_detail_pairs
    }

    trade_profit_list: List[float] = []
    profit_percentage_list: List[float] = []
    loss_percentage_list: List[float] = []
    holding_period_list: List[int] = []
    for completed_trade in filtered_trades:
        trade_profit_list.append(completed_trade.profit)
        holding_period_list.append(completed_trade.holding_period)
        percentage_change = completed_trade.profit / completed_trade.entry_price
        if percentage_change > 0:
            profit_percentage_list.append(percentage_change)
        elif percentage_change < 0:
            loss_percentage_list.append(abs(percentage_change))

    maximum_concurrent_positions = min(
        maximum_position_count,
        calculate_maximum_concurrent_positions(artifacts.simulation_results),
    )
    trade_details_by_year = _organize_trade_details_by_year(
        filtered_trade_detail_pairs.values()
    )
    simulation_start_date = artifacts.simulation_start_date
    if simulation_start_date is None:
        simulation_start_date = pandas.Timestamp.now()
    annual_returns = calculate_annual_returns(
        filtered_trades,
        starting_cash,
        maximum_position_count,
        simulation_start_date,
        withdraw_amount,
        margin_multiplier=margin_multiplier,
        margin_interest_annual_rate=margin_interest_annual_rate,
        trade_symbol_lookup=artifacts.trade_symbol_lookup,
        closing_price_series_by_symbol=artifacts.closing_price_series_by_symbol,
        settlement_lag_days=1,
    )
    annual_trade_counts = calculate_annual_trade_counts(filtered_trades)
    final_balance = simulate_portfolio_balance(
        filtered_trades,
        starting_cash,
        maximum_position_count,
        withdraw_amount,
        margin_multiplier=margin_multiplier,
        margin_interest_annual_rate=margin_interest_annual_rate,
    )
    maximum_drawdown = calculate_max_drawdown(
        filtered_trades,
        starting_cash,
        maximum_position_count,
        artifacts.trade_symbol_lookup,
        artifacts.closing_price_series_by_symbol,
        withdraw_amount,
        margin_multiplier=margin_multiplier,
        margin_interest_annual_rate=margin_interest_annual_rate,
    )
    if filtered_trades:
        last_trade_exit_date = max(
            completed_trade.exit_date for completed_trade in filtered_trades
        )
    else:
        last_trade_exit_date = simulation_start_date
    compound_annual_growth_rate_value = 0.0
    if (
        simulation_start_date is not None
        and last_trade_exit_date is not None
        and starting_cash > 0
    ):
        duration_days = (last_trade_exit_date - simulation_start_date).days
        if duration_days > 0:
            duration_years = duration_days / 365.25
            compound_annual_growth_rate_value = (final_balance / starting_cash) ** (
                1 / duration_years
            ) - 1
    return calculate_metrics(
        trade_profit_list,
        profit_percentage_list,
        loss_percentage_list,
        holding_period_list,
        maximum_concurrent_positions,
        maximum_drawdown,
        final_balance,
        compound_annual_growth_rate_value,
        annual_returns,
        annual_trade_counts,
        trade_details_by_year,
    )



def evaluate_kalman_channel_strategy(
    data_directory: Path,
    process_variance: float = 1e-5,
    observation_variance: float = 1.0,
) -> StrategyMetrics:
    """Evaluate a Kalman channel breakout strategy across CSV files.

    Entry occurs when the closing price crosses above the upper bound of the
    Kalman filter channel. Positions are opened at the next day's opening
    price. The position is closed when the closing price crosses below the
    lower bound of the channel, using the next day's opening price.

    Parameters
    ----------
    data_directory: Path
        Directory containing CSV files with ``open`` and ``close`` columns.
    process_variance: float, default 1e-5
        Expected variance in the underlying process used by the filter.
    observation_variance: float, default 1.0
        Expected variance in the observation noise.

    Returns
    -------
    StrategyMetrics
        Metrics including total trades, win rate, profit and loss statistics,
        and holding period analysis.
    """
    trade_profit_list: List[float] = []
    profit_percentage_list: List[float] = []
    loss_percentage_list: List[float] = []
    holding_period_list: List[int] = []
    simulation_results: List[SimulationResult] = []
    for csv_path in data_directory.glob("*.csv"):
        if csv_path.stem == SP500_SYMBOL:
            continue  # Skip the S&P 500 index; it is not a tradable asset.
        price_data_frame = pandas.read_csv(
            csv_path, parse_dates=["Date"], index_col="Date"
        )
        if isinstance(price_data_frame.columns, pandas.MultiIndex):
            price_data_frame.columns = price_data_frame.columns.get_level_values(0)
        price_data_frame.columns = [
            re.sub(r"[^a-z0-9]+", "_", str(column_name).strip().lower())
            for column_name in price_data_frame.columns
        ]
        price_data_frame.columns = [
            re.sub(
                r"^_+",
                "",
                re.sub(
                    r"(?:^|_)(open|close|high|low|volume)_.*",
                    r"\1",
                    column_name,
                ),
            )
            for column_name in price_data_frame.columns
        ]
        required_columns = {"open", "close"}
        missing_column_names = [
            column
            for column in required_columns
            if column not in price_data_frame.columns
        ]
        if missing_column_names:
            missing_columns_string = ", ".join(missing_column_names)
            raise ValueError(
                f"Missing required columns: {missing_columns_string} in file {csv_path.name}"
            )

        kalman_data_frame = kalman_filter(
            price_data_frame["close"], process_variance, observation_variance
        )
        price_data_frame["kalman_estimate"] = kalman_data_frame["estimate"]
        price_data_frame["kalman_upper"] = kalman_data_frame["upper_bound"]
        price_data_frame["kalman_lower"] = kalman_data_frame["lower_bound"]
        price_data_frame["close_previous"] = price_data_frame["close"].shift(1)
        price_data_frame["upper_previous"] = price_data_frame["kalman_upper"].shift(1)
        price_data_frame["lower_previous"] = price_data_frame["kalman_lower"].shift(1)
        price_data_frame["breaks_upper"] = (
            (price_data_frame["close_previous"] <= price_data_frame["upper_previous"])
            & (price_data_frame["close"] > price_data_frame["kalman_upper"])
        )
        price_data_frame["breaks_lower"] = (
            (price_data_frame["close_previous"] >= price_data_frame["lower_previous"])
            & (price_data_frame["close"] < price_data_frame["kalman_lower"])
        )
        price_data_frame["entry_signal"] = price_data_frame["breaks_upper"].shift(
            1, fill_value=False
        )
        price_data_frame["exit_signal"] = price_data_frame["breaks_lower"].shift(
            1, fill_value=False
        )

        def entry_rule(current_row: pandas.Series) -> bool:
            """Determine whether a trade should be entered."""
            # TODO: review
            return bool(current_row["entry_signal"])

        def exit_rule(
            current_row: pandas.Series, entry_row: pandas.Series
        ) -> bool:
            """Determine whether a trade should be exited."""
            # TODO: review
            return bool(current_row["exit_signal"])

        simulation_result = simulate_trades(
            data=price_data_frame,
            entry_rule=entry_rule,
            exit_rule=exit_rule,
            entry_price_column="open",
            exit_price_column="open",
        )
        simulation_results.append(simulation_result)
        for completed_trade in simulation_result.trades:
            trade_profit_list.append(completed_trade.profit)
            holding_period_list.append(completed_trade.holding_period)
            percentage_change = (
                completed_trade.profit / completed_trade.entry_price
            )
            if percentage_change > 0:
                profit_percentage_list.append(percentage_change)
            elif percentage_change < 0:
                loss_percentage_list.append(abs(percentage_change))

    maximum_concurrent_positions = calculate_maximum_concurrent_positions(
        simulation_results
    )
    total_trades = len(trade_profit_list)
    if total_trades == 0:
        return StrategyMetrics(
            total_trades=0,
            win_rate=0.0,
            mean_profit_percentage=0.0,
            profit_percentage_standard_deviation=0.0,
            mean_loss_percentage=0.0,
            loss_percentage_standard_deviation=0.0,
            mean_holding_period=0.0,
            holding_period_standard_deviation=0.0,
            maximum_concurrent_positions=maximum_concurrent_positions,
            maximum_drawdown=0.0,
            final_balance=0.0,
            compound_annual_growth_rate=0.0,
            annual_returns={},
            annual_trade_counts={},
        )

    winning_trade_count = sum(
        1 for profit_amount in trade_profit_list if profit_amount > 0
    )
    win_rate = winning_trade_count / total_trades

    def calculate_mean(values: List[float]) -> float:
        return mean(values) if values else 0.0

    def calculate_standard_deviation(values: List[float]) -> float:
        return stdev(values) if len(values) > 1 else 0.0

    return StrategyMetrics(
        total_trades=total_trades,
        win_rate=win_rate,
        mean_profit_percentage=calculate_mean(profit_percentage_list),
        profit_percentage_standard_deviation=calculate_standard_deviation(
            profit_percentage_list
        ),
        mean_loss_percentage=calculate_mean(loss_percentage_list),
        loss_percentage_standard_deviation=calculate_standard_deviation(
            loss_percentage_list
        ),
        mean_holding_period=calculate_mean(
            [float(value) for value in holding_period_list]
        ),
        holding_period_standard_deviation=calculate_standard_deviation(
            [float(value) for value in holding_period_list]
        ),
        maximum_concurrent_positions=maximum_concurrent_positions,
        maximum_drawdown=0.0,
        final_balance=0.0,
        compound_annual_growth_rate=0.0,
        annual_returns={},
        annual_trade_counts={},
    )
