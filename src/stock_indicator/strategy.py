"""Strategy evaluation utilities."""
# TODO: review

from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
import math
from math import ceil
from pathlib import Path
from statistics import mean, stdev
from typing import Callable, Dict, Iterable, List, Tuple
import re

import numpy
import pandas

from .indicators import ema, kalman_filter, sma
from .chip_filter import calculate_chip_concentration_metrics
from .simulator import (
    SimulationResult,
    Trade,
    calculate_annual_returns,
    calculate_annual_trade_counts,
    calculate_annual_profit_totals,
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

def load_symbols_excluded_by_industry() -> set[str]:
    """Return symbols that should be excluded based on industry classification.

    When the sector classification dataset is available (``data/symbols_with_sector``),
    exclude any symbols whose Fama–French 12-industry group (``ff12``) equals 12
    ("Other"). If the dataset is not present, return an empty set.
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
    # Normalize expected columns and filter ff12==12
    sector_frame.columns = [str(c).strip().lower() for c in sector_frame.columns]
    if "ticker" not in sector_frame.columns or "ff12" not in sector_frame.columns:
        excluded_symbols = set()
    else:
        mask_other = sector_frame["ff12"] == 12
        tickers_series = sector_frame.loc[mask_other, "ticker"].dropna().astype(str)
        excluded_symbols = set(tickers_series.str.upper().tolist())

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
                excluded_symbols.update(override_symbols)
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
    symbol_to_group: dict[str, int] = {
        str(row.ticker).upper(): int(row.ff12)
        for row in sector_frame.itertuples(index=False)
    }
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
    ``"signal"``, ``"stop_loss"``, or ``"end_of_data"``.
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
    histogram_node_count: int | None = None
    sma_angle: float | None = None
    # Number of concurrent open positions at this event.
    # For an "open" event, includes this position. For a "close" event,
    # excludes the position being closed.
    concurrent_position_count: int = 0
    strategy_set_label: str | None = None
    result: str | None = None  # TODO: review
    percentage_change: float | None = None  # TODO: review
    exit_reason: str = "signal"


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
    annual_profit_totals: Dict[int, float] = field(default_factory=dict)
    trade_details_by_year: Dict[int, List[TradeDetail]] = field(default_factory=dict)


@dataclass
class ComplexStrategySetDefinition:
    """Configuration for a strategy set used in complex simulations."""

    label: str
    buy_strategy_name: str
    sell_strategy_name: str
    stop_loss_percentage: float = 1.0
    minimum_average_dollar_volume: float | None = None
    minimum_average_dollar_volume_ratio: float | None = None
    top_dollar_volume_rank: int | None = None
    maximum_symbols_per_group: int = 1


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
) -> ComplexSimulationMetrics:
    """Evaluate multiple strategy sets under a shared configuration."""

    if maximum_position_count <= 0:
        raise ValueError("maximum_position_count must be positive")
    if not set_definitions:
        raise ValueError("set_definitions must not be empty")

    effective_interest_rate = (
        margin_interest_annual_rate if margin_multiplier != 1.0 else 0.0
    )
    artifacts_by_set: Dict[str, StrategyEvaluationArtifacts] = {}
    position_limits_by_set: Dict[str, int] = {}
    for label, definition in set_definitions.items():
        maximum_positions_for_set = maximum_position_count
        if label.upper() == "B":
            maximum_positions_for_set = max(
                1, math.ceil(maximum_position_count / 2)
            )
        position_limits_by_set[label] = maximum_positions_for_set
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
            allowed_symbols=None,
            exclude_other_ff12=True,
            stop_loss_percentage=definition.stop_loss_percentage,
            margin_multiplier=margin_multiplier,
            margin_interest_annual_rate=effective_interest_rate,
        )

    accepted_trades_by_set: Dict[str, List[Trade]] = {
        label: [] for label in set_definitions
    }
    open_position_counts_by_set: Dict[str, int] = {label: 0 for label in set_definitions}
    open_trade_keys: Dict[Tuple[str, int], str] = {}
    accepted_trade_keys: set[Tuple[str, int]] = set()
    events: List[tuple[pandas.Timestamp, int, str, Trade]] = []
    for label, artifacts in artifacts_by_set.items():
        for trade in artifacts.trades:
            events.append((trade.entry_date, 1, label, trade))
            events.append((trade.exit_date, 0, label, trade))
    events.sort(key=lambda event: (event[0], event[1]))

    for event_date, event_type, label, trade in events:
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
                normalized_label == "B"
                and current_open_total >= position_limits_by_set[label]
            ):
                continue
            if open_position_counts_by_set[label] >= position_limits_by_set[label]:
                continue
            accepted_trade_keys.add(trade_key)
            open_trade_keys[trade_key] = label
            open_position_counts_by_set[label] += 1
            accepted_trades_by_set[label].append(trade)

    metrics_by_set: Dict[str, StrategyMetrics] = {}
    aggregated_trades: List[Trade] = []
    aggregated_trade_profit_list: List[float] = []
    aggregated_profit_percentage_list: List[float] = []
    aggregated_loss_percentage_list: List[float] = []
    aggregated_holding_period_list: List[int] = []
    aggregated_detail_pairs_with_label: List[
        Tuple[TradeDetail, TradeDetail]
    ] = []
    aggregated_trade_symbol_lookup: Dict[Trade, str] = {}
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
            filtered_trade_symbol_lookup[trade] = artifacts.trade_symbol_lookup.get(
                trade, ""
            )
            entry_detail, exit_detail = artifacts.trade_detail_pairs[trade]
            detail_pairs_with_label.append(
                (
                    replace(entry_detail, strategy_set_label=label),
                    replace(exit_detail, strategy_set_label=label),
                )
            )

        trade_details_by_year = _organize_trade_details_by_year(detail_pairs_with_label)
        filtered_simulation_results: List[SimulationResult] = []
        if trades_for_set:
            filtered_simulation_results.append(
                SimulationResult(
                    trades=trades_for_set,
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
                {},
                trade_details_by_year,
            )
            continue

        simulation_start_date = artifacts.simulation_start_date
        if simulation_start_date is None:
            simulation_start_date = pandas.Timestamp.now()
        annual_returns = calculate_annual_returns(
            trades_for_set,
            starting_cash,
            position_limits_by_set[label],
            simulation_start_date,
            withdraw_amount,
            margin_multiplier=margin_multiplier,
            margin_interest_annual_rate=effective_interest_rate,
            trade_symbol_lookup=filtered_trade_symbol_lookup,
            closing_price_series_by_symbol=artifacts.closing_price_series_by_symbol,
            settlement_lag_days=1,
        )
        annual_trade_counts = calculate_annual_trade_counts(trades_for_set)
        annual_profit_totals = calculate_annual_profit_totals(trades_for_set)
        final_balance = simulate_portfolio_balance(
            trades_for_set,
            starting_cash,
            position_limits_by_set[label],
            withdraw_amount,
            margin_multiplier=margin_multiplier,
            margin_interest_annual_rate=effective_interest_rate,
        )
        maximum_drawdown = calculate_max_drawdown(
            trades_for_set,
            starting_cash,
            position_limits_by_set[label],
            filtered_trade_symbol_lookup,
            artifacts.closing_price_series_by_symbol,
            withdraw_amount,
            margin_multiplier=margin_multiplier,
            margin_interest_annual_rate=effective_interest_rate,
        )
        if trades_for_set:
            last_trade_exit_date = max(
                trade.exit_date for trade in trades_for_set
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
            annual_profit_totals,
            trade_details_by_year,
        )

        aggregated_trades.extend(trades_for_set)
        aggregated_trade_profit_list.extend(trade_profit_list)
        aggregated_profit_percentage_list.extend(profit_percentage_list)
        aggregated_loss_percentage_list.extend(loss_percentage_list)
        aggregated_holding_period_list.extend(holding_period_list)
        aggregated_detail_pairs_with_label.extend(detail_pairs_with_label)
        aggregated_trade_symbol_lookup.update(filtered_trade_symbol_lookup)
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
            aggregated_trades,
            starting_cash,
            maximum_position_count,
            aggregated_simulation_start_date,
            withdraw_amount,
            margin_multiplier=margin_multiplier,
            margin_interest_annual_rate=effective_interest_rate,
            trade_symbol_lookup=aggregated_trade_symbol_lookup,
            closing_price_series_by_symbol=(
                aggregated_closing_price_series_by_symbol
            ),
            settlement_lag_days=1,
        )
        aggregated_annual_trade_counts = calculate_annual_trade_counts(
            aggregated_trades
        )
        aggregated_annual_profit_totals = calculate_annual_profit_totals(
            aggregated_trades
        )
        aggregated_final_balance = simulate_portfolio_balance(
            aggregated_trades,
            starting_cash,
            maximum_position_count,
            withdraw_amount,
            margin_multiplier=margin_multiplier,
            margin_interest_annual_rate=effective_interest_rate,
        )
        aggregated_maximum_drawdown = calculate_max_drawdown(
            aggregated_trades,
            starting_cash,
            maximum_position_count,
            aggregated_trade_symbol_lookup,
            aggregated_closing_price_series_by_symbol,
            withdraw_amount,
            margin_multiplier=margin_multiplier,
            margin_interest_annual_rate=effective_interest_rate,
        )
        last_exit_date = max(trade.exit_date for trade in aggregated_trades)
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
        aggregated_annual_profit_totals = {}
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
        aggregated_annual_profit_totals,
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
        (1–11). Group 12 ("Other") is always excluded when sector data exists.
    allowed_symbols:
        Optional whitelist of symbols (CSV stems) to consider.
    exclude_other_ff12:
        When True, symbols in FF12 group 12 ("Other") are excluded.
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
        buy_signal_columns: list[str] = []
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
            if use_unshifted_signals:
                column_name = f"{buy_name}_raw_entry_signal"
                if column_name in price_data_frame.columns:
                    buy_signal_columns.append(column_name)
                elif f"{buy_name}_entry_signal" in price_data_frame.columns:
                    buy_signal_columns.append(f"{buy_name}_entry_signal")
            else:
                column_name = f"{buy_name}_entry_signal"
                if column_name in price_data_frame.columns:
                    buy_signal_columns.append(column_name)

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
    price_data_frame["ema_sma_cross_with_slope_entry_signal"] = (
        price_data_frame["ema_sma_cross_entry_signal"]
        & (price_data_frame["sma_angle"] >= angle_lower_bound)
        & (price_data_frame["sma_angle"] <= angle_upper_bound)
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
    for row_index in range(len(price_data_frame)):
        chip_metrics = calculate_chip_concentration_metrics(
            price_data_frame.iloc[: row_index + 1],
            lookback_window_size=60,
            include_volume_profile=False,
        )
        near_ratios.append(chip_metrics["near_price_volume_ratio"])
        above_ratios.append(chip_metrics["above_price_volume_ratio"])
    price_data_frame["near_price_volume_ratio"] = pandas.Series(
        near_ratios, index=price_data_frame.index
    )
    price_data_frame["above_price_volume_ratio"] = pandas.Series(
        above_ratios, index=price_data_frame.index
    )

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

    price_data_frame["ema_sma_cross_testing_entry_signal"] = (
        price_data_frame["ema_sma_cross_entry_signal"]
        & (price_data_frame["sma_angle"] >= angle_lower_bound)
        & (price_data_frame["sma_angle"] <= angle_upper_bound)
        & (
            near_price_ratio_previous_ok.fillna(False)
            & above_price_ratio_previous_ok.fillna(False)
        )
    )
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

    price_data_frame["ema_shift_cross_with_slope_entry_signal"] = (
        base_entry
        & (price_data_frame["shifted_ema_angle"] >= angle_lower_bound)
        & (price_data_frame["shifted_ema_angle"] <= angle_upper_bound)
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
    annual_profit_totals: Dict[int, float] | None = None,
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
            annual_profit_totals=
                {} if annual_profit_totals is None else annual_profit_totals,
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
        annual_profit_totals={}
        if annual_profit_totals is None
        else annual_profit_totals,
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
    margin_multiplier: float = 1.0,
    margin_interest_annual_rate: float = 0.048,
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
            symbol_is_eligible = bool(symbol_mask.loc[current_row.name])
            return bool(current_row["_combined_buy_entry"]) and symbol_is_eligible

        def exit_rule(current_row: pandas.Series, entry_row: pandas.Series) -> bool:
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
            cooldown_bars=cooldown_after_close,
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
            signal_date = completed_trade.entry_date
            if entry_index_position.size == 1:
                entry_index_position = int(entry_index_position[0])
                if (
                    entry_index_position >= 0
                    and entry_index_position < len(run_frame.index)
                    and entry_index_position > 0
                ):
                    signal_date = price_data_frame.index[entry_index_position - 1]
                else:
                    signal_date = completed_trade.entry_date
            chip_metrics = calculate_chip_concentration_metrics(
                price_data_frame.loc[: signal_date],
                lookback_window_size=60,
                include_volume_profile=False,
            )
            sma_angle_for_signal: float | None = None
            if "sma_angle" in price_data_frame.columns:
                for angle_date in (signal_date, completed_trade.entry_date):
                    if angle_date in price_data_frame.index:
                        angle_value = price_data_frame.at[angle_date, "sma_angle"]
                        if pandas.notna(angle_value):
                            sma_angle_for_signal = float(angle_value)
                            break
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
                histogram_node_count=chip_metrics["histogram_node_count"],
                sma_angle=sma_angle_for_signal,
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
        margin_multiplier=margin_multiplier,
        margin_interest_annual_rate=margin_interest_annual_rate,
    )

    trade_profit_list: List[float] = []
    profit_percentage_list: List[float] = []
    loss_percentage_list: List[float] = []
    holding_period_list: List[int] = []
    for completed_trade in artifacts.trades:
        trade_profit_list.append(completed_trade.profit)
        holding_period_list.append(completed_trade.holding_period)
        percentage_change = completed_trade.profit / completed_trade.entry_price
        if percentage_change > 0:
            profit_percentage_list.append(percentage_change)
        elif percentage_change < 0:
            loss_percentage_list.append(abs(percentage_change))

    maximum_concurrent_positions = calculate_maximum_concurrent_positions(
        artifacts.simulation_results
    )
    trade_details_by_year = _organize_trade_details_by_year(
        artifacts.trade_detail_pairs.values()
    )
    simulation_start_date = artifacts.simulation_start_date
    if simulation_start_date is None:
        simulation_start_date = pandas.Timestamp.now()
    annual_returns = calculate_annual_returns(
        artifacts.trades,
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
    annual_trade_counts = calculate_annual_trade_counts(artifacts.trades)
    annual_profit_totals = calculate_annual_profit_totals(artifacts.trades)
    final_balance = simulate_portfolio_balance(
        artifacts.trades,
        starting_cash,
        maximum_position_count,
        withdraw_amount,
        margin_multiplier=margin_multiplier,
        margin_interest_annual_rate=margin_interest_annual_rate,
    )
    maximum_drawdown = calculate_max_drawdown(
        artifacts.trades,
        starting_cash,
        maximum_position_count,
        artifacts.trade_symbol_lookup,
        artifacts.closing_price_series_by_symbol,
        withdraw_amount,
        margin_multiplier=margin_multiplier,
        margin_interest_annual_rate=margin_interest_annual_rate,
    )
    if artifacts.trades:
        last_trade_exit_date = max(
            completed_trade.exit_date for completed_trade in artifacts.trades
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
        annual_profit_totals,
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
