"""Strategy evaluation utilities."""
# TODO: review

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from statistics import mean, stdev
from typing import Callable, Dict, List

import re
import pandas

from .indicators import ema, kalman_filter, sma
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


def _split_strategy_choices(strategy_name: str) -> list[str]:
    """Split a strategy expression by common OR separators.

    Supports "or", "|", "/", and comma as separators. Trims whitespace and
    returns the list of non-empty tokens. If no separator is present, returns
    a single-item list containing the original name.
    """
    parts = re.split(r"\s*(?:\bor\b|\||/|,)\s*", strategy_name.strip())
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
                seen_groups: set[int] = set()
                for symbol_name in sorted_symbols:
                    group_identifier = symbol_to_group_lookup.get(symbol_name)
                    if group_identifier is None:
                        continue
                    if group_identifier in seen_groups:
                        continue
                    chosen_symbols.append(symbol_name)
                    seen_groups.add(group_identifier)
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
    approximates the number of significant volume clusters. The ``result``
    field marks whether a closed trade ended in a win or a loss. For closing
    trades, ``percentage_change`` records the fractional price change between
    entry and exit.
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
    # Number of concurrent open positions at this event.
    # For an "open" event, includes this position. For a "close" event,
    # excludes the position being closed.
    concurrent_position_count: int = 0
    result: str | None = None  # TODO: review
    percentage_change: float | None = None  # TODO: review


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
        Global Top-N ranking (with at most one symbol per FF12 group when
        sector data is available).
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

    Returns
    -------
    Dict[str, List[str]]
        Mapping with keys ``"entry_signals"`` and ``"exit_signals"`` containing
        lists of symbols that triggered the respective signals on the sampled
        row.
    """
    # TODO: review

    # Validate strategies first (supports composite expressions)
    buy_choice_names = _split_strategy_choices(buy_strategy_name)
    sell_choice_names = _split_strategy_choices(sell_strategy_name)

    def _has_supported(tokens: list[str], table: dict) -> bool:
        for token in tokens:
            try:
                base_name, _, _ = parse_strategy_name(token)
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

    def _apply_strategy(
        full_name: str,
        base_name: str,
        window_size: int | None,
        slope_range: tuple[float, float] | None,
        table: Dict[str, Callable[[pandas.DataFrame], None]],
        frame: pandas.DataFrame,
    ) -> None:
        kwargs: dict = {}
        if base_name == "20_50_sma_cross":
            maybe_windows = _extract_short_long_windows_for_20_50(full_name)
            if maybe_windows is not None:
                kwargs["short_window_size"], kwargs["long_window_size"] = maybe_windows
        else:
            if window_size is not None:
                kwargs["window_size"] = window_size
            if slope_range is not None:
                kwargs["slope_range"] = slope_range
            sma_factor_value = _extract_sma_factor(full_name)
            if sma_factor_value is not None and base_name in {"ema_sma_cross", "ema_sma_cross_with_slope"}:
                kwargs["sma_window_factor"] = sma_factor_value
        table[base_name](frame, **kwargs)
        if base_name != full_name:
            frame.rename(
                columns={
                    f"{base_name}_entry_signal": f"{full_name}_entry_signal",
                    f"{base_name}_exit_signal": f"{full_name}_exit_signal",
                },
                inplace=True,
            )

    # Build signals and sample the most recent bar at or before evaluation_date
    for csv_file_path, price_data_frame, symbol_mask in selected_symbol_data:
        # Build buy-side signals (support composite expressions)
        buy_signal_columns: list[str] = []
        for buy_name in buy_choice_names:
            try:
                base_name, window_size, slope_range = parse_strategy_name(buy_name)
            except Exception:  # noqa: BLE001
                continue
            if base_name not in BUY_STRATEGIES:
                continue
            _apply_strategy(
                buy_name, base_name, window_size, slope_range, BUY_STRATEGIES, price_data_frame
            )
            column_name = f"{buy_name}_entry_signal"
            if column_name in price_data_frame.columns:
                buy_signal_columns.append(column_name)

        sell_signal_columns: list[str] = []
        for sell_name in sell_choice_names:
            try:
                base_name, window_size, slope_range = parse_strategy_name(sell_name)
            except Exception:  # noqa: BLE001
                continue
            if base_name not in SELL_STRATEGIES:
                continue
            _apply_strategy(
                sell_name, base_name, window_size, slope_range, SELL_STRATEGIES, price_data_frame
            )
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

    return {"entry_signals": entry_signal_symbols, "exit_signals": exit_signal_symbols}


def load_price_data(csv_file_path: Path) -> pandas.DataFrame:
    """Load price data from ``csv_file_path`` and normalize column names.

    Duplicate dates are removed and the index is sorted to ensure that the
    resulting frame has unique, chronologically ordered entries. When the CSV
    file is empty, an empty data frame is returned so the caller can skip the
    symbol gracefully.
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
    else:
        price_data_frame["ema_sma_cross_entry_signal"] = base_entry_signal
    price_data_frame["ema_sma_cross_exit_signal"] = ema_cross_down.shift(
        1, fill_value=False
    )


def attach_20_50_sma_cross_signals(
    price_data_frame: pandas.DataFrame,
    short_window_size: int = 20,
    long_window_size: int = 50,
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


## Removed deprecated strategies: ema_sma_cross_and_rsi, ftd_ema_sma_cross


def attach_ema_sma_cross_with_slope_signals(
    price_data_frame: pandas.DataFrame,
    window_size: int = 40,
    slope_range: tuple[float, float] = (-0.3, 2.14),
    sma_window_factor: float | None = None,
) -> None:
    """Attach EMA/SMA cross signals filtered by simple moving average slope.

    Entry signals require the prior-day EMA cross, the simple moving average
    slope to fall within ``slope_range``, and the closing price to remain above
    the long-term simple moving average. Unless a slope range is provided in the
    strategy name, this function uses the default range ``(-0.3, 2.14)``. The
    magnitude of the slope depends on ``window_size``; larger windows produce
    smaller slope values, so adjust ``slope_range`` accordingly when overriding
    the default.

    Parameters
    ----------
    price_data_frame:
        DataFrame containing ``open`` and ``close`` price columns.
    window_size:
        Number of periods for EMA calculations.
    slope_range:
        Inclusive range ``(lower_bound, upper_bound)`` for the SMA slope.
    sma_window_factor:
        Optional multiplier applied to ``window_size`` to determine the SMA
        window as ``ceil(window_size * factor)``. When ``None``, uses
        ``window_size`` for SMA as well.

    Raises
    ------
    ValueError
        If ``slope_range`` has a lower bound greater than its upper bound.
    """
    # TODO: review

    slope_lower_bound, slope_upper_bound = slope_range
    if slope_lower_bound > slope_upper_bound:
        raise ValueError(
            "Invalid slope_range: lower bound cannot exceed upper bound"
        )

    attach_ema_sma_cross_signals(
        price_data_frame,
        window_size,
        require_close_above_long_term_sma=True,
        sma_window_factor=sma_window_factor,
    )
    price_data_frame["sma_slope"] = (
        price_data_frame["sma_value"] - price_data_frame["sma_previous"]
    )
    price_data_frame["ema_sma_cross_with_slope_entry_signal"] = (
        price_data_frame["ema_sma_cross_entry_signal"]
        & (price_data_frame["sma_slope"] >= slope_lower_bound)
        & (price_data_frame["sma_slope"] <= slope_upper_bound)
    )
    price_data_frame["ema_sma_cross_with_slope_exit_signal"] = price_data_frame[
        "ema_sma_cross_exit_signal"
    ]


def attach_ema_sma_cross_testing_signals(
    price_data_frame: pandas.DataFrame,
    window_size: int = 40,
    slope_range: tuple[float, float] = (-0.3, 2.14),
    near_pct: float = 0.12,
    above_pct: float = 0.10,
    sma_window_factor: float | None = None,
) -> None:
    """Attach EMA/SMA cross testing signals with slope and chip filters.

    Entry signals mirror :func:`attach_ema_sma_cross_with_slope_signals` but do
    not require the closing price to remain above the long-term simple moving
    average. Instead, this variant recomputes chip concentration metrics and
    validates that the near-price and above-price volume ratios fall below the
    ``near_pct`` and ``above_pct`` thresholds.

    Parameters
    ----------
    price_data_frame:
        DataFrame containing ``open``, ``high``, ``low``, ``close`` and
        ``volume`` columns.
    window_size:
        Number of periods for EMA calculations.
    slope_range:
        Inclusive range ``(lower_bound, upper_bound)`` for the SMA slope.
    near_pct:
        Maximum allowed fraction of volume near the current price.
    above_pct:
        Maximum allowed fraction of volume above the current price.
    sma_window_factor:
        Optional multiplier applied to ``window_size`` to determine the SMA
        window as ``ceil(window_size * factor)``. When ``None``, uses
        ``window_size`` for the SMA as well.

    Raises
    ------
    ValueError
        If ``slope_range`` has a lower bound greater than its upper bound.
    """
    # TODO: review

    slope_lower_bound, slope_upper_bound = slope_range
    if slope_lower_bound > slope_upper_bound:
        raise ValueError(
            "Invalid slope_range: lower bound cannot exceed upper bound"
        )

    attach_ema_sma_cross_signals(
        price_data_frame,
        window_size,
        require_close_above_long_term_sma=False,
        sma_window_factor=sma_window_factor,
    )
    price_data_frame["sma_slope"] = (
        price_data_frame["sma_value"] - price_data_frame["sma_previous"]
    )

    near_ratios: List[float | None] = []
    above_ratios: List[float | None] = []
    for row_index in range(len(price_data_frame)):
        chip_metrics = calculate_chip_concentration_metrics(
            price_data_frame.iloc[: row_index + 1],
            lookback_window_size=60,
        )
        near_ratios.append(chip_metrics["near_price_volume_ratio"])
        above_ratios.append(chip_metrics["above_price_volume_ratio"])
    price_data_frame["near_price_volume_ratio"] = pandas.Series(
        near_ratios, index=price_data_frame.index
    )
    price_data_frame["above_price_volume_ratio"] = pandas.Series(
        above_ratios, index=price_data_frame.index
    )

    near_ok = price_data_frame["near_price_volume_ratio"].le(near_pct)
    above_ok = price_data_frame["above_price_volume_ratio"].le(above_pct)
    price_data_frame["ema_sma_cross_testing_entry_signal"] = (
        price_data_frame["ema_sma_cross_entry_signal"]
        & (price_data_frame["sma_slope"] >= slope_lower_bound)
        & (price_data_frame["sma_slope"] <= slope_upper_bound)
        & near_ok.fillna(False)
        & above_ok.fillna(False)
    )
    price_data_frame["ema_sma_cross_testing_exit_signal"] = price_data_frame[
        "ema_sma_cross_exit_signal"
    ]


def attach_ema_shift_cross_with_slope_signals(
    price_data_frame: pandas.DataFrame,
    window_size: int = 35,
    slope_range: tuple[float, float] = (-1.0, 1.0),
) -> None:
    """Attach EMA/EMA(shifted) cross signals filtered by EMA slope.

    This strategy mirrors :func:`attach_ema_sma_cross_with_slope_signals` but
    replaces the simple moving average with an exponential moving average
    computed from the closing prices shifted back by three trading days.

    Entry conditions:
    - Previous day's EMA crosses above the shifted EMA
    - The shifted EMA slope falls within ``slope_range``

    Exit condition:
    - Previous day's EMA crosses below the shifted EMA

    Parameters
    ----------
    price_data_frame:
        DataFrame containing at least ``open`` and ``close`` columns.
    window_size:
        Number of periods for both EMAs. Defaults to ``35`` when not specified
        in the strategy name.
    slope_range:
        Inclusive range ``(lower_bound, upper_bound)`` for the shifted EMA
        slope (difference to previous value). Defaults to ``(-1.0, 1.0)``.

    Raises
    ------
    ValueError
        If ``slope_range`` has a lower bound greater than its upper bound.
    """
    slope_lower_bound, slope_upper_bound = slope_range
    if slope_lower_bound > slope_upper_bound:
        raise ValueError(
            "Invalid slope_range: lower bound cannot exceed upper bound"
        )

    # Core moving averages
    _close_r3 = price_data_frame["close"].round(3)
    price_data_frame["ema_value"] = ema(_close_r3, window_size)
    price_data_frame["shifted_close"] = price_data_frame["close"].round(3).shift(3)
    price_data_frame["shifted_ema_value"] = ema(
        price_data_frame["shifted_close"], window_size
    )

    # Previous-day values for stable cross detection
    price_data_frame["ema_previous"] = price_data_frame["ema_value"].shift(1)
    price_data_frame["shifted_ema_previous"] = price_data_frame[
        "shifted_ema_value"
    ].shift(1)

    # Cross detection
    crosses_up = (
        (price_data_frame["ema_previous"] <= price_data_frame["shifted_ema_previous"])
        & (price_data_frame["ema_value"] > price_data_frame["shifted_ema_value"])
    )
    crosses_down = (
        (price_data_frame["ema_previous"] >= price_data_frame["shifted_ema_previous"])
        & (price_data_frame["ema_value"] < price_data_frame["shifted_ema_value"])
    )

    # Slope of shifted EMA
    price_data_frame["shifted_ema_slope"] = (
        price_data_frame["shifted_ema_value"]
        - price_data_frame["shifted_ema_previous"]
    )

    # Build signals using prior-day confirmation
    base_entry = crosses_up.shift(1, fill_value=False)
    base_exit = crosses_down.shift(1, fill_value=False)

    price_data_frame["ema_shift_cross_with_slope_entry_signal"] = (
        base_entry
        & (price_data_frame["shifted_ema_slope"] >= slope_lower_bound)
        & (price_data_frame["shifted_ema_slope"] <= slope_upper_bound)
    )
    price_data_frame["ema_shift_cross_with_slope_exit_signal"] = base_exit

## Removed deprecated strategy: ema_sma_cross_with_slope_and_volume


## Removed deprecated strategy: ema_sma_double_cross


## Removed deprecated strategy: kalman_filtering

# TODO: review
BUY_STRATEGIES: Dict[str, Callable[[pandas.DataFrame], None]] = {
    "ema_sma_cross": attach_ema_sma_cross_signals,
    "20_50_sma_cross": attach_20_50_sma_cross_signals,
    "ema_sma_cross_with_slope": attach_ema_sma_cross_with_slope_signals,
    "ema_sma_cross_testing": attach_ema_sma_cross_testing_signals,
    "ema_shift_cross_with_slope": attach_ema_shift_cross_with_slope_signals,
}

# TODO: review
SELL_STRATEGIES: Dict[str, Callable[[pandas.DataFrame], None]] = {
    **BUY_STRATEGIES,
}

# TODO: review
SUPPORTED_STRATEGIES: Dict[str, Callable[[pandas.DataFrame], None]] = {
    **SELL_STRATEGIES,
}


def parse_strategy_name(
    strategy_name: str,
) -> tuple[str, int | None, tuple[float, float] | None]:
    """Split ``strategy_name`` into base name, window size, and slope range.

    Strategy identifiers may include a numeric window size suffix or a pair of
    floating-point numbers representing a slope range. These components are
    separated from the base name by underscores. For example,
    ``"ema_sma_cross_with_slope_40_-0.5_0.5"`` is interpreted as the base
    strategy ``"ema_sma_cross_with_slope"`` with a window size of ``40`` and a
    slope range from ``-0.5`` to ``0.5``.

    Any optional trailing ``_sma{factor}`` suffix (e.g., ``_sma1.2``) is ignored
    for base-name/window/slope parsing and should be obtained via
    :func:`_extract_sma_factor` if needed.

    Parameters
    ----------
    strategy_name:
        The full strategy name possibly containing a numeric suffix and an
        optional slope range.

    Returns
    -------
    tuple[str, int | None, tuple[float, float] | None]
        The base strategy name, the integer window size, and the slope range as
        a ``(lower, upper)`` tuple. When a component is not present, its value
        is ``None``.

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
    while name_parts:
        segment = name_parts[-1]
        try:
            float(segment)
        except ValueError:
            break
        numeric_segments.append(segment)
        name_parts.pop()
    numeric_segments.reverse()

    base_name = "_".join(name_parts)
    segment_count = len(numeric_segments)
    if segment_count == 0:
        return base_name, None, None

    if segment_count == 1:
        numeric_value = numeric_segments[0]
        if numeric_value.isdigit():
            window_size = int(numeric_value)
            if window_size <= 0:
                raise ValueError(
                    "Window size must be a positive integer in strategy name: "
                    f"{strategy_name}"
                )
            return base_name, window_size, None
        raise ValueError(
            "Malformed strategy name: expected two numeric segments for slope range "
            f"but found {segment_count} in '{strategy_name}'"
        )

    if segment_count == 2:
        lower_bound, upper_bound = (
            float(numeric_segments[0]),
            float(numeric_segments[1]),
        )
        return base_name, None, (lower_bound, upper_bound)

    if segment_count == 3:
        window_value = numeric_segments[0]
        if not window_value.isdigit():
            raise ValueError(
                "Malformed strategy name: expected two numeric segments for slope range "
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
        return base_name, window_size, (lower_bound, upper_bound)

    raise ValueError(
        "Malformed strategy name: expected up to three numeric segments but "
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


def evaluate_combined_strategy(
    data_directory: Path,
    buy_strategy_name: str,
    sell_strategy_name: str,
    minimum_average_dollar_volume: float | None = None,
    top_dollar_volume_rank: int | None = None,  # TODO: review
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
    """Evaluate a combination of strategies for entry and exit signals.

    The function evaluates strategies on full historical data and uses symbol
    eligibility to gate entries. Exit signals remain active even when a symbol
    becomes ineligible so existing positions can close.

    Parameters
    ----------
    data_directory: Path
        Directory containing price data in CSV format.
    buy_strategy_name: str
        Strategy name used to generate entry signals. The name may include a
        numeric window size or a pair of slope bounds, for example
        ``"ema_sma_cross_with_slope_40_-0.5_0.5"``.
    sell_strategy_name: str
        Strategy name used to generate exit signals. The same conventions as
        ``buy_strategy_name`` apply.
    minimum_average_dollar_volume: float | None, optional
        Minimum 50-day moving average dollar volume, in millions, required for a
        symbol to be included in the evaluation. When ``None``, no filter is
        applied.
    top_dollar_volume_rank: int | None, optional
        Retain only the ``N`` symbols with the highest 50-day simple moving
        average dollar volume on each trading day. When ``None``, no ranking
        filter is applied.
    minimum_average_dollar_volume_ratio: float | None, optional
        Minimum fraction of the total market 50-day average dollar volume that
        a symbol must exceed to be eligible. Specify values as decimals, for
        example ``0.01`` for ``1%``. When ``None``, no ratio filter is applied.
    starting_cash: float, default 3000.0
        Initial amount of cash used for portfolio simulation.
    withdraw_amount: float, default 0.0
        Cash amount removed from the balance at the end of each calendar year.
    stop_loss_percentage: float, default 1.0
        Fractional loss from the entry price that triggers an exit on the next
        bar's opening price. Values greater than or equal to ``1.0`` disable
        the stop-loss mechanism.
    start_date: pandas.Timestamp | None, optional
        First day of the simulation. When provided, price data is limited to
        rows on or after this date before any signals are calculated. The
        simulation begins on the later of ``start_date`` and the earliest date
        on which a symbol becomes eligible.
    maximum_position_count: int, default 3
        Upper bound on the number of simultaneous open positions. Each
        position uses a fixed fraction of equity based on this limit.
    allowed_fama_french_groups: set[int] | None, optional
        Restrict the tradable universe to the specified FF12 group identifiers
        (1–11). Symbols in group 12 ("Other") are always excluded. When this
        parameter is provided and a symbol lacks a known FF12 mapping, the
        symbol is excluded to avoid unintended trades.
    allowed_symbols: set[str] | None, optional
        When provided, restrict evaluation to the specified set of symbols
        (case-sensitive match to CSV stem). Useful for single-symbol
        simulations.
    exclude_other_ff12: bool, default True
        When True, symbols tagged with FF12 group 12 ("Other") are excluded.
        Set to False to allow trading of symbols in the "Other" group. This is
        ignored when sector data is unavailable.
    """
    # TODO: review

    # Support composite strategy expressions like "A or B"; require that at
    # least one option is a supported base strategy on each side.
    buy_choice_names = _split_strategy_choices(buy_strategy_name)
    sell_choice_names = _split_strategy_choices(sell_strategy_name)

    def _has_supported(tokens: list[str], table: dict) -> bool:
        for token in tokens:
            try:
                base, _, _ = parse_strategy_name(token)
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

    trade_profit_list: List[float] = []
    profit_percentage_list: List[float] = []
    loss_percentage_list: List[float] = []
    holding_period_list: List[int] = []
    simulation_results: List[SimulationResult] = []
    all_trades: List[Trade] = []
    simulation_start_date: pandas.Timestamp | None = None
    trade_details_by_year: Dict[int, List[TradeDetail]] = {}  # TODO: review
    trade_symbol_lookup: Dict[Trade, str] = {}
    closing_price_series_by_symbol: Dict[str, pandas.Series] = {}

    symbol_frames: List[tuple[Path, pandas.DataFrame]] = []
    symbols_excluded_by_industry = (
        load_symbols_excluded_by_industry() if exclude_other_ff12 else set()
    )
    symbol_to_group_map_for_filtering: dict[str, int] | None = None
    if allowed_fama_french_groups is not None:
        # Build a mapping from symbol to FF12 group for filtering.
        symbol_to_group_map_for_filtering = load_ff12_groups_by_symbol()
    for csv_file_path in data_directory.glob("*.csv"):
        if csv_file_path.stem == SP500_SYMBOL:
            continue
        if allowed_symbols is not None and csv_file_path.stem not in allowed_symbols:
            continue
        # Skip symbols classified as FF12==12 ("Other") when sector data exists.
        if csv_file_path.stem.upper() in symbols_excluded_by_industry:
            continue
        # If an allowed group list is provided, enforce it here.
        if symbol_to_group_map_for_filtering is not None:
            group_identifier = symbol_to_group_map_for_filtering.get(
                csv_file_path.stem.upper()
            )
            if group_identifier is None or group_identifier not in allowed_fama_french_groups:
                continue
        price_data_frame = load_price_data(csv_file_path)
        if price_data_frame.empty:
            continue
        # Compute dollar-volume SMA using full look-back before any slicing so
        # the initial bars after start_date have valid values.
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
        # Important: do NOT slice by start_date here. We keep full
        # price history so long-window indicators (e.g., 150-day SMA)
        # computed later have proper lookback across year boundaries.
        # We will slice to the actual simulation_start_date right before
        # running the trade simulation.
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
        )
    else:
        merged_volume_frame = pandas.DataFrame()
        eligibility_mask = pandas.DataFrame()

    market_total_dollar_volume_by_date = (
        merged_volume_frame.sum(axis=1).to_dict()
    )
    total_dollar_volume_by_date = (
        merged_volume_frame.where(eligibility_mask).sum(axis=1).to_dict()
    )

    # Build per-group total dollar volume by date to support group-aware ratios in
    # trade details. When sector data is not available, this remains empty and
    # callers should fall back to market totals.
    symbol_to_fama_french_group_id_for_details = load_ff12_groups_by_symbol()
    group_total_dollar_volume_by_group_and_date: dict[int, dict[pandas.Timestamp, float]] = {}
    if not merged_volume_frame.empty and symbol_to_fama_french_group_id_for_details:
        group_id_to_symbol_columns_for_details: dict[int, list[str]] = {}
        for column_name in merged_volume_frame.columns:
            group_id = symbol_to_fama_french_group_id_for_details.get(
                column_name.upper()
            )
            if group_id is None:
                continue
            group_id_to_symbol_columns_for_details.setdefault(group_id, []).append(
                column_name
            )
        for group_id, column_list in group_id_to_symbol_columns_for_details.items():
            group_frame = merged_volume_frame[column_list]
            group_total_series = group_frame.sum(axis=1)
            group_total_dollar_volume_by_group_and_date[group_id] = (
                group_total_series.to_dict()
            )

    selected_symbol_data: List[tuple[Path, pandas.DataFrame, pandas.Series]] = []
    simple_moving_average_dollar_volume_by_symbol_and_date: Dict[
        str, Dict[pandas.Timestamp, float]
    ] = {}
    first_eligible_dates: List[pandas.Timestamp] = []
    for csv_file_path, price_data_frame in symbol_frames:
        symbol_name = csv_file_path.stem
        if symbol_name not in eligibility_mask.columns:
            continue
        symbol_mask = eligibility_mask[symbol_name]
        symbol_mask = symbol_mask.reindex(price_data_frame.index, fill_value=False)
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

    def rename_signal_columns(
        price_data_frame: pandas.DataFrame,
        original_name: str,
        new_name: str,
    ) -> None:
        if original_name == new_name:
            return
        price_data_frame.rename(
            columns={
                f"{original_name}_entry_signal": f"{new_name}_entry_signal",
                f"{original_name}_exit_signal": f"{new_name}_exit_signal",
            },
            inplace=True,
        )

    for csv_file_path, price_data_frame, symbol_mask in selected_symbol_data:
        # Build signals for all buy-side choices
        buy_signal_columns: list[str] = []
        buy_bases_for_cooldown: set[str] = set()
        for buy_name in _split_strategy_choices(buy_strategy_name):
            try:
                base_name, window_size, slope_range = parse_strategy_name(buy_name)
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
                if slope_range is not None:
                    kwargs["slope_range"] = slope_range
                # Optional SMA window factor support for EMA/SMA strategies
                sma_factor_value = _extract_sma_factor(buy_name)
                if sma_factor_value is not None and base_name in {"ema_sma_cross", "ema_sma_cross_with_slope"}:
                    kwargs["sma_window_factor"] = sma_factor_value
            buy_function(price_data_frame, **kwargs)
            rename_signal_columns(price_data_frame, base_name, buy_name)
            col = f"{buy_name}_entry_signal"
            if col in price_data_frame.columns:
                buy_signal_columns.append(col)

        # Build signals for all sell-side choices
        sell_signal_columns: list[str] = []
        for sell_name in _split_strategy_choices(sell_strategy_name):
            try:
                base_name, window_size, slope_range = parse_strategy_name(sell_name)
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
                if slope_range is not None:
                    kwargs["slope_range"] = slope_range
                sma_factor_value = _extract_sma_factor(sell_name)
                if sma_factor_value is not None and base_name in {"ema_sma_cross", "ema_sma_cross_with_slope"}:
                    kwargs["sma_window_factor"] = sma_factor_value
            sell_function(price_data_frame, **kwargs)
            rename_signal_columns(price_data_frame, base_name, sell_name)
            col = f"{sell_name}_exit_signal"
            if col in price_data_frame.columns:
                sell_signal_columns.append(col)

        # Deduplicate column lists and build combined signals to avoid
        # ambiguity when duplicate labels are present in the row index.
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

        def entry_rule(current_row: pandas.Series) -> bool:
            symbol_is_eligible = bool(symbol_mask.loc[current_row.name])
            return bool(current_row["_combined_buy_entry"]) and symbol_is_eligible

        def exit_rule(current_row: pandas.Series, entry_row: pandas.Series) -> bool:
            return bool(current_row["_combined_sell_exit"]) 

        cooldown_after_close = 5 if any(
            base in {"ema_sma_cross", "ema_sma_cross_with_slope", "ema_shift_cross_with_slope"}
            for base in buy_bases_for_cooldown
        ) else 0
        # Slice to simulation_start_date only at simulation time so
        # previously computed indicators (including long-term SMAs)
        # retain lookback from pre-start history.
        if simulation_start_date is not None:
            run_frame = price_data_frame.loc[
                price_data_frame.index >= simulation_start_date
            ]
        else:
            run_frame = price_data_frame

        if run_frame.empty:
            # Nothing to simulate for this symbol in the requested window.
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
            symbol_name, {},
        )
        for completed_trade in simulation_result.trades:
            trade_symbol_lookup[completed_trade] = symbol_name
            trade_profit_list.append(completed_trade.profit)
            holding_period_list.append(completed_trade.holding_period)
            percentage_change = completed_trade.profit / completed_trade.entry_price
            if percentage_change > 0:
                profit_percentage_list.append(percentage_change)
            elif percentage_change < 0:
                loss_percentage_list.append(abs(percentage_change))

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

            # Compute group-aware totals and ratios for trade details.
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
                # Fallback to market total to avoid zero division and preserve interpretability
                group_entry_total = float(market_total_entry_dollar_volume)
                group_entry_ratio = float(entry_volume_ratio)
            else:
                group_entry_ratio = float(entry_dollar_volume) / group_entry_total
            chip_metrics = calculate_chip_concentration_metrics(
                price_data_frame.loc[: completed_trade.entry_date],
                lookback_window_size=60,
            )  # TODO: review
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
            )
            trade_result = "win" if completed_trade.profit > 0 else "lose"  # TODO: review
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
            )
            trade_details_by_year.setdefault(
                completed_trade.entry_date.year, []
            ).append(entry_detail)
            trade_details_by_year.setdefault(
                completed_trade.exit_date.year, []
            ).append(exit_detail)

    maximum_concurrent_positions = calculate_maximum_concurrent_positions(
        simulation_results
    )
    if simulation_start_date is None:
        simulation_start_date = pandas.Timestamp.now()
    annual_returns = calculate_annual_returns(
        all_trades,
        starting_cash,
        maximum_position_count,
        simulation_start_date,
        withdraw_amount,
        margin_multiplier=margin_multiplier,
        margin_interest_annual_rate=margin_interest_annual_rate,
        trade_symbol_lookup=trade_symbol_lookup,
        closing_price_series_by_symbol=closing_price_series_by_symbol,
        settlement_lag_days=1,
    )
    annual_trade_counts = calculate_annual_trade_counts(all_trades)
    final_balance = simulate_portfolio_balance(
        all_trades,
        starting_cash,
        maximum_position_count,
        withdraw_amount,
        margin_multiplier=margin_multiplier,
        margin_interest_annual_rate=margin_interest_annual_rate,
    )
    maximum_drawdown = calculate_max_drawdown(
        all_trades,
        starting_cash,
        maximum_position_count,
        trade_symbol_lookup,
        closing_price_series_by_symbol,
        withdraw_amount,
        margin_multiplier=margin_multiplier,
        margin_interest_annual_rate=margin_interest_annual_rate,
    )
    if all_trades:
        last_trade_exit_date = max(
            completed_trade.exit_date for completed_trade in all_trades
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
    for year_trades in trade_details_by_year.values():
        year_trades.sort(key=lambda detail: detail.date)
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


def evaluate_ema_sma_cross_strategy(
    data_directory: Path,
    window_size: int = 15,
) -> StrategyMetrics:
    """Evaluate EMA and SMA cross strategy across all CSV files in a directory.

    The function calculates the win rate of applying an EMA and SMA cross
    strategy to each CSV file in ``data_directory``. Entry occurs when the
    exponential moving average crosses above the simple moving average. Positions
    are opened at the next day's opening price. The position
    is closed when the exponential moving average crosses below the simple
    moving average, using the next day's opening price.

    Parameters
    ----------
    data_directory: Path
        Directory containing CSV files with columns ``open`` and ``close``.
    window_size: int, default 15
        Number of periods to use for both EMA and SMA calculations.

    Returns
    -------
    StrategyMetrics
        Metrics including total trades, win rate, profit and loss statistics, and
        holding period analysis.
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
        # Normalize column names to handle multi-level headers and varied casing
        # so required columns can be detected consistently
        price_data_frame.columns = [
            re.sub(r"[^a-z0-9]+", "_", str(column_name).strip().lower())
            for column_name in price_data_frame.columns
        ]
        # Remove trailing ticker identifiers such as "_riv" and any leading
        # underscores so column names are reduced to identifiers like "open"
        # and "close"
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
            required_column
            for required_column in required_columns
            if required_column not in price_data_frame.columns
        ]
        if missing_column_names:
            missing_columns_string = ", ".join(missing_column_names)
            raise ValueError(
                f"Missing required columns: {missing_columns_string} in file {csv_path.name}"
            )

        _close_r3 = price_data_frame["close"].round(3)
        price_data_frame["ema_value"] = ema(_close_r3, window_size)
        price_data_frame["sma_value"] = sma(_close_r3, window_size)
        price_data_frame["long_term_sma_value"] = sma(
            _close_r3, LONG_TERM_SMA_WINDOW
        )
        price_data_frame["ema_previous"] = price_data_frame["ema_value"].shift(1)
        price_data_frame["sma_previous"] = price_data_frame["sma_value"].shift(1)
        price_data_frame["long_term_sma_previous"] = price_data_frame[
            "long_term_sma_value"
        ].shift(1)
        price_data_frame["close_previous"] = price_data_frame["close"].shift(1)
        price_data_frame["cross_up"] = (
            (price_data_frame["ema_previous"] <= price_data_frame["sma_previous"])
            & (price_data_frame["ema_value"] > price_data_frame["sma_value"])
        )
        price_data_frame["cross_down"] = (
            (price_data_frame["ema_previous"] >= price_data_frame["sma_previous"])
            & (price_data_frame["ema_value"] < price_data_frame["sma_value"])
        )
        price_data_frame["entry_signal"] = price_data_frame["cross_up"].shift(
            1, fill_value=False
        )
        price_data_frame["exit_signal"] = price_data_frame["cross_down"].shift(
            1, fill_value=False
        )

        def entry_rule(current_row: pandas.Series) -> bool:
            """Determine whether a trade should be entered."""
            # TODO: review
            return bool(current_row["entry_signal"]) and (
                current_row["close_previous"]
                > current_row["long_term_sma_previous"]
            )

        def exit_rule(
            current_row: pandas.Series, entry_row: pandas.Series
        ) -> bool:
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
        mean_holding_period=calculate_mean([float(value) for value in holding_period_list]),
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


# TODO: review
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
