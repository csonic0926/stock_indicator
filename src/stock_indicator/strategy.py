"""Strategy evaluation utilities."""
# TODO: review

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Callable, Dict, List

import re
import pandas

from .indicators import ema, ftd, kalman_filter, rsi, sma
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

    The ``result`` field marks whether a closed trade ended in a win or a
    loss. For closing trades, ``percentage_change`` records the fractional
    price change between entry and exit.
    """
    # TODO: review
    date: pandas.Timestamp
    symbol: str
    action: str
    price: float
    simple_moving_average_dollar_volume: float
    total_simple_moving_average_dollar_volume: float
    simple_moving_average_dollar_volume_ratio: float
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


def load_price_data(csv_file_path: Path) -> pandas.DataFrame:
    """Load price data from ``csv_file_path`` and normalize column names.

    Duplicate dates are removed and the index is sorted to ensure that the
    resulting frame has unique, chronologically ordered entries.
    """
    # TODO: review

    price_data_frame = pandas.read_csv(
        csv_file_path, parse_dates=["Date"], index_col="Date"
    )
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
    require_close_above_long_term_sma: bool = True,
) -> None:
    """Attach EMA/SMA cross entry and exit signals to ``price_data_frame``.

    Entry signals require the EMA to cross above the SMA, a positive slope for
    the long-term SMA, and, when ``require_close_above_long_term_sma`` is
    ``True``, the previous day's closing price must also exceed the long-term
    SMA.

    Parameters
    ----------
    price_data_frame:
        DataFrame containing ``open`` and ``close`` price columns.
    window_size:
        Number of periods for both EMA and SMA calculations.
    require_close_above_long_term_sma:
        When ``True``, entry signals additionally require the previous day's
        closing price to be greater than the 150-day simple moving average.
    """
    # TODO: review

    price_data_frame["ema_value"] = ema(price_data_frame["close"], window_size)
    price_data_frame["sma_value"] = sma(price_data_frame["close"], window_size)
    price_data_frame["long_term_sma_value"] = sma(
        price_data_frame["close"], LONG_TERM_SMA_WINDOW
    )
    price_data_frame["ema_previous"] = price_data_frame["ema_value"].shift(1)
    price_data_frame["sma_previous"] = price_data_frame["sma_value"].shift(1)
    price_data_frame["long_term_sma_previous"] = price_data_frame[
        "long_term_sma_value"
    ].shift(1)
    price_data_frame["close_previous"] = price_data_frame["close"].shift(1)
    price_data_frame["long_term_sma_slope"] = (
        price_data_frame["long_term_sma_value"]
        - price_data_frame["long_term_sma_previous"]
    )
    price_data_frame["long_term_sma_slope_previous"] = price_data_frame[
        "long_term_sma_slope"
    ].shift(1)
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
            & (price_data_frame["long_term_sma_slope_previous"] > 0)
        )
    else:
        price_data_frame["ema_sma_cross_entry_signal"] = (
            base_entry_signal & (price_data_frame["long_term_sma_slope_previous"] > 0)
        )
    price_data_frame["ema_sma_cross_exit_signal"] = ema_cross_down.shift(
        1, fill_value=False
    )


def attach_20_50_sma_cross_signals(price_data_frame: pandas.DataFrame) -> None:
    """Attach 20/50 SMA cross entry and exit signals to ``price_data_frame``."""
    # TODO: review

    price_data_frame["sma_20_value"] = sma(price_data_frame["close"], 20)
    price_data_frame["sma_50_value"] = sma(price_data_frame["close"], 50)
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


def attach_ema_sma_cross_and_rsi_signals(
    price_data_frame: pandas.DataFrame,
    window_size: int = 40,
    rsi_window_size: int = 14,
) -> None:
    """Attach EMA/SMA cross signals filtered by RSI to ``price_data_frame``."""
    # TODO: review

    attach_ema_sma_cross_signals(price_data_frame, window_size)
    price_data_frame["rsi_value"] = rsi(
        price_data_frame["close"], rsi_window_size
    )
    price_data_frame["ema_sma_cross_and_rsi_entry_signal"] = (
        price_data_frame["ema_sma_cross_entry_signal"]
        & (price_data_frame["rsi_value"] <= 40)
    )
    price_data_frame["ema_sma_cross_and_rsi_exit_signal"] = price_data_frame[
        "ema_sma_cross_exit_signal"
    ]


def attach_ftd_ema_sma_cross_signals(
    price_data_frame: pandas.DataFrame, window_size: int = 40
) -> None:
    """Attach EMA/SMA cross signals gated by recent FTD signals."""
    # TODO: review

    attach_ema_sma_cross_signals(price_data_frame, window_size)
    ftd_signal_list: List[bool] = []
    for row_index in range(len(price_data_frame)):
        recent_price_data_frame = price_data_frame.iloc[: row_index + 1]
        ftd_signal_list.append(ftd(recent_price_data_frame, buy_mark_day=1))
    price_data_frame["ftd_signal"] = pandas.Series(
        ftd_signal_list, index=price_data_frame.index
    )
    price_data_frame["ftd_recent_signal"] = (
        price_data_frame["ftd_signal"].shift(1).rolling(window=4).max().fillna(0) >= 1
    )
    price_data_frame["ftd_ema_sma_cross_entry_signal"] = (
        price_data_frame["ema_sma_cross_entry_signal"]
        & price_data_frame["ftd_recent_signal"]
    )
    price_data_frame["ftd_ema_sma_cross_exit_signal"] = price_data_frame[
        "ema_sma_cross_exit_signal"
    ]


def attach_ema_sma_cross_with_slope_signals(
    price_data_frame: pandas.DataFrame,
    window_size: int = 40,
    slope_range: tuple[float, float] = (-0.3, 1.0),
) -> None:
    """Attach EMA/SMA cross signals filtered by simple moving average slope.

    Entry signals require the previous closing price to be above the long-term
    simple moving average and the simple moving average slope to fall within
    ``slope_range``. Unless a slope range is provided in the strategy name,
    this function uses the default range ``(-0.3, 1.0)``. The magnitude of the
    slope depends on ``window_size``; larger windows produce smaller slope
    values, so adjust ``slope_range`` accordingly when overriding the default.

    Parameters
    ----------
    price_data_frame:
        DataFrame containing ``open`` and ``close`` price columns.
    window_size:
        Number of periods for both EMA and SMA calculations.
    slope_range:
        Inclusive range ``(lower_bound, upper_bound)`` for the SMA slope.

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


def attach_ema_sma_cross_with_slope_and_volume_signals(
    price_data_frame: pandas.DataFrame,
    window_size: int = 40,
    slope_range: tuple[float, float] = (-0.3, 1.0),
) -> None:
    """Attach EMA/SMA cross signals filtered by SMA slope and dollar volume.

    These signals apply the same slope-based filter as
    :func:`attach_ema_sma_cross_with_slope_signals` and additionally require
    exponential dollar volume to exceed the simple moving average dollar
    volume. Unless the strategy name specifies otherwise, the default slope
    range ``(-0.3, 1.0)`` is used. Because the slope is computed from the
    simple moving average over ``window_size`` periods, the acceptable slope
    range should be tuned when the window size changes.

    Parameters
    ----------
    price_data_frame:
        DataFrame containing ``open``, ``close``, and ``volume`` columns.
    window_size:
        Number of periods for both EMA and SMA calculations.
    slope_range:
        Inclusive range ``(lower_bound, upper_bound)`` for the SMA slope.

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

    attach_ema_sma_cross_with_slope_signals(
        price_data_frame, window_size, slope_range=slope_range
    )
    price_data_frame["dollar_volume_value"] = (
        price_data_frame["close"] * price_data_frame["volume"]
    )
    price_data_frame["ema_dollar_volume_value"] = ema(
        price_data_frame["dollar_volume_value"], window_size
    )
    price_data_frame["sma_dollar_volume_value"] = sma(
        price_data_frame["dollar_volume_value"], window_size
    )
    price_data_frame["ema_sma_cross_with_slope_and_volume_entry_signal"] = (
        price_data_frame["ema_sma_cross_with_slope_entry_signal"]
        & (
            price_data_frame["ema_dollar_volume_value"]
            > price_data_frame["sma_dollar_volume_value"]
        )
    )
    price_data_frame["ema_sma_cross_with_slope_and_volume_exit_signal"] = (
        price_data_frame["ema_sma_cross_with_slope_exit_signal"]
    )


def attach_ema_sma_double_cross_signals(
    price_data_frame: pandas.DataFrame,
    window_size: int = 40,
) -> None:
    """Attach EMA/SMA cross signals requiring long-term EMA above SMA."""
    # TODO: review

    attach_ema_sma_cross_signals(
        price_data_frame,
        window_size,
        require_close_above_long_term_sma=False,
    )
    price_data_frame["long_term_ema_value"] = ema(
        price_data_frame["close"], LONG_TERM_SMA_WINDOW
    )
    price_data_frame["long_term_ema_previous"] = price_data_frame[
        "long_term_ema_value"
    ].shift(1)
    price_data_frame["ema_sma_double_cross_entry_signal"] = (
        price_data_frame["ema_sma_cross_entry_signal"]
        & (
            price_data_frame["long_term_ema_previous"]
            > price_data_frame["long_term_sma_previous"]
        )
    )
    price_data_frame["ema_sma_double_cross_exit_signal"] = price_data_frame[
        "ema_sma_cross_exit_signal"
    ]


def attach_kalman_filtering_signals(
    price_data_frame: pandas.DataFrame,
    process_variance: float = 1e-5,
    observation_variance: float = 1.0,
) -> None:
    """Attach Kalman filtering breakout signals to ``price_data_frame``."""
    # TODO: review

    kalman_data_frame = kalman_filter(
        price_data_frame["close"], process_variance, observation_variance
    )
    price_data_frame["kalman_estimate"] = kalman_data_frame["estimate"]
    price_data_frame["kalman_upper"] = kalman_data_frame["upper_bound"]
    price_data_frame["kalman_lower"] = kalman_data_frame["lower_bound"]
    price_data_frame["close_previous"] = price_data_frame["close"].shift(1)
    price_data_frame["upper_previous"] = price_data_frame["kalman_upper"].shift(1)
    price_data_frame["lower_previous"] = price_data_frame["kalman_lower"].shift(1)
    breaks_upper = (
        (price_data_frame["close_previous"] <= price_data_frame["upper_previous"])
        & (price_data_frame["close"] > price_data_frame["kalman_upper"])
    )
    breaks_lower = (
        (price_data_frame["close_previous"] >= price_data_frame["lower_previous"])
        & (price_data_frame["close"] < price_data_frame["kalman_lower"])
    )
    price_data_frame["kalman_filtering_entry_signal"] = breaks_upper.shift(
        1, fill_value=False
    )
    price_data_frame["kalman_filtering_exit_signal"] = breaks_lower.shift(
        1, fill_value=False
    )

# TODO: review
BUY_STRATEGIES: Dict[str, Callable[[pandas.DataFrame], None]] = {
    "ema_sma_cross": attach_ema_sma_cross_signals,
    "20_50_sma_cross": attach_20_50_sma_cross_signals,
    "ema_sma_cross_and_rsi": attach_ema_sma_cross_and_rsi_signals,
    "ftd_ema_sma_cross": attach_ftd_ema_sma_cross_signals,
    "ema_sma_cross_with_slope": attach_ema_sma_cross_with_slope_signals,
    "ema_sma_cross_with_slope_and_volume": attach_ema_sma_cross_with_slope_and_volume_signals,
    "ema_sma_double_cross": attach_ema_sma_double_cross_signals,
}

# TODO: review
SELL_STRATEGIES: Dict[str, Callable[[pandas.DataFrame], None]] = {
    **BUY_STRATEGIES,
    "kalman_filtering": attach_kalman_filtering_signals,
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
    name_parts = strategy_name.split("_")
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
    """
    # TODO: review

    buy_base_name, buy_window_size, buy_slope_range = parse_strategy_name(
        buy_strategy_name
    )
    sell_base_name, sell_window_size, sell_slope_range = parse_strategy_name(
        sell_strategy_name
    )
    if buy_base_name not in BUY_STRATEGIES:
        raise ValueError(f"Unsupported strategy: {buy_strategy_name}")
    if sell_base_name not in SELL_STRATEGIES:
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

    symbol_frames: List[tuple[Path, pandas.DataFrame]] = []
    for csv_file_path in data_directory.glob("*.csv"):
        if csv_file_path.stem == SP500_SYMBOL:
            continue
        price_data_frame = load_price_data(csv_file_path)
        if price_data_frame.empty:
            continue
        if start_date is not None:
            price_data_frame = price_data_frame.loc[
                price_data_frame.index >= start_date
            ]
            if price_data_frame.empty:
                continue
        if "volume" in price_data_frame.columns:
            dollar_volume_series = price_data_frame["close"] * price_data_frame["volume"]
            price_data_frame["simple_moving_average_dollar_volume"] = sma(
                dollar_volume_series, DOLLAR_VOLUME_SMA_WINDOW
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
        if (
            minimum_average_dollar_volume is None
            and top_dollar_volume_rank is None
            and minimum_average_dollar_volume_ratio is None
        ):
            eligibility_mask = pandas.DataFrame(
                True,
                index=merged_volume_frame.index,
                columns=merged_volume_frame.columns,
            )
        else:
            eligibility_mask = ~merged_volume_frame.isna()
            if minimum_average_dollar_volume is not None:
                eligibility_mask &= (
                    merged_volume_frame / 1_000_000 >= minimum_average_dollar_volume
                )
            if minimum_average_dollar_volume_ratio is not None:
                total_volume_series = merged_volume_frame.sum(axis=1)
                ratio_frame = merged_volume_frame.divide(
                    total_volume_series, axis=0
                )
                eligibility_mask &= (
                    ratio_frame >= minimum_average_dollar_volume_ratio
                )
            if top_dollar_volume_rank is not None:
                rank_frame = merged_volume_frame.rank(
                    axis=1, method="min", ascending=False
                )
                eligibility_mask &= rank_frame <= top_dollar_volume_rank
    else:
        merged_volume_frame = pandas.DataFrame()
        eligibility_mask = pandas.DataFrame()

    eligible_symbol_counts_by_date = (
        eligibility_mask.sum(axis=1).astype(int).to_dict()
    )
    market_total_dollar_volume_by_date = (
        merged_volume_frame.sum(axis=1).to_dict()
    )
    total_dollar_volume_by_date = (
        merged_volume_frame.where(eligibility_mask).sum(axis=1).to_dict()
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
        buy_function = BUY_STRATEGIES[buy_base_name]
        if buy_window_size is not None and buy_slope_range is not None:
            buy_function(
                price_data_frame,
                window_size=buy_window_size,
                slope_range=buy_slope_range,
            )
        elif buy_window_size is not None:
            buy_function(price_data_frame, window_size=buy_window_size)
        elif buy_slope_range is not None:
            buy_function(price_data_frame, slope_range=buy_slope_range)
        else:
            buy_function(price_data_frame)
        rename_signal_columns(price_data_frame, buy_base_name, buy_strategy_name)
        if buy_strategy_name != sell_strategy_name:
            sell_function = SELL_STRATEGIES[sell_base_name]
            if sell_window_size is not None and sell_slope_range is not None:
                sell_function(
                    price_data_frame,
                    window_size=sell_window_size,
                    slope_range=sell_slope_range,
                )
            elif sell_window_size is not None:
                sell_function(price_data_frame, window_size=sell_window_size)
            elif sell_slope_range is not None:
                sell_function(price_data_frame, slope_range=sell_slope_range)
            else:
                sell_function(price_data_frame)
            rename_signal_columns(price_data_frame, sell_base_name, sell_strategy_name)

        def entry_rule(current_row: pandas.Series) -> bool:
            symbol_is_eligible = bool(symbol_mask.loc[current_row.name])
            return bool(current_row[f"{buy_strategy_name}_entry_signal"]) and symbol_is_eligible

        def exit_rule(current_row: pandas.Series, entry_row: pandas.Series) -> bool:
            return bool(current_row[f"{sell_strategy_name}_exit_signal"])

        simulation_result = simulate_trades(
            data=price_data_frame,
            entry_rule=entry_rule,
            exit_rule=exit_rule,
            entry_price_column="open",
            exit_price_column="open",
            stop_loss_percentage=stop_loss_percentage,
        )
        simulation_results.append(simulation_result)
        all_trades.extend(simulation_result.trades)
        symbol_name = csv_file_path.stem
        symbol_volume_lookup = simple_moving_average_dollar_volume_by_symbol_and_date.get(
            symbol_name, {},
        )
        for completed_trade in simulation_result.trades:
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

            entry_detail = TradeDetail(
                date=completed_trade.entry_date,
                symbol=symbol_name,
                action="open",
                price=completed_trade.entry_price,
                simple_moving_average_dollar_volume=entry_dollar_volume,
                total_simple_moving_average_dollar_volume=market_total_entry_dollar_volume,
                simple_moving_average_dollar_volume_ratio=entry_volume_ratio,
            )
            trade_result = "win" if completed_trade.profit > 0 else "lose"  # TODO: review
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
        eligible_symbol_counts_by_date,
        simulation_start_date,
        withdraw_amount,
    )
    annual_trade_counts = calculate_annual_trade_counts(all_trades)
    final_balance = simulate_portfolio_balance(
        all_trades, starting_cash, eligible_symbol_counts_by_date, withdraw_amount
    )
    maximum_drawdown = calculate_max_drawdown(
        all_trades, starting_cash, eligible_symbol_counts_by_date, withdraw_amount
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
    exponential moving average crosses above the simple moving average and the
    previous day's closing price is higher than the 150-day simple moving
    average. Positions are opened at the next day's opening price. The position
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

        price_data_frame["ema_value"] = ema(price_data_frame["close"], window_size)
        price_data_frame["sma_value"] = sma(price_data_frame["close"], window_size)
        price_data_frame["long_term_sma_value"] = sma(
            price_data_frame["close"], LONG_TERM_SMA_WINDOW
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
