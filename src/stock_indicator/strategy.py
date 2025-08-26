"""Strategy evaluation utilities."""
# TODO: review

from __future__ import annotations

from dataclasses import dataclass
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
    calculate_annual_returns_with_withdrawal,
    calculate_annual_trade_counts,
    calculate_maximum_concurrent_positions,
    simulate_and_withdraw,
    simulate_portfolio_balance,
    simulate_trades,
)


LONG_TERM_SMA_WINDOW: int = 150


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
    final_balance: float
    annual_returns: Dict[int, float]
    annual_trade_counts: Dict[int, int]


def load_price_data(csv_file_path: Path) -> pandas.DataFrame:
    """Load price data from ``csv_file_path`` and normalize column names."""
    # TODO: review

    price_data_frame = pandas.read_csv(
        csv_file_path, parse_dates=["Date"], index_col="Date"
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
    window_size: int = 50,
    require_close_above_long_term_sma: bool = True,
) -> None:
    """Attach EMA/SMA cross entry and exit signals to ``price_data_frame``.

    Parameters
    ----------
    price_data_frame:
        DataFrame containing ``open`` and ``close`` price columns.
    window_size:
        Number of periods for both EMA and SMA calculations.
    require_close_above_long_term_sma:
        When ``True``, entry signals are only generated if the previous day's
        closing price is greater than the 150-day simple moving average.
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
    window_size: int = 50,
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
    price_data_frame: pandas.DataFrame, window_size: int = 50
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
    window_size: int = 50,
    slope_range: tuple[float, float] = (-0.3, 1.0),
) -> None:
    """Attach EMA/SMA cross signals filtered by SMA slope to ``price_data_frame``.

    Entry signals are generated only when the previous closing price is greater
    than the long-term simple moving average and the slope of the simple moving
    average lies within ``slope_range``.
    """
    # TODO: review

    attach_ema_sma_cross_signals(
        price_data_frame,
        window_size,
        require_close_above_long_term_sma=True,
    )
    price_data_frame["sma_slope"] = (
        price_data_frame["sma_value"] - price_data_frame["sma_previous"]
    )
    slope_lower_bound, slope_upper_bound = slope_range
    price_data_frame["ema_sma_cross_with_slope_entry_signal"] = (
        price_data_frame["ema_sma_cross_entry_signal"]
        & (price_data_frame["sma_slope"] >= slope_lower_bound)
        & (price_data_frame["sma_slope"] <= slope_upper_bound)
    )
    price_data_frame["ema_sma_cross_with_slope_exit_signal"] = price_data_frame[
        "ema_sma_cross_exit_signal"
    ]


def attach_ema_sma_double_cross_signals(
    price_data_frame: pandas.DataFrame,
    window_size: int = 50,
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


def calculate_metrics(
    trade_profit_list: List[float],
    profit_percentage_list: List[float],
    loss_percentage_list: List[float],
    holding_period_list: List[int],
    maximum_concurrent_positions: int = 0,
    final_balance: float = 0.0,
    annual_returns: Dict[int, float] | None = None,
    annual_trade_counts: Dict[int, int] | None = None,
) -> StrategyMetrics:
    """Compute summary metrics for a list of simulated trades."""
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
            final_balance=final_balance,
            annual_returns={} if annual_returns is None else annual_returns,
            annual_trade_counts={} if annual_trade_counts is None else annual_trade_counts,
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
        final_balance=final_balance,
        annual_returns={} if annual_returns is None else annual_returns,
        annual_trade_counts={} if annual_trade_counts is None else annual_trade_counts,
    )


def evaluate_combined_strategy(
    data_directory: Path,
    buy_strategy_name: str,
    sell_strategy_name: str,
    minimum_average_dollar_volume: float | None = None,
    top_dollar_volume_rank: int | None = None,  # TODO: review
    starting_cash: float = 3000.0,
    stop_loss_percentage: float = 1.0,
    annual_withdrawal: float = 0.0,
) -> StrategyMetrics:
    """Evaluate a combination of strategies for entry and exit signals.

    Parameters
    ----------
    data_directory: Path
        Directory containing price data in CSV format.
    buy_strategy_name: str
        Strategy name used to generate entry signals.
    sell_strategy_name: str
        Strategy name used to generate exit signals.
    minimum_average_dollar_volume: float | None, optional
        Minimum 50-day moving average dollar volume, in millions, required for a
        symbol to be included in the evaluation. When ``None``, no filter is
        applied.
    top_dollar_volume_rank: int | None, optional
        Select only the ``N`` symbols with the highest 50-day average dollar
        volume. When ``None``, no ranking filter is applied.
    starting_cash: float, default 3000.0
        Initial amount of cash used for portfolio simulation.
    stop_loss_percentage: float, default 1.0
        Fractional loss from the entry price that triggers an exit on the next
        bar's opening price. Values greater than or equal to ``1.0`` disable
        the stop-loss mechanism.
    annual_withdrawal: float, default 0.0
        Amount of cash removed from the portfolio at the end of each calendar
        year during balance simulation. A value of ``0.0`` disables
        withdrawals.
    """
    # TODO: review

    if buy_strategy_name not in BUY_STRATEGIES:
        raise ValueError(f"Unsupported strategy: {buy_strategy_name}")
    if sell_strategy_name not in SELL_STRATEGIES:
        raise ValueError(f"Unsupported strategy: {sell_strategy_name}")

    trade_profit_list: List[float] = []
    profit_percentage_list: List[float] = []
    loss_percentage_list: List[float] = []
    holding_period_list: List[int] = []
    simulation_results: List[SimulationResult] = []
    all_trades: List[Trade] = []
    simulation_start_date: pandas.Timestamp | None = None

    symbol_frames: List[tuple[Path, pandas.DataFrame]] = []  # TODO: review
    latest_dollar_volumes: List[tuple[Path, float]] = []  # TODO: review
    for csv_file_path in data_directory.glob("*.csv"):
        price_data_frame = load_price_data(csv_file_path)
        if price_data_frame.empty:
            continue
        symbol_frames.append((csv_file_path, price_data_frame))
        file_start_date = price_data_frame.index.min()
        if simulation_start_date is None or file_start_date < simulation_start_date:
            simulation_start_date = file_start_date
        if "volume" in price_data_frame.columns:
            dollar_volume_series = price_data_frame["close"] * price_data_frame["volume"]
            if dollar_volume_series.empty:
                recent_average_dollar_volume = 0.0
            else:
                recent_average_dollar_volume = float(
                    dollar_volume_series.rolling(window=50).mean().iloc[-1]
                )
        else:
            recent_average_dollar_volume = 0.0
        latest_dollar_volumes.append((csv_file_path, recent_average_dollar_volume))

    filtered_latest_dollar_volumes = [
        (csv_path, dollar_volume)
        for csv_path, dollar_volume in latest_dollar_volumes
        if (
            minimum_average_dollar_volume is None
            or (dollar_volume / 1_000_000) >= minimum_average_dollar_volume
        )
    ]

    if top_dollar_volume_rank is not None:
        filtered_latest_dollar_volumes.sort(
            key=lambda volume_item: volume_item[1], reverse=True
        )
        selected_paths = {
            csv_path
            for csv_path, _ in filtered_latest_dollar_volumes[:top_dollar_volume_rank]
        }
    else:
        selected_paths = {csv_path for csv_path, _ in filtered_latest_dollar_volumes}

    eligible_symbol_count = len(selected_paths)

    for csv_file_path, price_data_frame in symbol_frames:
        if csv_file_path not in selected_paths:
            continue
        if minimum_average_dollar_volume is not None:
            if "volume" not in price_data_frame.columns:
                raise ValueError(
                    "Volume column is required to compute dollar volume filter"
                )
            dollar_volume_series = price_data_frame["close"] * price_data_frame["volume"]
            if dollar_volume_series.empty:
                continue
            recent_average_dollar_volume = (
                dollar_volume_series.rolling(window=50).mean().iloc[-1] / 1_000_000
            )
            if pandas.isna(recent_average_dollar_volume) or (
                recent_average_dollar_volume < minimum_average_dollar_volume
            ):
                continue
        BUY_STRATEGIES[buy_strategy_name](price_data_frame)
        if buy_strategy_name != sell_strategy_name:
            SELL_STRATEGIES[sell_strategy_name](price_data_frame)

        def entry_rule(current_row: pandas.Series) -> bool:
            return bool(
                current_row[f"{buy_strategy_name}_entry_signal"]
            )

        def exit_rule(
            current_row: pandas.Series, entry_row: pandas.Series
        ) -> bool:
            return bool(
                current_row[f"{sell_strategy_name}_exit_signal"]
            )

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
    if simulation_start_date is None:
        simulation_start_date = pandas.Timestamp.now()
    if annual_withdrawal > 0:
        annual_returns = calculate_annual_returns_with_withdrawal(
            all_trades,
            starting_cash,
            eligible_symbol_count,
            simulation_start_date,
            annual_withdrawal,
        )
    else:
        annual_returns = calculate_annual_returns(
            all_trades, starting_cash, eligible_symbol_count, simulation_start_date
        )
    annual_trade_counts = calculate_annual_trade_counts(all_trades)
    if annual_withdrawal > 0:
        final_balance = simulate_and_withdraw(
            all_trades,
            starting_cash,
            eligible_symbol_count,
            annual_withdrawal,
        )
    else:
        final_balance = simulate_portfolio_balance(
            all_trades, starting_cash, eligible_symbol_count
        )
    return calculate_metrics(
        trade_profit_list,
        profit_percentage_list,
        loss_percentage_list,
        holding_period_list,
        maximum_concurrent_positions,
        final_balance,
        annual_returns,
        annual_trade_counts,
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
            final_balance=0.0,
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
        final_balance=0.0,
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
            final_balance=0.0,
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
        final_balance=0.0,
        annual_returns={},
        annual_trade_counts={},
    )
