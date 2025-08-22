"""Strategy evaluation utilities."""
# TODO: review

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import re
import pandas

from .indicators import ema, sma
from .simulator import simulate_trades


def evaluate_ema_sma_cross_strategy(
    data_directory: Path,
    window_size: int = 15,
) -> Tuple[int, float]:
    """Evaluate EMA and SMA cross strategy across all CSV files in a directory.

    The function calculates the win rate of applying an EMA and SMA cross
    strategy to each CSV file in ``data_directory``. Entry occurs when the
    exponential moving average crosses above the simple moving average and the
    position is opened at the next day's opening price. The position is closed
    when the exponential moving average crosses below the simple moving average,
    using the next day's closing price.

    Parameters
    ----------
    data_directory: Path
        Directory containing CSV files with columns ``open`` and ``close``.
    window_size: int, default 15
        Number of periods to use for both EMA and SMA calculations.

    Returns
    -------
    tuple[int, float]
        Total number of trades and the win rate as a float between 0 and 1.
    """
    trade_profit_list: List[float] = []
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
        # Remove trailing ticker identifiers such as "_riv" so that column names
        # are reduced to plain identifiers like "open" and "close"
        price_data_frame.columns = [
            re.sub(r"(open|close|high|low|volume)_[a-z0-9]+$", r"\1", column_name)
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
        price_data_frame["ema_previous"] = price_data_frame["ema_value"].shift(1)
        price_data_frame["sma_previous"] = price_data_frame["sma_value"].shift(1)
        price_data_frame["cross_up"] = (
            (price_data_frame["ema_previous"] <= price_data_frame["sma_previous"])
            & (price_data_frame["ema_value"] > price_data_frame["sma_value"])
        )
        price_data_frame["cross_down"] = (
            (price_data_frame["ema_previous"] >= price_data_frame["sma_previous"])
            & (price_data_frame["ema_value"] < price_data_frame["sma_value"])
        )
        price_data_frame["entry_signal"] = price_data_frame["cross_up"].shift(1, fill_value=False)
        price_data_frame["exit_signal"] = price_data_frame["cross_down"].shift(1, fill_value=False)

        def entry_rule(current_row: pandas.Series) -> bool:
            return bool(current_row["entry_signal"])

        def exit_rule(
            current_row: pandas.Series, entry_row: pandas.Series
        ) -> bool:
            return bool(current_row["exit_signal"])

        simulation_result = simulate_trades(
            data=price_data_frame,
            entry_rule=entry_rule,
            exit_rule=exit_rule,
            entry_price_column="open",
            exit_price_column="close",
        )
        for completed_trade in simulation_result.trades:
            trade_profit_list.append(completed_trade.profit)

    total_trades = len(trade_profit_list)
    if total_trades == 0:
        return 0, 0.0
    winning_trade_count = sum(
        1 for profit_amount in trade_profit_list if profit_amount > 0
    )
    win_rate = winning_trade_count / total_trades
    return total_trades, win_rate
