"""Simulation utilities for executing trading strategies."""
# TODO: review

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import pandas

# TODO: review
TRADE_COMMISSION: float = 1.0

@dataclass(frozen=True)
class Trade:
    """Record details for a completed trade."""

    entry_date: pandas.Timestamp
    exit_date: pandas.Timestamp
    entry_price: float
    exit_price: float
    profit: float
    holding_period: int


@dataclass
class SimulationResult:
    """Aggregate outcome of a trade simulation."""

    trades: List[Trade]
    total_profit: float


def simulate_trades(
    data: pandas.DataFrame,
    entry_rule: Callable[[pandas.Series], bool],
    exit_rule: Callable[[pandas.Series, pandas.Series], bool],
    entry_price_column: str = "close",
    exit_price_column: str | None = None,
    stop_loss_percentage: float = 1.0,
) -> SimulationResult:
    """Simulate trades using supplied entry and exit rules.

    Parameters
    ----------
    data: pandas.DataFrame
        Data frame containing the price data.
    entry_rule: Callable[[pandas.Series], bool]
        Function invoked for each row to determine trade entry.
    exit_rule: Callable[[pandas.Series, pandas.Series], bool]
        Function invoked with the current row and the entry row to determine
        when to close the trade.
    entry_price_column: str, default "close"
        Column name used for calculating entry price.
    exit_price_column: str, optional
        Column name used for calculating exit price. When ``None``,
        ``entry_price_column`` is used for both entry and exit prices.
    stop_loss_percentage: float, default 1.0
        Fractional loss from the entry price that triggers an exit on the next
        bar's opening price. Values greater than or equal to ``1.0`` disable
        the stop-loss mechanism.
    A fixed commission defined by ``TRADE_COMMISSION`` is subtracted from each
    trade's profit.

    Returns
    -------
    SimulationResult
        Contains the list of completed trades and aggregate profit.
    """
    trades: List[Trade] = []
    in_position = False
    entry_row: pandas.Series | None = None
    entry_row_index: int | None = None
    stop_loss_pending = False
    for row_index in range(len(data)):
        current_row = data.iloc[row_index]
        is_last_row = row_index == len(data) - 1
        if not in_position:
            if is_last_row:
                continue
            if entry_rule(current_row):
                in_position = True
                entry_row = current_row
                entry_row_index = row_index
        else:
            if entry_row is None or entry_row_index is None:
                continue
            if stop_loss_pending:
                entry_price = float(entry_row[entry_price_column])
                price_column_name = (
                    exit_price_column if exit_price_column is not None else entry_price_column
                )
                exit_price = float(current_row[price_column_name])
                profit_value = exit_price - entry_price - TRADE_COMMISSION
                holding_period_value = row_index - entry_row_index
                trades.append(
                    Trade(
                        entry_date=data.index[entry_row_index],
                        exit_date=data.index[row_index],
                        entry_price=entry_price,
                        exit_price=exit_price,
                        profit=profit_value,
                        holding_period=holding_period_value,
                    )
                )
                in_position = False
                entry_row = None
                entry_row_index = None
                stop_loss_pending = False
                continue
            if exit_rule(current_row, entry_row):
                entry_price = float(entry_row[entry_price_column])
                price_column_name = (
                    exit_price_column if exit_price_column is not None else entry_price_column
                )
                exit_price = float(current_row[price_column_name])
                profit_value = exit_price - entry_price - TRADE_COMMISSION
                holding_period_value = row_index - entry_row_index
                trades.append(
                    Trade(
                        entry_date=data.index[entry_row_index],
                        exit_date=data.index[row_index],
                        entry_price=entry_price,
                        exit_price=exit_price,
                        profit=profit_value,
                        holding_period=holding_period_value,
                    )
                )
                in_position = False
                entry_row = None
                entry_row_index = None
                continue
            if 0 < stop_loss_percentage < 1 and not is_last_row:
                entry_price = float(entry_row[entry_price_column])
                if float(current_row["close"]) <= entry_price * (1 - stop_loss_percentage):
                    stop_loss_pending = True
    if in_position and entry_row is not None and entry_row_index is not None:
        # TODO: review
        price_column_name = (
            exit_price_column if exit_price_column is not None else entry_price_column
        )
        final_row = data.iloc[-1]
        entry_price = float(entry_row[entry_price_column])
        exit_price = float(final_row[price_column_name])
        profit_value = exit_price - entry_price - TRADE_COMMISSION
        holding_period_value = len(data) - entry_row_index - 1
        trades.append(
            Trade(
                entry_date=data.index[entry_row_index],
                exit_date=data.index[-1],
                entry_price=entry_price,
                exit_price=exit_price,
                profit=profit_value,
                holding_period=holding_period_value,
            )
        )
    total_profit = sum(completed_trade.profit for completed_trade in trades)
    return SimulationResult(trades=trades, total_profit=total_profit)


# TODO: review
def calculate_maximum_concurrent_positions(
    simulation_results: Iterable[SimulationResult],
) -> int:
    """Determine the highest number of simultaneous open positions.

    Parameters
    ----------
    simulation_results : Iterable[SimulationResult]
        Collection of simulation outputs containing trades to evaluate.

    Returns
    -------
    int
        Maximum count of concurrent positions across all simulations.
    """
    events: List[tuple[pandas.Timestamp, int]] = []
    for simulation_result in simulation_results:
        for current_trade in simulation_result.trades:
            events.append((current_trade.entry_date, 1))
            events.append((current_trade.exit_date, -1))
    events.sort(key=lambda event: (event[0], event[1]))
    open_position_count = 0
    maximum_open_position_count = 0
    for _, change in events:
        open_position_count += change
        if open_position_count > maximum_open_position_count:
            maximum_open_position_count = open_position_count
    return maximum_open_position_count
# TODO: review
def simulate_portfolio_balance(
    trades: Iterable[Trade],
    starting_cash: float,
    maximum_positions: int,
) -> float:
    """Simulate capital allocation across multiple trades.

    Parameters
    ----------
    trades : Iterable[Trade]
        Collection of trades containing entry and exit information.
    starting_cash : float
        Initial cash available for trading.
    maximum_positions : int
        Maximum number of concurrent positions allowed.
    Each trade closure deducts ``TRADE_COMMISSION`` from the cash balance.

    Returns
    -------
    float
        Cash balance after all trades have been executed.
    """
    events: List[tuple[pandas.Timestamp, int, Trade]] = []
    for trade in trades:
        events.append((trade.entry_date, 1, trade))
        events.append((trade.exit_date, 0, trade))
    events.sort(key=lambda event_tuple: (event_tuple[0], event_tuple[1]))
    cash_balance = starting_cash
    open_trades: dict[Trade, float] = {}
    for event_timestamp, event_type, trade in events:
        if event_type == 0:
            if trade in open_trades:
                invested_amount = open_trades.pop(trade)
                cash_balance += invested_amount * (
                    trade.exit_price / trade.entry_price
                )
                cash_balance -= TRADE_COMMISSION
        else:
            if len(open_trades) >= maximum_positions or cash_balance <= 0:
                continue
            allocation = cash_balance / (maximum_positions - len(open_trades))
            open_trades[trade] = allocation
            cash_balance -= allocation
    return cash_balance


# TODO: review
def calculate_annual_returns(
    trades: Iterable[Trade],
    starting_cash: float,
    maximum_positions: int,
) -> Dict[int, float]:
    """Compute yearly portfolio returns for a series of trades.

    The portfolio balance is simulated using the same allocation logic as
    :func:`simulate_portfolio_balance`. The function records the portfolio
    value at the start and end of each calendar year and calculates the return
    for that year.

    Parameters
    ----------
    trades: Iterable[Trade]
        Collection of trades containing entry and exit information.
    starting_cash: float
        Initial cash available for trading.
    maximum_positions: int
        Maximum number of concurrent positions allowed.

    Returns
    -------
    Dict[int, float]
        Mapping of year to return percentage for that year. The value is
        expressed as a decimal where ``0.10`` represents a ten percent return.
    """
    events: List[tuple[pandas.Timestamp, int, Trade]] = []
    for trade in trades:
        events.append((trade.entry_date, 1, trade))
        events.append((trade.exit_date, 0, trade))
    events.sort(key=lambda event_tuple: (event_tuple[0], event_tuple[1]))

    cash_balance = starting_cash
    open_trades: dict[Trade, float] = {}
    annual_returns: Dict[int, float] = {}

    current_year = events[0][0].year if events else None
    year_start_value = cash_balance

    for event_timestamp, event_type, trade in events:
        if current_year is None:
            current_year = event_timestamp.year
            year_start_value = cash_balance + sum(open_trades.values())

        while event_timestamp.year > current_year:
            year_end_value = cash_balance + sum(open_trades.values())
            if year_start_value == 0:
                annual_returns[current_year] = 0.0
            else:
                annual_returns[current_year] = (
                    (year_end_value - year_start_value) / year_start_value
                )
            year_start_value = year_end_value
            current_year += 1

        if event_type == 0:
            if trade in open_trades:
                invested_amount = open_trades.pop(trade)
                cash_balance += invested_amount * (
                    trade.exit_price / trade.entry_price
                )
                cash_balance -= TRADE_COMMISSION
        else:
            if len(open_trades) >= maximum_positions or cash_balance <= 0:
                continue
            allocation = cash_balance / (maximum_positions - len(open_trades))
            open_trades[trade] = allocation
            cash_balance -= allocation

    if current_year is not None:
        year_end_value = cash_balance + sum(open_trades.values())
        if year_start_value == 0:
            annual_returns[current_year] = 0.0
        else:
            annual_returns[current_year] = (
                (year_end_value - year_start_value) / year_start_value
            )

    return annual_returns
