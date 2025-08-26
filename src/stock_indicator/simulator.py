"""Simulation utilities for executing trading strategies."""
# TODO: review

from __future__ import annotations

import math
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
    eligible_symbol_count: int,
) -> float:
    """Simulate capital allocation across multiple trades.

    Parameters
    ----------
    trades : Iterable[Trade]
        Collection of trades containing entry and exit information.
    starting_cash : float
        Initial cash available for trading.
    eligible_symbol_count : int
        Total number of symbols considered for trading. This value is typically
        produced by :func:`stock_indicator.volume.count_symbols_with_average_dollar_volume_above`.
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
            remaining_slots = eligible_symbol_count - len(open_trades)
            if remaining_slots <= 0 or cash_balance <= 0:
                continue
            budget_per_position = cash_balance / remaining_slots
            # TODO: review
            share_count = math.floor(budget_per_position / trade.entry_price)
            if share_count <= 0:
                continue
            invested_amount = share_count * trade.entry_price
            if cash_balance - invested_amount >= trade.entry_price:
                share_count += 1
                invested_amount = share_count * trade.entry_price
            open_trades[trade] = invested_amount
            cash_balance -= invested_amount
    return cash_balance


def simulate_and_withdraw(
    trades: Iterable[Trade],
    starting_cash: float,
    eligible_symbol_count: int,
    annual_withdrawal: float,
) -> float:
    """Simulate capital allocation with yearly cash withdrawals.

    This function mirrors :func:`simulate_portfolio_balance` but deducts
    ``annual_withdrawal`` from the cash balance at the end of each calendar
    year. The withdrawal only affects available cash and does not force the
    closure of open positions.

    Parameters
    ----------
    trades:
        Collection of trades containing entry and exit information.
    starting_cash:
        Initial cash available for trading.
    eligible_symbol_count:
        Total number of symbols considered for trading. This value is typically
        produced by
        :func:`stock_indicator.volume.count_symbols_with_average_dollar_volume_above`.
    annual_withdrawal:
        Amount of cash removed from the portfolio at the end of each calendar
        year. When the cash balance is less than the withdrawal amount, the
        balance is reduced to zero.

    Returns
    -------
    float
        Cash balance after executing all trades and annual withdrawals.
    """
    events: List[tuple[pandas.Timestamp, int, Trade]] = []
    for trade in trades:
        events.append((trade.entry_date, 1, trade))
        events.append((trade.exit_date, 0, trade))
    events.sort(key=lambda event_tuple: (event_tuple[0], event_tuple[1]))
    cash_balance = starting_cash
    open_trades: dict[Trade, float] = {}
    current_year = events[0][0].year if events else pandas.Timestamp.now().year
    for event_timestamp, event_type, trade in events:
        while event_timestamp.year > current_year:
            cash_balance = max(cash_balance - annual_withdrawal, 0.0)
            current_year += 1
        if event_type == 0:
            if trade in open_trades:
                invested_amount = open_trades.pop(trade)
                cash_balance += invested_amount * (
                    trade.exit_price / trade.entry_price
                )
                cash_balance -= TRADE_COMMISSION
        else:
            remaining_slots = eligible_symbol_count - len(open_trades)
            if remaining_slots <= 0 or cash_balance <= 0:
                continue
            budget_per_position = cash_balance / remaining_slots
            share_count = math.floor(budget_per_position / trade.entry_price)
            if share_count <= 0:
                continue
            invested_amount = share_count * trade.entry_price
            if cash_balance - invested_amount >= trade.entry_price:
                share_count += 1
                invested_amount = share_count * trade.entry_price
            open_trades[trade] = invested_amount
            cash_balance -= invested_amount
    cash_balance = max(cash_balance - annual_withdrawal, 0.0)
    return cash_balance


def calculate_annual_returns(
    trades: Iterable[Trade],
    starting_cash: float,
    eligible_symbol_count: int,
    simulation_start: pandas.Timestamp,
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
    eligible_symbol_count: int
        Total number of symbols considered for trading. This value is typically
        produced by :func:`stock_indicator.volume.count_symbols_with_average_dollar_volume_above`.
    simulation_start: pandas.Timestamp
        Timestamp indicating the first day of the simulation. Years prior to
        the first trade will be emitted with a zero return.

    Returns
    -------
    Dict[int, float]
        Mapping of year to return percentage for that year. The value is
        expressed as a decimal where ``0.10`` represents a ten percent return.
    """  # TODO: review
    events: List[tuple[pandas.Timestamp, int, Trade]] = []
    for completed_trade in trades:
        events.append((completed_trade.entry_date, 1, completed_trade))
        events.append((completed_trade.exit_date, 0, completed_trade))
    events.sort(key=lambda event_tuple: (event_tuple[0], event_tuple[1]))

    cash_balance = starting_cash
    open_trades: dict[Trade, float] = {}
    annual_returns: Dict[int, float] = {}

    current_year = simulation_start.year
    year_start_value = cash_balance

    for event_timestamp, event_type, completed_trade in events:
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
            if completed_trade in open_trades:
                invested_amount = open_trades.pop(completed_trade)
                cash_balance += invested_amount * (
                    completed_trade.exit_price / completed_trade.entry_price
                )
                cash_balance -= TRADE_COMMISSION
        else:
            remaining_slots = eligible_symbol_count - len(open_trades)
            if remaining_slots <= 0 or cash_balance <= 0:
                continue
            budget_per_position = cash_balance / remaining_slots
            # TODO: review
            share_count = math.floor(
                budget_per_position / completed_trade.entry_price
            )
            if share_count <= 0:
                continue
            invested_amount = share_count * completed_trade.entry_price
            if cash_balance - invested_amount >= completed_trade.entry_price:
                share_count += 1
                invested_amount = share_count * completed_trade.entry_price
            open_trades[completed_trade] = invested_amount
            cash_balance -= invested_amount

    year_end_value = cash_balance + sum(open_trades.values())
    if year_start_value == 0:
        annual_returns[current_year] = 0.0
    else:
        annual_returns[current_year] = (
            (year_end_value - year_start_value) / year_start_value
        )

    return annual_returns


def calculate_annual_returns_with_withdrawal(
    trades: Iterable[Trade],
    starting_cash: float,
    eligible_symbol_count: int,
    simulation_start: pandas.Timestamp,
    annual_withdrawal: float,
) -> Dict[int, float]:
    """Compute yearly portfolio returns with annual cash withdrawals.

    The logic matches :func:`calculate_annual_returns` but subtracts
    ``annual_withdrawal`` from the cash balance at the end of each calendar
    year before the next year's calculation begins. The withdrawal does not
    alter the invested value of open positions.

    Parameters
    ----------
    trades:
        Collection of trades containing entry and exit information.
    starting_cash:
        Initial cash available for trading.
    eligible_symbol_count:
        Total number of symbols considered for trading.
    simulation_start:
        Timestamp indicating the first day of the simulation.
    annual_withdrawal:
        Amount of cash removed from the portfolio at the end of each year.

    Returns
    -------
    Dict[int, float]
        Mapping of year to return percentage for that year.
    """
    events: List[tuple[pandas.Timestamp, int, Trade]] = []
    for completed_trade in trades:
        events.append((completed_trade.entry_date, 1, completed_trade))
        events.append((completed_trade.exit_date, 0, completed_trade))
    events.sort(key=lambda event_tuple: (event_tuple[0], event_tuple[1]))

    cash_balance = starting_cash
    open_trades: dict[Trade, float] = {}
    annual_returns: Dict[int, float] = {}

    current_year = simulation_start.year
    year_start_value = cash_balance

    for event_timestamp, event_type, completed_trade in events:
        while event_timestamp.year > current_year:
            year_end_value = cash_balance + sum(open_trades.values())
            if year_start_value == 0:
                annual_returns[current_year] = 0.0
            else:
                annual_returns[current_year] = (
                    (year_end_value - year_start_value) / year_start_value
                )
            cash_balance = max(cash_balance - annual_withdrawal, 0.0)
            year_start_value = cash_balance + sum(open_trades.values())
            current_year += 1

        if event_type == 0:
            if completed_trade in open_trades:
                invested_amount = open_trades.pop(completed_trade)
                cash_balance += invested_amount * (
                    completed_trade.exit_price / completed_trade.entry_price
                )
                cash_balance -= TRADE_COMMISSION
        else:
            remaining_slots = eligible_symbol_count - len(open_trades)
            if remaining_slots <= 0 or cash_balance <= 0:
                continue
            budget_per_position = cash_balance / remaining_slots
            share_count = math.floor(
                budget_per_position / completed_trade.entry_price
            )
            if share_count <= 0:
                continue
            invested_amount = share_count * completed_trade.entry_price
            if cash_balance - invested_amount >= completed_trade.entry_price:
                share_count += 1
                invested_amount = share_count * completed_trade.entry_price
            open_trades[completed_trade] = invested_amount
            cash_balance -= invested_amount

    year_end_value = cash_balance + sum(open_trades.values())
    if year_start_value == 0:
        annual_returns[current_year] = 0.0
    else:
        annual_returns[current_year] = (
            (year_end_value - year_start_value) / year_start_value
        )

    return annual_returns


def calculate_annual_trade_counts(trades: Iterable[Trade]) -> Dict[int, int]:
    """Count completed trades for each calendar year.

    Parameters
    ----------
    trades:
        Collection of trades containing exit dates.

    Returns
    -------
    Dict[int, int]
        Mapping of year to number of trades that closed during that year.
    """
    trade_counts: Dict[int, int] = {}
    for completed_trade in trades:
        year_value = completed_trade.exit_date.year
        if year_value in trade_counts:
            trade_counts[year_value] += 1
        else:
            trade_counts[year_value] = 1
    return trade_counts
