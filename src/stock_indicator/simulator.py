"""Simulation utilities for executing trading strategies."""
# TODO: review

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import pandas


def calc_commission(shares: float, price: float) -> float:
    """Calculate the commission for a trade.

    Parameters
    ----------
    shares:
        Number of shares traded.
    price:
        Price per share.

    Returns
    -------
    float
        Commission charged for the trade side. The commission is ``0.0049``
        dollars per share with a minimum charge of ``0.99`` dollars and a
        maximum of ``0.5`` percent of the trade value.
    """
    basic_commission = shares * 0.0049
    commission_with_minimum = max(0.99, basic_commission)
    commission_cap = 0.005 * shares * price
    return min(commission_with_minimum, commission_cap)

@dataclass(frozen=True)
class Trade:
    """Record details for a completed trade."""

    entry_date: pandas.Timestamp
    exit_date: pandas.Timestamp
    entry_price: float
    exit_price: float
    profit: float
    holding_period: int
    exit_reason: str = "signal"


@dataclass
class SimulationResult:
    """Aggregate outcome of a trade simulation."""

    trades: List[Trade]
    total_profit: float


@dataclass
class OpenPosition:
    """Track details for an active portfolio position."""

    invested_amount: float
    share_count: float
    symbol: str | None = None
    last_known_price: float | None = None


def simulate_trades(
    data: pandas.DataFrame,
    entry_rule: Callable[[pandas.Series], bool],
    exit_rule: Callable[[pandas.Series, pandas.Series], bool],
    entry_price_column: str = "close",
    exit_price_column: str | None = None,
    stop_loss_percentage: float = 1.0,
    cooldown_bars: int = 0,
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
        Fractional loss from the entry price for a stop order. When the current
        bar's low touches the stop price, the position exits on the same bar at
        the stop price (cap on loss). If the bar never reaches the stop but the
        close is below the stop, an exit is scheduled for the next bar's open.
        Values greater than or equal to ``1.0`` disable the stop-loss.
    cooldown_bars: int, default 0
        Minimum number of bars that must pass after a position closes before a
        new position may be opened. A value of ``5`` blocks re-entry for the
        next five bars following the exit bar.

    Commissions are calculated separately on entry and exit using
    :func:`calc_commission` and deducted from each trade's profit.

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
    last_exit_index: int | None = None
    for row_index in range(len(data)):
        current_row = data.iloc[row_index]
        is_last_row = row_index == len(data) - 1
        if not in_position:
            if is_last_row:
                continue
            # Enforce cool-down window after the most recent exit
            if (
                entry_rule(current_row)
                and not (
                    last_exit_index is not None
                    and (row_index - last_exit_index) <= max(0, cooldown_bars)
                )
            ):
                in_position = True
                entry_row = current_row
                entry_row_index = row_index
        else:
            if entry_row is None or entry_row_index is None:
                continue
            # Intraday stop-loss: if the bar's low touches the stop price,
            # exit on this bar at the stop price (caps loss at threshold).
            if 0 < stop_loss_percentage < 1:
                entry_price = float(entry_row[entry_price_column])
                stop_price = entry_price * (1 - stop_loss_percentage)
                # Prefer using the 'low' column when available; fallback to 'close'
                # if 'low' does not exist in the dataset.
                has_low = "low" in data.columns
                if has_low and float(current_row["low"]) <= stop_price:
                    price_column_name = (
                        exit_price_column if exit_price_column is not None else entry_price_column
                    )
                    # Fill at the stop price to emulate a stop order execution.
                    exit_price = float(stop_price)
                    entry_commission = calc_commission(1, entry_price)
                    exit_commission = calc_commission(1, exit_price)
                    profit_value = (
                        exit_price - entry_price - entry_commission - exit_commission
                    )
                    holding_period_value = row_index - entry_row_index
                    trades.append(
                        Trade(
                            entry_date=data.index[entry_row_index],
                            exit_date=data.index[row_index],
                            entry_price=entry_price,
                            exit_price=exit_price,
                            profit=profit_value,
                            holding_period=holding_period_value,
                            exit_reason="stop_loss",
                        )
                    )
                    in_position = False
                    entry_row = None
                    entry_row_index = None
                    stop_loss_pending = False
                    last_exit_index = row_index
                    continue
            if stop_loss_pending:
                entry_price = float(entry_row[entry_price_column])
                price_column_name = (
                    exit_price_column if exit_price_column is not None else entry_price_column
                )
                exit_price = float(current_row[price_column_name])
                entry_commission = calc_commission(1, entry_price)
                exit_commission = calc_commission(1, exit_price)
                profit_value = (
                    exit_price - entry_price - entry_commission - exit_commission
                )
                holding_period_value = row_index - entry_row_index
                trades.append(
                    Trade(
                        entry_date=data.index[entry_row_index],
                        exit_date=data.index[row_index],
                        entry_price=entry_price,
                        exit_price=exit_price,
                        profit=profit_value,
                        holding_period=holding_period_value,
                        exit_reason="stop_loss",
                    )
                )
                in_position = False
                entry_row = None
                entry_row_index = None
                stop_loss_pending = False
                last_exit_index = row_index
                continue
            if exit_rule(current_row, entry_row):
                entry_price = float(entry_row[entry_price_column])
                price_column_name = (
                    exit_price_column if exit_price_column is not None else entry_price_column
                )
                exit_price = float(current_row[price_column_name])
                entry_commission = calc_commission(1, entry_price)
                exit_commission = calc_commission(1, exit_price)
                profit_value = (
                    exit_price - entry_price - entry_commission - exit_commission
                )
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
                last_exit_index = row_index
                continue
            if 0 < stop_loss_percentage < 1 and not is_last_row:
                entry_price = float(entry_row[entry_price_column])
                # For trailing behavior across days using close only: if today's
                # close is beyond the stop threshold (but intraday did not hit),
                # schedule an exit on the next bar's open.
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
        entry_commission = calc_commission(1, entry_price)
        exit_commission = calc_commission(1, exit_price)
        profit_value = exit_price - entry_price - entry_commission - exit_commission
        holding_period_value = len(data) - entry_row_index - 1
        trades.append(
            Trade(
                entry_date=data.index[entry_row_index],
                exit_date=data.index[-1],
                entry_price=entry_price,
                exit_price=exit_price,
                profit=profit_value,
                holding_period=holding_period_value,
                exit_reason="end_of_data",
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
    maximum_position_count: int,
    withdraw_amount: float = 0.0,
    maximum_position_weight: float = 1.0,
    margin_multiplier: float = 1.0,
    margin_interest_annual_rate: float = 0.048,
    settlement_lag_days: int = 1,
) -> float:
    """Simulate capital allocation across multiple trades.

    Parameters
    ----------
    trades : Iterable[Trade]
        Collection of trades containing entry and exit information.
    starting_cash : float
        Initial cash available for trading.
    maximum_position_count : int
        Maximum number of concurrent open positions allowed.
    withdraw_amount : float, optional
        Cash amount removed from the balance at the end of each calendar year.
        Defaults to ``0.0``.
    maximum_position_weight : float, optional
        Upper bound on the fraction of equity allocated to a single position.
        Defaults to ``1.0`` which indicates no additional cap beyond the slot
        sizing limit.

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
    open_trades: dict[Trade, OpenPosition] = {}
    current_year: int | None = None
    last_event_date: pandas.Timestamp | None = None

    slot_weight = min(margin_multiplier / maximum_position_count, maximum_position_weight)

    # Track delayed proceeds to be credited on settlement dates
    pending_credits: dict[pandas.Timestamp, float] = {}

    for event_timestamp, event_type, trade in events:
        if current_year is None:
            current_year = event_timestamp.year
            last_event_date = event_timestamp
        # Before processing this event, apply any settlements due up to this timestamp,
        # accruing interest up to each credit date.
        if last_event_date is not None and event_timestamp > last_event_date:
            # Process credits in chronological order up to event_timestamp
            while True:
                due_dates = [d for d in pending_credits.keys() if d <= event_timestamp]
                if not due_dates:
                    break
                next_credit_date = min(due_dates)
                # Accrue interest from last_event_date to next_credit_date
                days_elapsed = (next_credit_date - last_event_date).days
                if days_elapsed > 0 and cash_balance < 0:
                    daily_rate = margin_interest_annual_rate / 365.0
                    cash_balance -= (-cash_balance) * daily_rate * days_elapsed
                # Apply credit
                cash_balance += pending_credits.pop(next_credit_date)
                last_event_date = next_credit_date
            # Finally accrue from last_event_date to event_timestamp
            days_elapsed = (event_timestamp - last_event_date).days
            if days_elapsed > 0 and cash_balance < 0:
                daily_rate = margin_interest_annual_rate / 365.0
                cash_balance -= (-cash_balance) * daily_rate * days_elapsed
            last_event_date = event_timestamp
        while event_timestamp.year > current_year:
            cash_balance -= withdraw_amount
            current_year += 1
        if event_type == 0:
            if trade in open_trades:
                position_details = open_trades.pop(trade)
                proceeds = position_details.share_count * trade.exit_price
                exit_commission = calc_commission(
                    position_details.share_count, trade.exit_price
                )
                credit_date = event_timestamp + pandas.Timedelta(days=settlement_lag_days)
                pending_credits[credit_date] = pending_credits.get(credit_date, 0.0) + (
                    proceeds - exit_commission
                )
        else:
            if len(open_trades) >= maximum_position_count:
                continue
            equity = cash_balance + sum(
                position.invested_amount for position in open_trades.values()
            )
            budget_per_position = equity * slot_weight
            share_count = math.floor(budget_per_position / trade.entry_price)
            if share_count <= 0:
                continue
            invested_amount = share_count * trade.entry_price
            entry_commission = calc_commission(share_count, trade.entry_price)
            open_trades[trade] = OpenPosition(
                invested_amount=invested_amount, share_count=share_count
            )
            cash_balance -= invested_amount + entry_commission

    # After processing all events, settle any remaining pending credits
    # (e.g., proceeds from the last exit with T+settlement_lag_days).
    # Accrue margin interest day-by-day up to each credit date and apply
    # year-end withdrawals that occur between the last processed event and
    # each settlement date.
    if last_event_date is not None and pending_credits:
        for credit_date in sorted(pending_credits.keys()):
            # Apply withdrawals for any year boundaries crossed before this credit
            if current_year is not None:
                while credit_date.year > current_year:
                    cash_balance -= withdraw_amount
                    current_year += 1
            # Accrue interest from the last processed date to the credit date
            days_elapsed = (credit_date - last_event_date).days
            if days_elapsed > 0 and cash_balance < 0:
                daily_rate = margin_interest_annual_rate / 365.0
                cash_balance -= (-cash_balance) * daily_rate * days_elapsed
            # Apply the credit and advance the clock
            cash_balance += pending_credits[credit_date]
            last_event_date = credit_date
        pending_credits.clear()

    # Apply the final year-end withdrawal once for the last simulation year.
    if current_year is not None:
        cash_balance -= withdraw_amount
    return cash_balance


def calculate_max_drawdown(
    trades: Iterable[Trade],
    starting_cash: float,
    maximum_position_count: int,
    trade_symbol_lookup: Dict[Trade, str],
    closing_price_series_by_symbol: Dict[str, pandas.Series],
    withdraw_amount: float = 0.0,
    maximum_position_weight: float = 1.0,
    margin_multiplier: float = 1.0,
    margin_interest_annual_rate: float = 0.048,
) -> float:
    """Compute the maximum portfolio drawdown across the simulation period.

    The function simulates the portfolio balance for each calendar day using
    the same allocation logic as :func:`simulate_portfolio_balance`. It tracks
    the running peak balance and records the greatest percentage decline from
    that peak. The drawdown is expressed as a decimal where ``0.25`` denotes a
    twenty-five percent drop.

    Parameters
    ----------
    trades: Iterable[Trade]
        Collection of trades with entry and exit information.
    starting_cash: float
        Initial cash available for trading.
    maximum_position_count: int
        Maximum number of concurrent open positions allowed.
    trade_symbol_lookup: Dict[Trade, str]
        Mapping from each trade to its associated trading symbol.
    closing_price_series_by_symbol: Dict[str, pandas.Series]
        Mapping of symbol to a series of daily closing prices used to revalue
        open positions.
    withdraw_amount: float, optional
        Cash amount removed from the balance at the end of each calendar year.
        Defaults to ``0.0``.
    maximum_position_weight: float, optional
        Upper bound on the fraction of equity allocated to a single position.
        Defaults to ``1.0`` which indicates no additional cap beyond the slot
        sizing limit.

    Returns
    -------
    float
        Largest fractional decline from any previous portfolio peak.
    """

    events: List[tuple[pandas.Timestamp, int, Trade]] = []
    for current_trade in trades:
        events.append((current_trade.entry_date, 1, current_trade))
        events.append((current_trade.exit_date, 0, current_trade))
    events.sort(key=lambda event_tuple: (event_tuple[0], event_tuple[1]))
    if not events:
        return 0.0

    cash_balance = starting_cash
    open_trades: dict[Trade, OpenPosition] = {}
    maximum_portfolio_value = starting_cash
    maximum_drawdown_value = 0.0
    current_year = events[0][0].year
    slot_weight = min(margin_multiplier / maximum_position_count, maximum_position_weight)

    start_date = events[0][0]
    # Run through settlement of the last exit as well
    end_date = events[-1][0] + pandas.Timedelta(days=1)
    event_index = 0
    current_date = start_date
    # Pending credits scheduled for settlement (T+1)
    pending_credits: Dict[pandas.Timestamp, float] = {}

    while current_date <= end_date:
        # Apply any credits due today before handling events
        if current_date in pending_credits:
            cash_balance += pending_credits.pop(current_date)
        while event_index < len(events) and events[event_index][0] == current_date:
            _, event_type, trade = events[event_index]
            symbol_name = trade_symbol_lookup.get(trade, "")
            if event_type == 0:
                if trade in open_trades:
                    position_details = open_trades.pop(trade)
                    proceeds = position_details.share_count * trade.exit_price
                    exit_commission = calc_commission(
                        position_details.share_count, trade.exit_price
                    )
                    credit_date = current_date + pandas.Timedelta(days=1)
                    pending_credits[credit_date] = pending_credits.get(credit_date, 0.0) + (
                        proceeds - exit_commission
                    )
            else:
                if len(open_trades) < maximum_position_count and cash_balance > 0:
                    equity = cash_balance + sum(
                        position.share_count * (position.last_known_price or 0.0)
                        for position in open_trades.values()
                    )
                    budget_per_position = equity * slot_weight
                    share_count = math.floor(budget_per_position / trade.entry_price)
                    if share_count > 0:
                        invested_amount = share_count * trade.entry_price
                        entry_commission = calc_commission(share_count, trade.entry_price)
                        open_trades[trade] = OpenPosition(
                            invested_amount=invested_amount,
                            share_count=share_count,
                            symbol=symbol_name,
                            last_known_price=trade.entry_price,
                        )
                        cash_balance -= invested_amount + entry_commission
            event_index += 1

        for position_details in open_trades.values():
            price_series = closing_price_series_by_symbol.get(position_details.symbol)
            if price_series is not None and current_date in price_series.index:
                position_details.last_known_price = float(price_series.loc[current_date])

        # Accrue daily margin interest on negative cash
        if cash_balance < 0:
            cash_balance -= (-cash_balance) * (margin_interest_annual_rate / 365.0)

        # Portfolio value includes cash, MTM of open positions, and receivables
        # from pending settlements (e.g., T+1 proceeds). Including receivables
        # prevents artificial dips on exit days that would otherwise inflate
        # drawdown.
        portfolio_value = (
            cash_balance
            + sum(
                position_details.share_count * (position_details.last_known_price or 0.0)
                for position_details in open_trades.values()
            )
            + sum(pending_credits.values())
        )
        if portfolio_value > maximum_portfolio_value:
            maximum_portfolio_value = portfolio_value
        elif maximum_portfolio_value > 0:
            drawdown = (maximum_portfolio_value - portfolio_value) / maximum_portfolio_value
            if drawdown > maximum_drawdown_value:
                maximum_drawdown_value = drawdown

        next_day = current_date + pandas.Timedelta(days=1)
        if next_day.year > current_year:
            cash_balance -= withdraw_amount
            maximum_portfolio_value = max(0.0, maximum_portfolio_value - withdraw_amount)
            portfolio_value = (
                cash_balance
                + sum(
                    position_details.share_count * (position_details.last_known_price or 0.0)
                    for position_details in open_trades.values()
                )
                + sum(pending_credits.values())
            )
            if portfolio_value > maximum_portfolio_value:
                maximum_portfolio_value = portfolio_value
            elif maximum_portfolio_value > 0:
                drawdown = (
                    (maximum_portfolio_value - portfolio_value) / maximum_portfolio_value
                )
                if drawdown > maximum_drawdown_value:
                    maximum_drawdown_value = drawdown
            current_year = next_day.year

        current_date = next_day

    return maximum_drawdown_value


def calculate_annual_returns(
    trades: Iterable[Trade],
    starting_cash: float,
    maximum_position_count: int,
    simulation_start: pandas.Timestamp,
    withdraw_amount: float = 0.0,
    maximum_position_weight: float = 1.0,
    margin_multiplier: float = 1.0,
    margin_interest_annual_rate: float = 0.048,
    trade_symbol_lookup: Dict[Trade, str] | None = None,
    closing_price_series_by_symbol: Dict[str, pandas.Series] | None = None,
    settlement_lag_days: int = 1,
) -> Dict[int, float]:
    """Compute yearly portfolio returns with mark-to-market and margin interest.

    This routine simulates day-by-day portfolio value, including:
    - Slot allocation with margin (budget per position = equity * margin / slots)
    - Daily interest on negative cash at ``margin_interest_annual_rate / 365``
    - Proceeds credited on exit settlement day (``T+settlement_lag_days``)
    - Mark-to-market valuation of open positions using daily closes
    """
    # Build event schedule
    events_by_day: Dict[pandas.Timestamp, List[tuple[int, Trade]]] = {}
    start_date = simulation_start
    end_date = simulation_start
    for trade in trades:
        events_by_day.setdefault(trade.entry_date, []).append((1, trade))
        events_by_day.setdefault(trade.exit_date, []).append((0, trade))
        # Ensure the simulation runs through settlement of the last exit
        candidate_end = trade.exit_date + pandas.Timedelta(days=settlement_lag_days)
        if candidate_end > end_date:
            end_date = candidate_end
    if not trades:
        return {}

    cash_balance = starting_cash
    open_trades: dict[Trade, OpenPosition] = {}
    annual_returns: Dict[int, float] = {}
    current_year = start_date.year
    year_start_value = starting_cash
    slot_weight = min(margin_multiplier / maximum_position_count, maximum_position_weight)
    # Pending cash credits scheduled for settlement
    pending_credits: Dict[pandas.Timestamp, float] = {}

    current_date = start_date
    last_processed_date = start_date
    while current_date <= end_date:
        # Credit any settlements due today
        if current_date in pending_credits:
            cash_balance += pending_credits.pop(current_date)

        # Process all events on this day
        for event_type, trade in events_by_day.get(current_date, []):
            if event_type == 0:  # exit
                if trade in open_trades:
                    position_details = open_trades.pop(trade)
                    proceeds = position_details.share_count * trade.exit_price
                    exit_commission = calc_commission(position_details.share_count, trade.exit_price)
                    credit_date = current_date + pandas.Timedelta(days=settlement_lag_days)
                    pending_credits[credit_date] = pending_credits.get(credit_date, 0.0) + (proceeds - exit_commission)
            else:  # entry
                if len(open_trades) >= maximum_position_count:
                    continue
                equity = cash_balance + sum(
                    position.invested_amount for position in open_trades.values()
                )
                budget_per_position = equity * slot_weight
                share_count = math.floor(budget_per_position / trade.entry_price)
                if share_count <= 0:
                    continue
                invested_amount = share_count * trade.entry_price
                entry_commission = calc_commission(share_count, trade.entry_price)
                open_trades[trade] = OpenPosition(
                    invested_amount=invested_amount, share_count=share_count
                )
                cash_balance -= invested_amount + entry_commission

        # Update MTM using closes for today
        if trade_symbol_lookup is not None and closing_price_series_by_symbol is not None:
            for trade, position in open_trades.items():
                symbol_name = trade_symbol_lookup.get(trade)
                if not symbol_name:
                    continue
                series = closing_price_series_by_symbol.get(symbol_name)
                if series is not None and current_date in series.index:
                    position.last_known_price = float(series.loc[current_date])

        # Accrue daily interest on negative cash
        if cash_balance < 0:
            cash_balance -= (-cash_balance) * (margin_interest_annual_rate / 365.0)

        # Year roll handling on Jan 1 or at end_date
        next_date = current_date + pandas.Timedelta(days=1)
        next_year = next_date.year
        if next_year != current_year or current_date == end_date:
            unsettled_proceeds_value = sum(pending_credits.values())
            # Mark unsettled exit proceeds to market by including pending credits.
            portfolio_value = cash_balance + unsettled_proceeds_value + sum(
                pos.share_count * (pos.last_known_price or 0.0)
                for pos in open_trades.values()
            )
            if year_start_value == 0:
                annual_returns[current_year] = 0.0
            else:
                annual_returns[current_year] = (portfolio_value - year_start_value) / year_start_value
            # Apply withdrawal at year end
            cash_balance -= withdraw_amount
            # Carry the mark-to-market unsettled proceeds into the new year baseline.
            year_start_value = cash_balance + unsettled_proceeds_value + sum(
                pos.share_count * (pos.last_known_price or 0.0)
                for pos in open_trades.values()
            )
            current_year = next_year

        current_date = next_date

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


def calculate_annual_profit_totals(trades: Iterable[Trade]) -> Dict[int, float]:
    """Sum realized trade profits for each calendar year.

    Parameters
    ----------
    trades:
        Collection of completed trades whose profits should be grouped by
        calendar year based on exit timestamps.

    Returns
    -------
    Dict[int, float]
        Mapping of calendar year to the total realized profit for trades that
        closed during that year.
    """

    annual_profit_totals: Dict[int, float] = {}
    for completed_trade in trades:
        year_value = completed_trade.exit_date.year
        profit_value = float(completed_trade.profit)
        if year_value in annual_profit_totals:
            annual_profit_totals[year_value] += profit_value
        else:
            annual_profit_totals[year_value] = profit_value
    return annual_profit_totals
