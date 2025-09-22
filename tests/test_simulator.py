"""Tests for trade simulation utilities."""
# TODO: review

import math
import os
import sys

import pandas
import pytest

from stock_indicator.indicators import sma

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from stock_indicator.simulator import (
    SimulationResult,
    Trade,
    calc_commission,
    calculate_maximum_concurrent_positions,
    calculate_annual_returns,
    calculate_annual_trade_counts,
    calculate_annual_profit_totals,
    simulate_trades,
    simulate_portfolio_balance,
    calculate_max_drawdown,
)


def test_calc_commission_uses_minimum_when_share_count_is_small() -> None:
    commission = calc_commission(shares=1, price=300.0)
    assert commission == pytest.approx(0.99)


def test_calc_commission_caps_at_percentage_of_trade_value() -> None:
    commission = calc_commission(shares=1_000_000, price=0.5)
    expected_commission = 0.005 * 1_000_000 * 0.5
    assert commission == pytest.approx(expected_commission)


def test_simulate_trades_executes_trade_flow_with_default_column() -> None:
    price_data_frame = pandas.DataFrame(
        {"close": [100.0, 102.0, 104.0, 103.0, 106.0]}
    )

    def entry_rule(current_row: pandas.Series) -> bool:
        return current_row["close"] > 101.0

    def exit_rule(current_row: pandas.Series, entry_row: pandas.Series) -> bool:
        return current_row["close"] > 105.0

    result = simulate_trades(price_data_frame, entry_rule, exit_rule)

    assert isinstance(result, SimulationResult)
    assert len(result.trades) == 1
    completed_trade = result.trades[0]
    expected_entry_date = price_data_frame.index[1]
    expected_exit_date = price_data_frame.index[4]
    assert completed_trade.entry_date == expected_entry_date
    assert completed_trade.exit_date == expected_exit_date
    assert completed_trade.entry_price == 102.0
    assert completed_trade.exit_price == 106.0
    expected_profit = (
        4.0 - calc_commission(1, 102.0) - calc_commission(1, 106.0)
    )
    assert completed_trade.profit == expected_profit
    assert completed_trade.holding_period == 3
    assert result.total_profit == expected_profit


def test_simulate_trades_with_sma_strategy_uses_aligned_labels() -> None:
    """Verify SMA-based rules use matching index labels during comparison."""
    price_data_frame = pandas.DataFrame(
        {"close": [100.0, 102.0, 104.0, 103.0, 106.0]},
        index=[10, 11, 12, 13, 14],
    )
    simple_moving_average_series = sma(
        price_data_frame["close"], window_size=2
    )
    simple_moving_average_series = simple_moving_average_series.iloc[::-1]

    def entry_rule(current_row: pandas.Series) -> bool:
        """Determine when to enter a trade based on SMA."""
        row_label = current_row.name
        indicator_at_label = simple_moving_average_series.loc[row_label]
        if pandas.isna(indicator_at_label):
            return False
        return current_row["close"] > indicator_at_label

    def exit_rule(
        current_row: pandas.Series, entry_row: pandas.Series
    ) -> bool:
        """Determine when to exit a trade based on SMA."""
        row_label = current_row.name
        indicator_at_label = simple_moving_average_series.loc[row_label]
        if pandas.isna(indicator_at_label):
            return False
        return current_row["close"] < indicator_at_label

    result = simulate_trades(
        price_data_frame, entry_rule, exit_rule, entry_price_column="close"
    )

    assert isinstance(result, SimulationResult)
    assert len(result.trades) == 1
    completed_trade = result.trades[0]
    expected_entry_date = price_data_frame.index[1]
    expected_exit_date = price_data_frame.index[3]
    assert completed_trade.entry_date == expected_entry_date
    assert completed_trade.exit_date == expected_exit_date
    assert completed_trade.entry_price == 102.0
    assert completed_trade.exit_price == 103.0
    expected_profit = (
        1.0 - calc_commission(1, 102.0) - calc_commission(1, 103.0)
    )
    assert completed_trade.profit == expected_profit
    assert completed_trade.holding_period == 2
    assert result.total_profit == expected_profit


def test_simulate_trades_handles_distinct_entry_and_exit_price_columns() -> None:
    price_data_frame = pandas.DataFrame(
        {"open": [10.0, 12.0], "close": [11.0, 13.0]}
    )

    def entry_rule(current_row: pandas.Series) -> bool:
        return current_row["open"] == 10.0

    def exit_rule(current_row: pandas.Series, entry_row: pandas.Series) -> bool:
        return current_row["close"] >= 13.0

    result = simulate_trades(
        price_data_frame,
        entry_rule,
        exit_rule,
        entry_price_column="open",
        exit_price_column="close",
    )

    assert isinstance(result, SimulationResult)
    assert len(result.trades) == 1
    completed_trade = result.trades[0]
    expected_entry_date = price_data_frame.index[0]
    expected_exit_date = price_data_frame.index[1]
    assert completed_trade.entry_date == expected_entry_date
    assert completed_trade.exit_date == expected_exit_date
    assert completed_trade.entry_price == 10.0
    assert completed_trade.exit_price == 13.0
    expected_profit = (
        3.0 - calc_commission(1, 10.0) - calc_commission(1, 13.0)
    )
    assert completed_trade.profit == expected_profit
    assert completed_trade.holding_period == 1
    assert result.total_profit == expected_profit


def test_simulate_trades_closes_open_position_at_end() -> None:
    """Open positions should close using the final available price."""
    price_data_frame = pandas.DataFrame({"close": [1.0, 2.0, 3.0]})

    def entry_rule(current_row: pandas.Series) -> bool:
        return current_row["close"] > 1.5

    def exit_rule(current_row: pandas.Series, entry_row: pandas.Series) -> bool:
        return False

    result = simulate_trades(price_data_frame, entry_rule, exit_rule)

    assert len(result.trades) == 1
    final_trade = result.trades[0]
    assert final_trade.exit_date == price_data_frame.index[-1]
    assert final_trade.exit_price == 3.0
    assert final_trade.holding_period == 1
    assert final_trade.exit_reason == "end_of_data"


def test_simulate_trades_applies_stop_loss_next_open() -> None:
    """Trades should close at the next open when the stop loss is reached."""
    price_data_frame = pandas.DataFrame(
        {
            "open": [100.0, 95.0, 96.0],
            "close": [100.0, 92.0, 97.0],
        }
    )

    def entry_rule(current_row: pandas.Series) -> bool:
        return current_row.name == 0

    def exit_rule(current_row: pandas.Series, entry_row: pandas.Series) -> bool:
        return False

    result = simulate_trades(
        price_data_frame,
        entry_rule,
        exit_rule,
        entry_price_column="open",
        exit_price_column="open",
        stop_loss_percentage=0.075,
    )

    assert len(result.trades) == 1
    completed_trade = result.trades[0]
    assert completed_trade.entry_price == 100.0
    assert completed_trade.exit_price == 96.0
    assert completed_trade.exit_date == price_data_frame.index[2]
    expected_profit = (
        -4.0 - calc_commission(1, 100.0) - calc_commission(1, 96.0)
    )
    assert completed_trade.profit == expected_profit
    assert completed_trade.holding_period == 2


def test_calculate_maximum_concurrent_positions_counts_overlaps() -> None:
    """Count overlapping trades across multiple simulations."""
    trade_alpha = Trade(
        entry_date=pandas.Timestamp("2024-01-01"),
        exit_date=pandas.Timestamp("2024-01-05"),
        entry_price=1.0,
        exit_price=1.0,
        profit=0.0,
        holding_period=0,
    )
    trade_beta = Trade(
        entry_date=pandas.Timestamp("2024-01-03"),
        exit_date=pandas.Timestamp("2024-01-04"),
        entry_price=1.0,
        exit_price=1.0,
        profit=0.0,
        holding_period=0,
    )
    trade_gamma = Trade(
        entry_date=pandas.Timestamp("2024-01-02"),
        exit_date=pandas.Timestamp("2024-01-06"),
        entry_price=1.0,
        exit_price=1.0,
        profit=0.0,
        holding_period=0,
    )

    result_alpha = SimulationResult(trades=[trade_alpha], total_profit=0.0)
    result_beta = SimulationResult(trades=[trade_beta], total_profit=0.0)
    result_gamma = SimulationResult(trades=[trade_gamma], total_profit=0.0)

    maximum_positions = calculate_maximum_concurrent_positions(
        [result_alpha, result_beta, result_gamma]
    )

    assert maximum_positions == 3


def test_calculate_maximum_concurrent_positions_orders_exit_before_entry() -> None:
    """Process exit events before entry events occurring on the same date."""
    trade_delta = Trade(
        entry_date=pandas.Timestamp("2024-01-01"),
        exit_date=pandas.Timestamp("2024-01-02"),
        entry_price=1.0,
        exit_price=1.0,
        profit=0.0,
        holding_period=0,
    )
    trade_epsilon = Trade(
        entry_date=pandas.Timestamp("2024-01-02"),
        exit_date=pandas.Timestamp("2024-01-03"),
        entry_price=1.0,
        exit_price=1.0,
        profit=0.0,
        holding_period=0,
    )

    result_delta = SimulationResult(trades=[trade_delta], total_profit=0.0)
    result_epsilon = SimulationResult(trades=[trade_epsilon], total_profit=0.0)

    maximum_positions = calculate_maximum_concurrent_positions(
        [result_delta, result_epsilon]
    )

    assert maximum_positions == 1


def test_simulate_portfolio_balance_uses_fixed_slot_weight() -> None:
    """Portfolio simulation should size positions using a fixed slot weight."""
    trade_alpha = Trade(
        entry_date=pandas.Timestamp("2024-01-01"),
        exit_date=pandas.Timestamp("2024-01-02"),
        entry_price=10.0,
        exit_price=20.0,
        profit=10.0,
        holding_period=1,
    )
    trade_beta = Trade(
        entry_date=pandas.Timestamp("2024-01-01"),
        exit_date=pandas.Timestamp("2024-01-02"),
        entry_price=10.0,
        exit_price=10.0,
        profit=0.0,
        holding_period=1,
    )
    final_balance = simulate_portfolio_balance(
        [trade_alpha, trade_beta], 100.0, 5
    )
    entry_commission_alpha = calc_commission(2, 10.0)
    entry_commission_beta = calc_commission(1, 10.0)
    cash_after_entries = (
        100.0
        - 2 * 10.0
        - entry_commission_alpha
        - 1 * 10.0
        - entry_commission_beta
    )
    exit_commission_alpha = calc_commission(2, 20.0)
    exit_commission_beta = calc_commission(1, 10.0)
    expected_final_balance = (
        cash_after_entries
        + 2 * 20.0
        - exit_commission_alpha
        + 1 * 10.0
        - exit_commission_beta
    )
    assert pytest.approx(final_balance, rel=1e-6) == expected_final_balance


def test_simulate_portfolio_balance_skips_trade_when_budget_insufficient() -> None:
    """Positions should be ignored when the slot budget cannot buy one share."""
    trade_primary = Trade(
        entry_date=pandas.Timestamp("2024-01-01"),
        exit_date=pandas.Timestamp("2024-01-03"),
        entry_price=30.0,
        exit_price=60.0,
        profit=30.0,
        holding_period=2,
    )
    trade_secondary = Trade(
        entry_date=pandas.Timestamp("2024-01-02"),
        exit_date=pandas.Timestamp("2024-01-04"),
        entry_price=70.0,
        exit_price=70.0,
        profit=0.0,
        holding_period=2,
    )
    final_balance = simulate_portfolio_balance(
        [trade_primary, trade_secondary], 100.0, 2
    )
    entry_commission = calc_commission(1, 30.0)
    cash_after_entry = 100.0 - 1 * 30.0 - entry_commission
    exit_commission = calc_commission(1, 60.0)
    expected_final_balance = cash_after_entry + 1 * 60.0 - exit_commission
    assert pytest.approx(final_balance, rel=1e-6) == expected_final_balance


def test_calculate_annual_returns_computes_yearly_returns() -> None:
    trade_one = Trade(
        entry_date=pandas.Timestamp("2023-01-10"),
        exit_date=pandas.Timestamp("2023-03-10"),
        entry_price=100.0,
        exit_price=110.0,
        profit=10.0
        - calc_commission(1, 100.0)
        - calc_commission(1, 110.0),
        holding_period=1,
    )
    trade_two = Trade(
        entry_date=pandas.Timestamp("2024-02-15"),
        exit_date=pandas.Timestamp("2024-06-15"),
        entry_price=200.0,
        exit_price=220.0,
        profit=20.0
        - calc_commission(1, 200.0)
        - calc_commission(1, 220.0),
        holding_period=1,
    )
    simulation_start = pandas.Timestamp("2018-01-01")
    annual_returns = calculate_annual_returns(
        [trade_one, trade_two],
        starting_cash=1000.0,
        maximum_position_count=1,
        simulation_start=simulation_start,
    )
    first_year_end = (
        1000.0 * (110.0 / 100.0)
        - calc_commission(10, 100.0)
        - calc_commission(10, 110.0)
    )
    expected_return_2023 = (first_year_end - 1000.0) / 1000.0
    share_count_year_two = math.floor(first_year_end / 200.0)
    second_year_end = (
        first_year_end
        - share_count_year_two * 200.0
        - calc_commission(share_count_year_two, 200.0)
        + share_count_year_two * 220.0
        - calc_commission(share_count_year_two, 220.0)
    )
    expected_return_2024 = (second_year_end - first_year_end) / first_year_end
    assert annual_returns[2018] == 0.0
    assert pytest.approx(annual_returns[2023], rel=1e-6) == expected_return_2023
    assert pytest.approx(annual_returns[2024], rel=1e-6) == expected_return_2024


def test_calculate_annual_returns_marks_unsettled_proceeds_in_year_end() -> None:
    """Unsettled proceeds should count toward the year they were earned."""
    starting_cash = 1000.0
    entry_price = 100.0
    exit_price = 110.0
    allocated_share_count = math.floor(starting_cash / entry_price)
    entry_commission = calc_commission(allocated_share_count, entry_price)
    exit_commission = calc_commission(allocated_share_count, exit_price)
    trade_profit = (
        allocated_share_count * (exit_price - entry_price)
        - entry_commission
        - exit_commission
    )
    trade_record = Trade(
        entry_date=pandas.Timestamp("2020-12-30"),
        exit_date=pandas.Timestamp("2020-12-31"),
        entry_price=entry_price,
        exit_price=exit_price,
        profit=trade_profit,
        holding_period=1,
    )
    simulation_start = pandas.Timestamp("2020-12-30")
    annual_returns = calculate_annual_returns(
        [trade_record],
        starting_cash=starting_cash,
        maximum_position_count=1,
        simulation_start=simulation_start,
        margin_interest_annual_rate=0.0,
        settlement_lag_days=1,
    )
    expected_return_for_2020 = trade_profit / starting_cash
    assert pytest.approx(annual_returns[2020], rel=1e-6) == expected_return_for_2020
    assert pytest.approx(annual_returns[2021], rel=1e-6) == 0.0


def test_simulate_portfolio_balance_applies_withdraw() -> None:
    """Portfolio simulation should deduct annual withdrawals."""
    trade_record = Trade(
        entry_date=pandas.Timestamp("2023-01-01"),
        exit_date=pandas.Timestamp("2023-01-02"),
        entry_price=100.0,
        exit_price=100.0,
        profit=0.0,
        holding_period=1,
    )
    final_balance = simulate_portfolio_balance(
        [trade_record],
        starting_cash=100.0,
        maximum_position_count=1,
        withdraw_amount=10.0,
    )
    expected_balance = 89.0
    assert pytest.approx(final_balance, rel=1e-6) == expected_balance


def test_calculate_annual_returns_applies_withdraw() -> None:
    """Annual return calculation should account for yearly withdrawals."""
    trade_one = Trade(
        entry_date=pandas.Timestamp("2023-01-01"),
        exit_date=pandas.Timestamp("2023-01-02"),
        entry_price=50.0,
        exit_price=60.0,
        profit=10.0
        - calc_commission(1, 50.0)
        - calc_commission(1, 60.0),
        holding_period=1,
    )
    trade_two = Trade(
        entry_date=pandas.Timestamp("2024-01-01"),
        exit_date=pandas.Timestamp("2024-01-02"),
        entry_price=50.0,
        exit_price=60.0,
        profit=10.0
        - calc_commission(1, 50.0)
        - calc_commission(1, 60.0),
        holding_period=1,
    )
    simulation_start = pandas.Timestamp("2023-01-01")
    annual_returns = calculate_annual_returns(
        [trade_one, trade_two],
        starting_cash=100.0,
        maximum_position_count=1,
        simulation_start=simulation_start,
        withdraw_amount=10.0,
    )
    first_year_end = (
        100.0 * (60.0 / 50.0)
        - calc_commission(2, 50.0)
        - calc_commission(2, 60.0)
    )
    expected_return_2023 = (first_year_end - 100.0) / 100.0
    second_year_start = first_year_end - 10.0
    share_count_year_two = math.floor(second_year_start / 50.0)
    second_year_end = (
        second_year_start
        - share_count_year_two * 50.0
        - calc_commission(share_count_year_two, 50.0)
        + share_count_year_two * 60.0
        - calc_commission(share_count_year_two, 60.0)
    )
    expected_return_2024 = (
        (second_year_end - second_year_start) / second_year_start
    )
    assert pytest.approx(annual_returns[2023], rel=1e-6) == expected_return_2023
    assert pytest.approx(annual_returns[2024], rel=1e-6) == expected_return_2024


def test_calculate_annual_trade_counts_counts_trades_per_year() -> None:
    trade_alpha = Trade(
        entry_date=pandas.Timestamp("2023-01-01"),
        exit_date=pandas.Timestamp("2023-02-01"),
        entry_price=10.0,
        exit_price=11.0,
        profit=1.0
        - calc_commission(1, 10.0)
        - calc_commission(1, 11.0),
        holding_period=1,
    )
    trade_beta = Trade(
        entry_date=pandas.Timestamp("2024-03-01"),
        exit_date=pandas.Timestamp("2024-04-01"),
        entry_price=10.0,
        exit_price=12.0,
        profit=2.0
        - calc_commission(1, 10.0)
        - calc_commission(1, 12.0),
        holding_period=1,
    )
    trade_gamma = Trade(
        entry_date=pandas.Timestamp("2024-05-01"),
        exit_date=pandas.Timestamp("2024-06-01"),
        entry_price=10.0,
        exit_price=9.0,
        profit=-1.0
        - calc_commission(1, 10.0)
        - calc_commission(1, 9.0),
        holding_period=1,
    )
    trade_counts = calculate_annual_trade_counts(
        [trade_alpha, trade_beta, trade_gamma]
    )
    assert trade_counts == {2023: 1, 2024: 2}


def test_calculate_annual_profit_totals_sums_profit_by_exit_year() -> None:
    trade_winner = Trade(
        entry_date=pandas.Timestamp("2023-01-01"),
        exit_date=pandas.Timestamp("2023-02-01"),
        entry_price=10.0,
        exit_price=12.0,
        profit=2.0
        - calc_commission(1, 10.0)
        - calc_commission(1, 12.0),
        holding_period=1,
    )
    trade_loser = Trade(
        entry_date=pandas.Timestamp("2024-03-01"),
        exit_date=pandas.Timestamp("2024-04-01"),
        entry_price=10.0,
        exit_price=8.0,
        profit=-2.0
        - calc_commission(1, 10.0)
        - calc_commission(1, 8.0),
        holding_period=1,
    )
    annual_profit_totals = calculate_annual_profit_totals(
        [trade_winner, trade_loser]
    )
    expected_totals = {
        2023: pytest.approx(trade_winner.profit, rel=1e-9),
        2024: pytest.approx(trade_loser.profit, rel=1e-9),
    }
    assert annual_profit_totals.keys() == expected_totals.keys()
    for year_value, expected_profit in expected_totals.items():
        assert annual_profit_totals[year_value] == expected_profit


def test_calculate_max_drawdown_marks_to_market() -> None:
    """calculate_max_drawdown should revalue open positions using closing prices."""
    trade = Trade(
        entry_date=pandas.Timestamp("2020-01-01"),
        exit_date=pandas.Timestamp("2020-01-04"),
        entry_price=10.0,
        exit_price=12.0,
        profit=2.0 - calc_commission(1, 10.0) - calc_commission(1, 12.0),
        holding_period=3,
    )
    trade_symbol_lookup = {trade: "AAA"}
    closing_price_series_by_symbol = {
        "AAA": pandas.Series(
            [10.0, 8.0, 12.0],
            index=pandas.to_datetime([
                "2020-01-01",
                "2020-01-03",
                "2020-01-04",
            ]),
        )
    }
    maximum_drawdown_value = calculate_max_drawdown(
        [trade],
        starting_cash=1000.0,
        maximum_position_count=1,
        trade_symbol_lookup=trade_symbol_lookup,
        closing_price_series_by_symbol=closing_price_series_by_symbol,
        withdraw_amount=0.0,
    )
    entry_commission = calc_commission(100, 10.0)
    cash_after_entry = 1000.0 - 100 * 10.0 - entry_commission
    lowest_portfolio_value = cash_after_entry + 100 * 8.0
    expected_drawdown = (1000.0 - lowest_portfolio_value) / 1000.0
    assert maximum_drawdown_value == pytest.approx(expected_drawdown)
