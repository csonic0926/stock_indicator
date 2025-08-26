"""Tests for trade simulation utilities."""
# TODO: review

import math
import os
import sys

import pandas
import pytest

from stock_indicator.indicators import sma

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from stock_indicator.simulator import (
    TRADE_COMMISSION,
    SimulationResult,
    Trade,
    calculate_maximum_concurrent_positions,
    calculate_annual_returns,
    calculate_annual_returns_with_withdrawal,
    calculate_annual_trade_counts,
    simulate_trades,
    simulate_portfolio_balance,
    simulate_and_withdraw,
)


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
    expected_profit = 4.0 - TRADE_COMMISSION
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
    expected_profit = 1.0 - TRADE_COMMISSION
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
    expected_profit = 3.0 - TRADE_COMMISSION
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
    expected_profit = -4.0 - TRADE_COMMISSION
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


def test_simulate_portfolio_balance_allocates_budget_by_symbol_count() -> None:
    """Portfolio simulation should allocate cash based on remaining symbols."""
    trade_alpha = Trade(
        entry_date=pandas.Timestamp("2024-01-01"),
        exit_date=pandas.Timestamp("2024-01-05"),
        entry_price=10.0,
        exit_price=20.0,
        profit=10.0,
        holding_period=4,
    )
    trade_beta = Trade(
        entry_date=pandas.Timestamp("2024-01-02"),
        exit_date=pandas.Timestamp("2024-01-06"),
        entry_price=10.0,
        exit_price=10.0,
        profit=0.0,
        holding_period=4,
    )
    trade_gamma = Trade(
        entry_date=pandas.Timestamp("2024-01-03"),
        exit_date=pandas.Timestamp("2024-01-04"),
        entry_price=10.0,
        exit_price=20.0,
        profit=10.0,
        holding_period=1,
    )
    final_balance = simulate_portfolio_balance(
        [trade_alpha, trade_beta, trade_gamma], 100.0, 3
    )
    expected_final_balance = 157.0
    assert pytest.approx(final_balance, rel=1e-6) == expected_final_balance


def test_simulate_portfolio_balance_invests_additional_share_when_cash_available() -> None:
    """Portfolio simulation should buy an extra share if cash remains."""
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
    expected_final_balance = 159.0
    assert pytest.approx(final_balance, rel=1e-6) == expected_final_balance


def test_simulate_and_withdraw_deducts_cash_each_year() -> None:
    """Yearly withdrawals should reduce the final cash balance."""
    trade_first = Trade(
        entry_date=pandas.Timestamp("2023-01-10"),
        exit_date=pandas.Timestamp("2023-01-20"),
        entry_price=10.0,
        exit_price=20.0,
        profit=10.0,
        holding_period=10,
    )
    trade_second = Trade(
        entry_date=pandas.Timestamp("2024-02-10"),
        exit_date=pandas.Timestamp("2024-02-20"),
        entry_price=10.0,
        exit_price=20.0,
        profit=10.0,
        holding_period=10,
    )
    final_balance = simulate_and_withdraw(
        [trade_first, trade_second], 100.0, 1, 100.0
    )
    expected_final_balance = 88.0
    assert pytest.approx(final_balance, rel=1e-6) == expected_final_balance


def test_calculate_annual_returns_with_withdrawal_accounts_for_deductions() -> None:
    """Annual returns should be computed prior to withdrawals."""
    trade_first = Trade(
        entry_date=pandas.Timestamp("2023-01-10"),
        exit_date=pandas.Timestamp("2023-01-20"),
        entry_price=10.0,
        exit_price=20.0,
        profit=10.0,
        holding_period=10,
    )
    trade_second = Trade(
        entry_date=pandas.Timestamp("2024-02-10"),
        exit_date=pandas.Timestamp("2024-02-20"),
        entry_price=10.0,
        exit_price=20.0,
        profit=10.0,
        holding_period=10,
    )
    annual_returns = calculate_annual_returns_with_withdrawal(
        [trade_first, trade_second],
        starting_cash=100.0,
        eligible_symbol_count=1,
        simulation_start=pandas.Timestamp("2023-01-01"),
        annual_withdrawal=100.0,
    )
    expected_return_2023 = (199.0 - 100.0) / 100.0
    expected_return_2024 = (188.0 - 99.0) / 99.0
    assert pytest.approx(annual_returns[2023], rel=1e-6) == expected_return_2023
    assert pytest.approx(annual_returns[2024], rel=1e-6) == expected_return_2024


def test_calculate_annual_returns_computes_yearly_returns() -> None:
    trade_one = Trade(
        entry_date=pandas.Timestamp("2023-01-10"),
        exit_date=pandas.Timestamp("2023-03-10"),
        entry_price=100.0,
        exit_price=110.0,
        profit=10.0 - TRADE_COMMISSION,
        holding_period=1,
    )
    trade_two = Trade(
        entry_date=pandas.Timestamp("2024-02-15"),
        exit_date=pandas.Timestamp("2024-06-15"),
        entry_price=200.0,
        exit_price=220.0,
        profit=20.0 - TRADE_COMMISSION,
        holding_period=1,
    )
    simulation_start = pandas.Timestamp("2018-01-01")
    annual_returns = calculate_annual_returns(
        [trade_one, trade_two],
        starting_cash=1000.0,
        eligible_symbol_count=1,
        simulation_start=simulation_start,
    )
    first_year_end = 1000.0 * (110.0 / 100.0) - TRADE_COMMISSION
    expected_return_2023 = (first_year_end - 1000.0) / 1000.0
    share_count_year_two = math.floor(first_year_end / 200.0)
    remaining_cash_year_two = first_year_end - share_count_year_two * 200.0
    second_year_end = (
        remaining_cash_year_two + share_count_year_two * 220.0 - TRADE_COMMISSION
    )
    expected_return_2024 = (second_year_end - first_year_end) / first_year_end
    assert annual_returns[2018] == 0.0
    assert pytest.approx(annual_returns[2023], rel=1e-6) == expected_return_2023
    assert pytest.approx(annual_returns[2024], rel=1e-6) == expected_return_2024


def test_calculate_annual_trade_counts_counts_trades_per_year() -> None:
    trade_alpha = Trade(
        entry_date=pandas.Timestamp("2023-01-01"),
        exit_date=pandas.Timestamp("2023-02-01"),
        entry_price=10.0,
        exit_price=11.0,
        profit=1.0 - TRADE_COMMISSION,
        holding_period=1,
    )
    trade_beta = Trade(
        entry_date=pandas.Timestamp("2024-03-01"),
        exit_date=pandas.Timestamp("2024-04-01"),
        entry_price=10.0,
        exit_price=12.0,
        profit=2.0 - TRADE_COMMISSION,
        holding_period=1,
    )
    trade_gamma = Trade(
        entry_date=pandas.Timestamp("2024-05-01"),
        exit_date=pandas.Timestamp("2024-06-01"),
        entry_price=10.0,
        exit_price=9.0,
        profit=-1.0 - TRADE_COMMISSION,
        holding_period=1,
    )
    trade_counts = calculate_annual_trade_counts(
        [trade_alpha, trade_beta, trade_gamma]
    )
    assert trade_counts == {2023: 1, 2024: 2}
