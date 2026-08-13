"""Tests for trade simulation utilities."""
# TODO: review

import math
import os
import statistics
import sys

import pandas
import pytest

from stock_indicator.indicators import sma

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from stock_indicator.simulator import (
    BROKER_COST_MODELS,
    DEFAULT_BROKER_COST_MODEL_NAME,
    TRADING_DAYS_PER_YEAR,
    SimulationResult,
    Trade,
    calc_commission,
    calculate_maximum_concurrent_positions,
    calculate_annual_returns,
    calculate_annual_trade_counts,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    get_active_broker_cost_model,
    override_broker_cost_model,
    resolve_broker_cost_model,
    simulate_trades,
    simulate_portfolio_balance,
    calculate_max_drawdown,
)


def test_calculate_sharpe_ratio_uses_sample_standard_deviation() -> None:
    # TODO: review
    period_returns = [0.10, -0.05, 0.20]

    sharpe_ratio = calculate_sharpe_ratio(period_returns)

    expected_ratio = (
        math.sqrt(TRADING_DAYS_PER_YEAR)
        * statistics.mean(period_returns)
        / statistics.stdev(period_returns)
    )
    assert sharpe_ratio == pytest.approx(expected_ratio)


def test_calculate_sharpe_ratio_handles_unmeasurable_volatility() -> None:
    # TODO: review
    assert calculate_sharpe_ratio([]) == 0.0
    assert calculate_sharpe_ratio([0.10]) == 0.0
    assert calculate_sharpe_ratio([0.10, 0.10]) == math.inf
    assert calculate_sharpe_ratio([0.0, 0.0]) == 0.0


def test_calculate_sortino_ratio_uses_zero_target_downside_deviation() -> None:
    # TODO: review
    period_returns = [0.10, -0.05, 0.20]

    sortino_ratio = calculate_sortino_ratio(period_returns)

    downside_deviation = math.sqrt((0.05 ** 2) / len(period_returns))
    expected_ratio = (
        math.sqrt(TRADING_DAYS_PER_YEAR)
        * statistics.mean(period_returns)
        / downside_deviation
    )
    assert sortino_ratio == pytest.approx(expected_ratio)


def test_calculate_sortino_ratio_handles_missing_downside_returns() -> None:
    # TODO: review
    assert calculate_sortino_ratio([]) == 0.0
    assert calculate_sortino_ratio([0.05, 0.10]) == math.inf
    assert calculate_sortino_ratio([0.0, 0.0]) == 0.0


def test_calc_commission_uses_minimum_when_share_count_is_small() -> None:
    # One share leaves both Futu per-order minimums binding: 0.99 commission
    # plus 1.00 platform fee, then 0.003 settlement on the single share.
    commission = calc_commission(shares=1, price=300.0)
    assert commission == pytest.approx(0.99 + 1.00 + 0.003, abs=1e-4)


def test_calc_commission_caps_commission_and_platform_fee_but_not_settlement() -> None:
    commission = calc_commission(shares=1_000_000, price=0.5)
    capped_commission_and_platform_fee = 0.005 * 1_000_000 * 0.5
    settlement_fee = 1_000_000 * 0.003
    cat_fee = 1_000_000 * 0.000046
    assert commission == pytest.approx(
        capped_commission_and_platform_fee + settlement_fee + cat_fee
    )


def test_default_broker_cost_model_is_futu() -> None:
    assert get_active_broker_cost_model().name == DEFAULT_BROKER_COST_MODEL_NAME
    assert DEFAULT_BROKER_COST_MODEL_NAME == "futu_hk"


def test_override_broker_cost_model_switches_and_restores() -> None:
    original_commission = calc_commission(shares=1_000, price=50.0)
    with override_broker_cost_model("ibkr_fixed") as active_model:
        assert active_model.name == "ibkr_fixed"
        ibkr_commission = calc_commission(shares=1_000, price=50.0)
    assert get_active_broker_cost_model().name == DEFAULT_BROKER_COST_MODEL_NAME
    assert calc_commission(shares=1_000, price=50.0) == pytest.approx(
        original_commission
    )
    # Futu bills 0.0129 per share against IBKR Fixed's 0.005 per share.
    assert ibkr_commission < original_commission


def test_ibkr_tiered_costs_more_than_fixed_for_market_orders() -> None:
    with override_broker_cost_model("ibkr_fixed"):
        fixed_commission = calc_commission(shares=1_000, price=50.0)
    with override_broker_cost_model("ibkr_tiered"):
        tiered_commission = calc_commission(shares=1_000, price=50.0)
    assert tiered_commission > fixed_commission


def test_resolve_broker_cost_model_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown broker_cost_model"):
        resolve_broker_cost_model("no_such_broker")


def test_every_registered_broker_cost_model_is_self_consistent() -> None:
    for model_name, model in BROKER_COST_MODELS.items():
        assert model.name == model_name
        assert model.margin_interest_annual_rate > 0.0
        buy_fee = model.commission_function(1_000, 50.0, False)
        sell_fee = model.commission_function(1_000, 50.0, True)
        assert buy_fee > 0.0
        assert sell_fee >= buy_fee


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
    # simulate_trades reports gross profit; commission is applied by
    # simulate_portfolio_balance, not at the trade level.
    expected_profit = 4.0
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
    # simulate_trades reports gross profit; commission is applied by
    # simulate_portfolio_balance, not at the trade level.
    expected_profit = 1.0
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
    # simulate_trades reports gross profit; commission is applied by
    # simulate_portfolio_balance, not at the trade level.
    expected_profit = 3.0
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


def test_simulate_trades_applies_stop_loss_at_gap_down_open() -> None:
    """Stop-loss market orders should exit at the open on a gap down."""
    price_data_frame = pandas.DataFrame(
        {
            "open": [100.0, 90.0],
            "low": [100.0, 89.0],
            "close": [100.0, 91.0],
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
    assert completed_trade.exit_price == 90.0
    assert completed_trade.exit_date == price_data_frame.index[1]
    assert completed_trade.exit_reason == "stop_loss"


def test_simulate_trades_applies_stop_loss_intraday_trigger() -> None:
    """Stop-loss market orders should exit at stop price after intraday trigger."""
    price_data_frame = pandas.DataFrame(
        {
            "open": [100.0, 95.0],
            "low": [100.0, 92.0],
            "close": [100.0, 94.0],
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
    assert completed_trade.exit_price == pytest.approx(92.5)
    assert completed_trade.exit_date == price_data_frame.index[1]
    assert completed_trade.exit_reason == "stop_loss"


def test_simulate_trades_places_stop_loss_after_minimum_hold_passes() -> None:
    """Stop-loss should not be active until the session after min hold passes."""
    price_data_frame = pandas.DataFrame(
        {
            "open": [100.0, 95.0, 90.0],
            "low": [100.0, 80.0, 89.0],
            "close": [100.0, 81.0, 91.0],
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
        minimum_holding_bars=1,
    )

    assert len(result.trades) == 1
    completed_trade = result.trades[0]
    assert completed_trade.exit_price == 90.0
    assert completed_trade.exit_date == price_data_frame.index[2]
    assert completed_trade.exit_reason == "stop_loss"


def test_simulate_trades_applies_take_profit_intraday() -> None:
    """Trades should close immediately when the profit target is reached intraday."""

    price_data_frame = pandas.DataFrame(
        {
            "open": [100.0, 102.0],
            "high": [100.0, 112.0],
            "close": [100.0, 111.0],
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
        take_profit_percentage=0.1,
    )

    assert len(result.trades) == 1
    completed_trade = result.trades[0]
    assert completed_trade.exit_price == pytest.approx(110.0)
    assert completed_trade.exit_date == price_data_frame.index[1]
    assert completed_trade.exit_reason == "take_profit"


def test_simulate_trades_applies_take_profit_next_open() -> None:
    """Trades should close on the next open when only the close beats the target."""

    price_data_frame = pandas.DataFrame(
        {
            "open": [100.0, 104.0, 115.0],
            "high": [100.0, 108.0, 120.0],
            "close": [100.0, 111.0, 118.0],
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
        take_profit_percentage=0.1,
    )

    assert len(result.trades) == 1
    completed_trade = result.trades[0]
    assert completed_trade.exit_date == price_data_frame.index[2]
    assert completed_trade.exit_price == pytest.approx(110.0)
    assert completed_trade.exit_reason == "take_profit"


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
    exit_commission_alpha = calc_commission(2, 20.0, is_sell=True)
    exit_commission_beta = calc_commission(1, 10.0, is_sell=True)
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
    exit_commission = calc_commission(1, 60.0, is_sell=True)
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
        - calc_commission(1, 110.0, is_sell=True),
        holding_period=1,
    )
    trade_two = Trade(
        entry_date=pandas.Timestamp("2024-02-15"),
        exit_date=pandas.Timestamp("2024-06-15"),
        entry_price=200.0,
        exit_price=220.0,
        profit=20.0
        - calc_commission(1, 200.0)
        - calc_commission(1, 220.0, is_sell=True),
        holding_period=1,
    )
    simulation_start = pandas.Timestamp("2018-01-01")
    annual_returns = calculate_annual_returns(
        [trade_one, trade_two],
        starting_cash=1000.0,
        maximum_position_count=1,
        simulation_start=simulation_start,
        margin_interest_annual_rate=0.0,
    )
    first_year_end = (
        1000.0 * (110.0 / 100.0)
        - calc_commission(10, 100.0)
        - calc_commission(10, 110.0, is_sell=True)
    )
    expected_return_2023 = (first_year_end - 1000.0) / 1000.0
    share_count_year_two = math.floor(first_year_end / 200.0)
    second_year_end = (
        first_year_end
        - share_count_year_two * 200.0
        - calc_commission(share_count_year_two, 200.0)
        + share_count_year_two * 220.0
        - calc_commission(share_count_year_two, 220.0, is_sell=True)
    )
    expected_return_2024 = (second_year_end - first_year_end) / first_year_end
    assert annual_returns[2018] == 0.0
    assert pytest.approx(annual_returns[2023], rel=1e-6) == expected_return_2023
    assert pytest.approx(annual_returns[2024], rel=1e-6) == expected_return_2024


def test_calculate_annual_returns_collects_daily_portfolio_returns() -> None:
    """Daily returns should follow mark-to-market portfolio values."""
    # TODO: review
    starting_cash = 1000.0
    entry_price = 10.0
    exit_price = 10.0
    allocated_share_count = math.floor(starting_cash / entry_price)
    trade_record = Trade(
        entry_date=pandas.Timestamp("2020-01-02"),
        exit_date=pandas.Timestamp("2020-01-06"),
        entry_price=entry_price,
        exit_price=exit_price,
        profit=0.0,
        holding_period=2,
    )
    closing_prices = pandas.Series(
        [10.0, 8.0, 10.0],
        index=pandas.to_datetime(
            ["2020-01-02", "2020-01-03", "2020-01-06"]
        ),
    )
    daily_portfolio_returns: list[float] = []

    calculate_annual_returns(
        [trade_record],
        starting_cash=starting_cash,
        maximum_position_count=1,
        simulation_start=pandas.Timestamp("2020-01-02"),
        margin_interest_annual_rate=0.0,
        trade_symbol_lookup={trade_record: "AAA"},
        closing_price_series_by_symbol={"AAA": closing_prices},
        daily_portfolio_returns_output=daily_portfolio_returns,
    )

    entry_commission = calc_commission(
        allocated_share_count,
        entry_price,
    )
    exit_commission = calc_commission(
        allocated_share_count,
        exit_price,
        is_sell=True,
    )
    entry_day_portfolio_value = starting_cash - entry_commission
    drawdown_day_portfolio_value = (
        starting_cash
        - allocated_share_count * entry_price
        - entry_commission
        + allocated_share_count * 8.0
    )
    exit_day_portfolio_value = (
        starting_cash - entry_commission - exit_commission
    )
    expected_daily_returns = [
        (entry_day_portfolio_value - starting_cash) / starting_cash,
        (
            drawdown_day_portfolio_value - entry_day_portfolio_value
        ) / entry_day_portfolio_value,
        (
            exit_day_portfolio_value - drawdown_day_portfolio_value
        ) / drawdown_day_portfolio_value,
        0.0,
    ]
    assert daily_portfolio_returns == pytest.approx(expected_daily_returns)


def test_calculate_annual_returns_honors_inclusive_simulation_end() -> None:
    """A bounded report should retain flat returns through its requested end."""
    # TODO: review
    trade_record = Trade(
        entry_date=pandas.Timestamp("2024-01-02"),
        exit_date=pandas.Timestamp("2024-01-03"),
        entry_price=10.0,
        exit_price=10.0,
        profit=0.0,
        holding_period=1,
    )
    valuation_dates = pandas.bdate_range("2024-01-02", "2024-01-10")
    closing_prices = pandas.Series(10.0, index=valuation_dates)
    daily_portfolio_returns: list[float] = []

    calculate_annual_returns(
        [trade_record],
        starting_cash=1000.0,
        maximum_position_count=1,
        simulation_start=pandas.Timestamp("2024-01-02"),
        simulation_end=pandas.Timestamp("2024-01-10"),
        margin_interest_annual_rate=0.0,
        trade_symbol_lookup={trade_record: "AAA"},
        closing_price_series_by_symbol={"AAA": closing_prices},
        daily_portfolio_returns_output=daily_portfolio_returns,
    )

    assert len(daily_portfolio_returns) == len(valuation_dates)
    assert daily_portfolio_returns[-1] == 0.0


def test_calculate_annual_returns_marks_unsettled_proceeds_in_year_end() -> None:
    """Unsettled proceeds should count toward the year they were earned."""
    starting_cash = 1000.0
    entry_price = 100.0
    exit_price = 110.0
    allocated_share_count = math.floor(starting_cash / entry_price)
    entry_commission = calc_commission(allocated_share_count, entry_price)
    exit_commission = calc_commission(allocated_share_count, exit_price, is_sell=True)
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
        margin_interest_annual_rate=0.0,
    )
    expected_balance = (
        100.0
        - 10.0
        - calc_commission(1, 100.0)
        - calc_commission(1, 100.0, is_sell=True)
    )
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
        - calc_commission(1, 60.0, is_sell=True),
        holding_period=1,
    )
    trade_two = Trade(
        entry_date=pandas.Timestamp("2024-01-01"),
        exit_date=pandas.Timestamp("2024-01-02"),
        entry_price=50.0,
        exit_price=60.0,
        profit=10.0
        - calc_commission(1, 50.0)
        - calc_commission(1, 60.0, is_sell=True),
        holding_period=1,
    )
    simulation_start = pandas.Timestamp("2023-01-01")
    annual_returns = calculate_annual_returns(
        [trade_one, trade_two],
        starting_cash=100.0,
        maximum_position_count=1,
        simulation_start=simulation_start,
        withdraw_amount=10.0,
        margin_interest_annual_rate=0.0,
    )
    first_year_end = (
        100.0 * (60.0 / 50.0)
        - calc_commission(2, 50.0)
        - calc_commission(2, 60.0, is_sell=True)
    )
    expected_return_2023 = (first_year_end - 100.0) / 100.0
    second_year_start = first_year_end - 10.0
    share_count_year_two = math.floor(second_year_start / 50.0)
    second_year_end = (
        second_year_start
        - share_count_year_two * 50.0
        - calc_commission(share_count_year_two, 50.0)
        + share_count_year_two * 60.0
        - calc_commission(share_count_year_two, 60.0, is_sell=True)
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
        - calc_commission(1, 11.0, is_sell=True),
        holding_period=1,
    )
    trade_beta = Trade(
        entry_date=pandas.Timestamp("2024-03-01"),
        exit_date=pandas.Timestamp("2024-04-01"),
        entry_price=10.0,
        exit_price=12.0,
        profit=2.0
        - calc_commission(1, 10.0)
        - calc_commission(1, 12.0, is_sell=True),
        holding_period=1,
    )
    trade_gamma = Trade(
        entry_date=pandas.Timestamp("2024-05-01"),
        exit_date=pandas.Timestamp("2024-06-01"),
        entry_price=10.0,
        exit_price=9.0,
        profit=-1.0
        - calc_commission(1, 10.0)
        - calc_commission(1, 9.0, is_sell=True),
        holding_period=1,
    )
    trade_counts = calculate_annual_trade_counts(
        [trade_alpha, trade_beta, trade_gamma]
    )
    assert trade_counts == {2023: 1, 2024: 2}


def test_calculate_max_drawdown_marks_to_market() -> None:
    """calculate_max_drawdown should revalue open positions using closing prices."""
    trade = Trade(
        entry_date=pandas.Timestamp("2020-01-01"),
        exit_date=pandas.Timestamp("2020-01-04"),
        entry_price=10.0,
        exit_price=12.0,
        profit=2.0 - calc_commission(1, 10.0) - calc_commission(1, 12.0, is_sell=True),
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
        margin_interest_annual_rate=0.0,
    )
    entry_commission = calc_commission(100, 10.0)
    cash_after_entry = 1000.0 - 100 * 10.0 - entry_commission
    lowest_portfolio_value = cash_after_entry + 100 * 8.0
    expected_drawdown = (1000.0 - lowest_portfolio_value) / 1000.0
    assert maximum_drawdown_value == pytest.approx(expected_drawdown)


def test_simulate_trades_refire_resets_min_hold_blocks_signal_exit() -> None:
    """A re-fire during hold should reset bars_since_anchor so an exit
    signal that would normally fire stays gated for another min_hold bars.

    Setup: entry at bar 0, exit_rule fires at bar 5 and again at bar 9.
    Without re-fire, min_hold=4 lets bar 5 close the trade.
    With re-fire at bar 5 (entry_rule re-fires there), bars_since_anchor
    resets to 0; next valid signal exit is gated until bars_since_anchor
    reaches 4 again — bar 9, which is the next time exit_rule fires.
    """
    closes = [100.0] * 12
    # Re-fire bars: 0 (initial entry) and 5 (re-fire while held).
    fire_bars = {0, 5}
    # Exit signal bars: 5 and 9.
    exit_bars = {5, 9}

    def entry_rule(current_row: pandas.Series) -> bool:
        return current_row.name in fire_bars

    def exit_rule(current_row: pandas.Series, entry_row: pandas.Series) -> bool:
        return current_row.name in exit_bars

    price_data_frame = pandas.DataFrame({"close": closes})

    baseline = simulate_trades(
        price_data_frame, entry_rule, exit_rule,
        minimum_holding_bars=4,
        reset_hold_on_reentry_signal=False,
    )
    treatment = simulate_trades(
        price_data_frame, entry_rule, exit_rule,
        minimum_holding_bars=4,
        reset_hold_on_reentry_signal=True,
    )

    assert len(baseline.trades) == 1
    assert baseline.trades[0].exit_date == 5

    assert len(treatment.trades) == 1
    assert treatment.trades[0].exit_date == 9


def test_simulate_portfolio_balance_margin_overrides_reduce_sizing() -> None:
    """Per-month margin_overrides must shrink the slot weight for trades
    entered in those months — final balance differs because the same
    trade gets sized smaller when its entry month is overridden.
    """
    trade_a = Trade(
        entry_date=pandas.Timestamp("2010-01-15"),
        exit_date=pandas.Timestamp("2010-01-25"),
        entry_price=100.0,
        exit_price=120.0,  # +20% winner
        profit=20.0,
        holding_period=10,
    )
    trade_b = Trade(
        entry_date=pandas.Timestamp("2010-04-15"),
        exit_date=pandas.Timestamp("2010-04-25"),
        entry_price=100.0,
        exit_price=120.0,
        profit=20.0,
        holding_period=10,
    )

    # Baseline: full margin 1.5 for both trades.
    balance_full = simulate_trades  # only to satisfy linter (unused). Real call below.
    balance_full = simulate_portfolio_balance(
        trades=[trade_a, trade_b],
        starting_cash=10_000.0,
        maximum_position_count=2,
        margin_multiplier=1.5,
    )
    # Treatment: Jan 2010 forced down to margin 1.0; Apr 2010 unchanged.
    balance_gated = simulate_portfolio_balance(
        trades=[trade_a, trade_b],
        starting_cash=10_000.0,
        maximum_position_count=2,
        margin_multiplier=1.5,
        margin_overrides={"2010-01": 1.0},
    )
    # Both trades are winners, but the Jan trade in the gated run is
    # sized smaller (margin 1.0 vs 1.5) so the gated final balance must
    # be strictly lower (less leveraged upside).
    assert balance_gated < balance_full
