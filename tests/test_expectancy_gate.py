"""Tests for the simulation-only rolling expectancy circuit breaker."""

# TODO: review

from __future__ import annotations

from dataclasses import asdict
import io
import json
from pathlib import Path

import pandas
import pytest

from stock_indicator import manage, simulator, strategy


def _build_trade(
    entry_date: str,
    exit_date: str,
    percentage_change: float,
    symbol_name: str,
) -> tuple[strategy.Trade, tuple[strategy.TradeDetail, strategy.TradeDetail]]:
    """Build one trade and its reporting details."""

    entry_timestamp = pandas.Timestamp(entry_date)
    exit_timestamp = pandas.Timestamp(exit_date)
    entry_price = 100.0
    exit_price = entry_price * (1.0 + percentage_change)
    completed_trade = strategy.Trade(
        entry_date=entry_timestamp,
        exit_date=exit_timestamp,
        entry_price=entry_price,
        exit_price=exit_price,
        profit=exit_price - entry_price,
        holding_period=(exit_timestamp - entry_timestamp).days,
        exit_reason="signal",
    )
    entry_detail = strategy.TradeDetail(
        date=entry_timestamp,
        symbol=symbol_name,
        action="open",
        price=entry_price,
        simple_moving_average_dollar_volume=0.0,
        total_simple_moving_average_dollar_volume=0.0,
        simple_moving_average_dollar_volume_ratio=0.0,
    )
    exit_detail = strategy.TradeDetail(
        date=exit_timestamp,
        symbol=symbol_name,
        action="close",
        price=exit_price,
        simple_moving_average_dollar_volume=0.0,
        total_simple_moving_average_dollar_volume=0.0,
        simple_moving_average_dollar_volume_ratio=0.0,
        result="win" if percentage_change > 0 else "lose",
        percentage_change=percentage_change,
    )
    return completed_trade, (entry_detail, exit_detail)


def _build_artifacts(
    trades_with_details: list[
        tuple[
            strategy.Trade,
            tuple[strategy.TradeDetail, strategy.TradeDetail],
        ]
    ],
) -> strategy.StrategyEvaluationArtifacts:
    """Build deterministic strategy artifacts for a simulation fixture."""

    completed_trades = [
        completed_trade
        for completed_trade, _detail_pair in trades_with_details
    ]
    trade_symbol_lookup = {
        completed_trade: detail_pair[0].symbol
        for completed_trade, detail_pair in trades_with_details
    }
    closing_price_series_by_symbol = {
        detail_pair[0].symbol: pandas.Series(
            [detail_pair[0].price, detail_pair[1].price],
            index=[detail_pair[0].date, detail_pair[1].date],
        )
        for _completed_trade, detail_pair in trades_with_details
    }
    trade_detail_pairs = {
        completed_trade: detail_pair
        for completed_trade, detail_pair in trades_with_details
    }
    return strategy.StrategyEvaluationArtifacts(
        trades=completed_trades,
        simulation_results=[
            strategy.SimulationResult(
                trades=completed_trades,
                total_profit=sum(
                    completed_trade.profit
                    for completed_trade in completed_trades
                ),
            )
        ],
        trade_symbol_lookup=trade_symbol_lookup,
        closing_price_series_by_symbol=closing_price_series_by_symbol,
        trade_detail_pairs=trade_detail_pairs,
        simulation_start_date=min(
            completed_trade.entry_date
            for completed_trade in completed_trades
        ),
    )


def _stub_money_simulations(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, list[tuple[list[strategy.Trade], dict[str, object]]]]:
    """Replace money simulations and record their trade-weight arguments."""

    call_records: dict[
        str, list[tuple[list[strategy.Trade], dict[str, object]]]
    ] = {
        "annual_returns": [],
        "portfolio_balance": [],
        "maximum_drawdown": [],
    }

    def fake_calculate_annual_returns(
        completed_trades: list[strategy.Trade],
        *unused_arguments: object,
        **keyword_arguments: object,
    ) -> dict[int, float]:
        call_records["annual_returns"].append(
            (list(completed_trades), dict(keyword_arguments))
        )
        return {}

    def fake_simulate_portfolio_balance(
        completed_trades: list[strategy.Trade],
        starting_cash: float,
        *unused_arguments: object,
        **keyword_arguments: object,
    ) -> float:
        call_records["portfolio_balance"].append(
            (list(completed_trades), dict(keyword_arguments))
        )
        return starting_cash

    def fake_calculate_max_drawdown(
        completed_trades: list[strategy.Trade],
        *unused_arguments: object,
        **keyword_arguments: object,
    ) -> float:
        call_records["maximum_drawdown"].append(
            (list(completed_trades), dict(keyword_arguments))
        )
        return 0.0

    monkeypatch.setattr(
        strategy,
        "calculate_annual_returns",
        fake_calculate_annual_returns,
    )
    monkeypatch.setattr(
        strategy,
        "simulate_portfolio_balance",
        fake_simulate_portfolio_balance,
    )
    monkeypatch.setattr(
        strategy,
        "calculate_max_drawdown",
        fake_calculate_max_drawdown,
    )
    monkeypatch.setattr(
        strategy,
        "calculate_annual_trade_counts",
        lambda completed_trades: {},
    )
    return call_records


def _gate_config(cold_start: str = "open") -> strategy.ExpectancyGateConfig:
    """Return the small-window gate used by lifecycle fixtures."""

    return strategy.ExpectancyGateConfig(
        window=2,
        baseline_mean=0.0,
        baseline_sigma=0.05,
        sigma_multiplier=1.0,
        cold_start=cold_start,
    )


def _run_fixture_simulation(
    monkeypatch: pytest.MonkeyPatch,
    artifacts: strategy.StrategyEvaluationArtifacts,
    expectancy_gate: strategy.ExpectancyGateConfig | None,
    *,
    margin_multiplier: float = 1.0,
) -> strategy.ComplexSimulationMetrics:
    """Run one fixture through the shared-slot simulation."""

    monkeypatch.setattr(
        strategy,
        "_generate_strategy_evaluation_artifacts",
        lambda *arguments, **keyword_arguments: artifacts,
    )
    definitions = {
        "all_trades": strategy.ComplexStrategySetDefinition(
            label="all_trades",
            buy_strategy_name="fixture",
            sell_strategy_name="fixture",
        )
    }
    return strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=10,
        multi_bucket_mode=True,
        margin_multiplier=margin_multiplier,
        expectancy_gate=expectancy_gate,
    )


def _entry_details(
    simulation_metrics: strategy.ComplexSimulationMetrics,
) -> list[strategy.TradeDetail]:
    """Return overall entry details ordered by entry date and symbol."""

    return sorted(
        (
            trade_detail
            for yearly_details in (
                simulation_metrics.overall_metrics.trade_details_by_year.values()
            )
            for trade_detail in yearly_details
            if trade_detail.action == "open"
        ),
        key=lambda trade_detail: (trade_detail.date, trade_detail.symbol),
    )


def test_expectancy_sensor_uses_exit_entry_and_stable_ordering() -> None:
    """The deque should consume ties by exit, entry, then stable CSV order."""

    gate_config = strategy.ExpectancyGateConfig(
        window=3,
        baseline_mean=0.0,
        baseline_sigma=0.01,
        sigma_multiplier=1.0,
    )
    sensor = strategy.ExpectancyGateSensor(gate_config)
    later_entry_trade = _build_trade(
        "2024-01-02", "2024-01-05", 0.10, "LATER"
    )[0]
    stable_second_trade = _build_trade(
        "2024-01-01", "2024-01-05", 0.20, "SECOND"
    )[0]
    stable_first_trade = _build_trade(
        "2024-01-01", "2024-01-05", 0.30, "FIRST"
    )[0]

    sensor.schedule_accepted_trade(1, later_entry_trade, 0)
    sensor.schedule_accepted_trade(2, stable_second_trade, 2)
    sensor.schedule_accepted_trade(3, stable_first_trade, 1)

    assert sensor.is_alarmed_before_entry(pandas.Timestamp("2024-01-05")) is False
    assert sensor.rolling_returns == ()
    assert sensor.is_alarmed_before_entry(pandas.Timestamp("2024-01-06")) is False
    assert sensor.rolling_returns == pytest.approx((0.30, 0.20, 0.10))
    assert sensor.rolling_mean == pytest.approx(0.20)
    assert [
        outcome.entry_date for outcome in sensor.consumed_outcomes
    ] == [
        pandas.Timestamp("2024-01-01"),
        pandas.Timestamp("2024-01-01"),
        pandas.Timestamp("2024-01-02"),
    ]
    assert all(
        outcome.exit_date == pandas.Timestamp("2024-01-05")
        for outcome in sensor.consumed_outcomes
    )


def test_expectancy_sensor_is_open_at_threshold_equality() -> None:
    """The alarm uses strict less-than, so equality must be open."""

    gate_config = strategy.ExpectancyGateConfig(
        window=2,
        baseline_mean=0.10,
        baseline_sigma=0.05,
        sigma_multiplier=1.0,
    )
    sensor = strategy.ExpectancyGateSensor(gate_config)
    first_trade = _build_trade(
        "2024-01-01", "2024-01-02", 0.05, "FIRST"
    )[0]
    second_trade = _build_trade(
        "2024-01-01", "2024-01-03", 0.05, "SECOND"
    )[0]
    sensor.schedule_accepted_trade(1, first_trade, 1)
    sensor.schedule_accepted_trade(2, second_trade, 2)

    assert gate_config.threshold == pytest.approx(0.05)
    assert sensor.is_alarmed_before_entry(pandas.Timestamp("2024-01-04")) is False


def test_expectancy_gate_closes_phantoms_and_reopens_from_phantom_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A losing window should close once and a phantom win should reopen it."""

    trades_with_details = [
        _build_trade("2024-01-01", "2024-01-02", -0.10, "LOSS_ONE"),
        _build_trade("2024-01-01", "2024-01-03", -0.10, "LOSS_TWO"),
        _build_trade("2024-01-04", "2024-01-05", 0.20, "PHANTOM_WIN"),
        _build_trade("2024-01-06", "2024-01-07", 0.10, "REOPENED"),
    ]
    artifacts = _build_artifacts(trades_with_details)
    _stub_money_simulations(monkeypatch)

    simulation_metrics = _run_fixture_simulation(
        monkeypatch,
        artifacts,
        _gate_config(),
    )

    entry_details_by_symbol = {
        entry_detail.symbol: entry_detail
        for entry_detail in _entry_details(simulation_metrics)
    }
    assert entry_details_by_symbol["LOSS_ONE"].expectancy_gated is False
    assert entry_details_by_symbol["LOSS_TWO"].expectancy_gated is False
    assert entry_details_by_symbol["PHANTOM_WIN"].expectancy_gated is True
    assert entry_details_by_symbol["PHANTOM_WIN"].phantom is False
    assert entry_details_by_symbol["REOPENED"].expectancy_gated is False
    assert simulation_metrics.overall_metrics.total_trades == 4
    assert simulation_metrics.expectancy_gated_trade_count == 1
    assert [
        asdict(episode)
        for episode in simulation_metrics.expectancy_gate_closed_episodes
    ] == [
        {
            "first_entry_date": pandas.Timestamp("2024-01-04"),
            "last_entry_date": pandas.Timestamp("2024-01-04"),
            "gated_trade_count": 1,
        }
    ]


def test_expectancy_sensor_uses_adaptive_adjusted_trade_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate should read the adaptive SL outcome rather than raw profit."""

    first_raw_trade, first_detail_pair = _build_trade(
        "2024-01-01", "2024-01-10", 0.20, "RAW_WIN_ONE"
    )
    second_raw_trade, second_detail_pair = _build_trade(
        "2024-01-01", "2024-01-10", 0.20, "RAW_WIN_TWO"
    )
    candidate_trade_with_details = _build_trade(
        "2024-01-04", "2024-01-08", 0.10, "CANDIDATE"
    )
    adaptive_stop_excursions = [
        (pandas.Timestamp("2024-01-02"), 0.01, -0.06, 0.0)
    ]
    first_raw_trade = strategy.replace(
        first_raw_trade,
        bar_excursions=adaptive_stop_excursions,
    )
    second_raw_trade = strategy.replace(
        second_raw_trade,
        bar_excursions=adaptive_stop_excursions,
    )
    artifacts = _build_artifacts(
        [
            (first_raw_trade, first_detail_pair),
            (second_raw_trade, second_detail_pair),
            candidate_trade_with_details,
        ]
    )
    _stub_money_simulations(monkeypatch)
    monkeypatch.setattr(
        strategy,
        "_generate_strategy_evaluation_artifacts",
        lambda *arguments, **keyword_arguments: artifacts,
    )
    definitions = {
        "all_trades": strategy.ComplexStrategySetDefinition(
            label="all_trades",
            buy_strategy_name="fixture",
            sell_strategy_name="fixture",
        )
    }
    gate_config = strategy.ExpectancyGateConfig(
        window=2,
        baseline_mean=0.01,
        baseline_sigma=0.05,
        sigma_multiplier=1.0,
    )

    simulation_metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=10,
        multi_bucket_mode=True,
        adaptive_tp_sl=strategy.AdaptiveTPSLConfig(
            min_sl=0.05,
            fixed_sl=0.05,
            override_min_hold_sl_only=True,
            min_hold_sl=1,
        ),
        expectancy_gate=gate_config,
    )

    entry_details_by_symbol = {
        entry_detail.symbol: entry_detail
        for entry_detail in _entry_details(simulation_metrics)
    }
    assert entry_details_by_symbol["CANDIDATE"].expectancy_gated is True
    adaptive_close_percentages = sorted(
        trade_detail.percentage_change
        for yearly_details in (
            simulation_metrics.overall_metrics.trade_details_by_year.values()
        )
        for trade_detail in yearly_details
        if trade_detail.action == "close"
        and trade_detail.symbol.startswith("RAW_WIN")
    )
    assert adaptive_close_percentages == pytest.approx([-0.05, -0.05])


@pytest.mark.parametrize(
    ("cold_start", "expected_first_two_flags"),
    [
        ("open", [False, False]),
        ("closed", [True, True]),
    ],
)
def test_expectancy_gate_cold_start_modes(
    monkeypatch: pytest.MonkeyPatch,
    cold_start: str,
    expected_first_two_flags: list[bool],
) -> None:
    """Cold-start mode should govern entries until the deque is full."""

    artifacts = _build_artifacts(
        [
            _build_trade("2024-01-01", "2024-01-02", 0.10, "FIRST"),
            _build_trade("2024-01-01", "2024-01-03", 0.10, "SECOND"),
            _build_trade("2024-01-04", "2024-01-05", 0.10, "WARM"),
        ]
    )
    _stub_money_simulations(monkeypatch)

    simulation_metrics = _run_fixture_simulation(
        monkeypatch,
        artifacts,
        _gate_config(cold_start),
    )

    flags_by_symbol = {
        entry_detail.symbol: entry_detail.expectancy_gated
        for entry_detail in _entry_details(simulation_metrics)
    }
    assert [flags_by_symbol["FIRST"], flags_by_symbol["SECOND"]] == (
        expected_first_two_flags
    )
    assert flags_by_symbol["WARM"] is False


def test_expectancy_gated_ids_zero_all_money_sims_without_touching_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only expectancy-gated slot trades should receive the zero override."""

    trades_with_details = [
        _build_trade("2024-01-01", "2024-01-02", -0.10, "LOSS_ONE"),
        _build_trade("2024-01-01", "2024-01-03", -0.10, "LOSS_TWO"),
        _build_trade("2024-01-04", "2024-01-05", 0.20, "GATED"),
        _build_trade("2024-01-06", "2024-01-07", 0.10, "NORMAL"),
    ]
    completed_trades = [
        completed_trade
        for completed_trade, _detail_pair in trades_with_details
    ]
    artifacts = _build_artifacts(trades_with_details)
    call_records = _stub_money_simulations(monkeypatch)

    _run_fixture_simulation(
        monkeypatch,
        artifacts,
        _gate_config(),
        margin_multiplier=1.5,
    )

    gated_trade_identifier = id(completed_trades[2])
    non_gated_trade_identifiers = {
        id(completed_trades[0]),
        id(completed_trades[1]),
        id(completed_trades[3]),
    }
    for money_simulation_calls in call_records.values():
        assert money_simulation_calls
        for _passed_trades, keyword_arguments in money_simulation_calls:
            assert keyword_arguments["margin_multiplier"] == 1.5
            assert set(keyword_arguments["margin_overrides"].values()) == {
                0.0
            }
            gated_trade_ids = keyword_arguments["gated_trade_ids"]
            assert gated_trade_ids == {gated_trade_identifier}
            assert non_gated_trade_identifiers.isdisjoint(gated_trade_ids)


def test_zero_weight_expectancy_trade_has_no_money_simulation_effect() -> None:
    """All three money simulators should match a normal-trade-only run."""

    expectancy_gated_trade = _build_trade(
        "2024-01-01", "2024-01-03", -0.50, "GATED"
    )[0]
    normal_trade = _build_trade(
        "2024-01-05", "2024-01-08", 0.10, "NORMAL"
    )[0]
    all_trades = [expectancy_gated_trade, normal_trade]
    zero_margin_overrides = {"2024-01": 0.0}
    expectancy_gated_trade_ids = {id(expectancy_gated_trade)}
    symbol_lookup = {
        expectancy_gated_trade: "GATED",
        normal_trade: "NORMAL",
    }
    closing_prices = {
        "GATED": pandas.Series(
            [100.0, 50.0],
            index=[
                expectancy_gated_trade.entry_date,
                expectancy_gated_trade.exit_date,
            ],
        ),
        "NORMAL": pandas.Series(
            [100.0, 110.0],
            index=[normal_trade.entry_date, normal_trade.exit_date],
        ),
    }
    common_arguments = {
        "starting_cash": 10_000.0,
        "maximum_position_count": 2,
        "margin_multiplier": 1.5,
        "margin_overrides": zero_margin_overrides,
        "gated_trade_ids": expectancy_gated_trade_ids,
    }

    gated_balance = simulator.simulate_portfolio_balance(
        all_trades,
        **common_arguments,
    )
    normal_only_balance = simulator.simulate_portfolio_balance(
        [normal_trade],
        starting_cash=10_000.0,
        maximum_position_count=2,
        margin_multiplier=1.5,
    )
    gated_annual_returns = simulator.calculate_annual_returns(
        all_trades,
        simulation_start=pandas.Timestamp("2024-01-01"),
        trade_symbol_lookup=symbol_lookup,
        closing_price_series_by_symbol=closing_prices,
        **common_arguments,
    )
    normal_only_annual_returns = simulator.calculate_annual_returns(
        [normal_trade],
        starting_cash=10_000.0,
        maximum_position_count=2,
        simulation_start=pandas.Timestamp("2024-01-01"),
        margin_multiplier=1.5,
        trade_symbol_lookup={normal_trade: "NORMAL"},
        closing_price_series_by_symbol={"NORMAL": closing_prices["NORMAL"]},
    )
    gated_maximum_drawdown = simulator.calculate_max_drawdown(
        all_trades,
        trade_symbol_lookup=symbol_lookup,
        closing_price_series_by_symbol=closing_prices,
        **common_arguments,
    )
    normal_only_maximum_drawdown = simulator.calculate_max_drawdown(
        [normal_trade],
        starting_cash=10_000.0,
        maximum_position_count=2,
        margin_multiplier=1.5,
        trade_symbol_lookup={normal_trade: "NORMAL"},
        closing_price_series_by_symbol={"NORMAL": closing_prices["NORMAL"]},
    )

    assert gated_balance == pytest.approx(normal_only_balance)
    assert gated_annual_returns == pytest.approx(normal_only_annual_returns)
    assert gated_maximum_drawdown == pytest.approx(
        normal_only_maximum_drawdown
    )


def test_expectancy_and_wr_phantom_flags_share_one_zero_capital_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WR and expectancy phantoms should be unioned without widening scope."""

    trades_with_details = [
        _build_trade("2024-01-01", "2024-01-02", -0.10, "WR_PHANTOM"),
        _build_trade("2024-01-01", "2024-01-03", -0.10, "LOSS_TWO"),
        _build_trade("2024-01-04", "2024-01-05", 0.20, "EXPECTANCY"),
        _build_trade("2024-01-06", "2024-01-07", 0.10, "NORMAL"),
    ]
    trades_with_details[0][1][0].phantom = True
    completed_trades = [
        completed_trade
        for completed_trade, _detail_pair in trades_with_details
    ]
    artifacts = _build_artifacts(trades_with_details)
    call_records = _stub_money_simulations(monkeypatch)

    simulation_metrics = _run_fixture_simulation(
        monkeypatch,
        artifacts,
        _gate_config(),
    )

    expected_zero_capital_identifiers = {
        id(completed_trades[0]),
        id(completed_trades[2]),
    }
    normal_trade_identifiers = {
        id(completed_trades[1]),
        id(completed_trades[3]),
    }
    for money_simulation_calls in call_records.values():
        for _passed_trades, keyword_arguments in money_simulation_calls:
            assert keyword_arguments["gated_trade_ids"] == (
                expected_zero_capital_identifiers
            )
            assert normal_trade_identifiers.isdisjoint(
                keyword_arguments["gated_trade_ids"]
            )
    entry_details_by_symbol = {
        entry_detail.symbol: entry_detail
        for entry_detail in _entry_details(simulation_metrics)
    }
    assert entry_details_by_symbol["WR_PHANTOM"].phantom is True
    assert entry_details_by_symbol["WR_PHANTOM"].expectancy_gated is False
    assert entry_details_by_symbol["EXPECTANCY"].phantom is False
    assert entry_details_by_symbol["EXPECTANCY"].expectancy_gated is True


def test_expectancy_gate_absent_and_disabled_are_inert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing and explicitly disabled blocks should produce equal metrics."""

    trades_with_details = [
        _build_trade("2024-01-01", "2024-01-02", -0.10, "LOSS"),
        _build_trade("2024-01-03", "2024-01-04", 0.10, "WIN"),
    ]
    _stub_money_simulations(monkeypatch)
    absent_metrics = _run_fixture_simulation(
        monkeypatch,
        _build_artifacts(trades_with_details),
        None,
    )
    disabled_gate = strategy.parse_expectancy_gate_config(
        {"enabled": False}
    )
    disabled_metrics = _run_fixture_simulation(
        monkeypatch,
        _build_artifacts(
            [
                _build_trade("2024-01-01", "2024-01-02", -0.10, "LOSS"),
                _build_trade("2024-01-03", "2024-01-04", 0.10, "WIN"),
            ]
        ),
        disabled_gate,
    )

    assert disabled_gate is None
    assert absent_metrics == disabled_metrics


def test_statistics_range_uses_warmup_trades_for_expectancy_sensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-statistics trades should warm the sensor but stay out of metrics."""

    artifacts = _build_artifacts(
        [
            _build_trade("2023-01-01", "2023-01-02", -0.10, "WARMUP_ONE"),
            _build_trade("2023-01-03", "2023-01-04", -0.10, "WARMUP_TWO"),
            _build_trade("2024-01-02", "2024-01-03", 0.10, "STATISTICS"),
        ]
    )
    _stub_money_simulations(monkeypatch)
    monkeypatch.setattr(
        strategy,
        "_generate_strategy_evaluation_artifacts",
        lambda *arguments, **keyword_arguments: artifacts,
    )
    definitions = {
        "all_trades": strategy.ComplexStrategySetDefinition(
            label="all_trades",
            buy_strategy_name="fixture",
            sell_strategy_name="fixture",
        )
    }

    simulation_metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=10,
        multi_bucket_mode=True,
        start_date=pandas.Timestamp("2023-01-01"),
        end_date=pandas.Timestamp("2024-12-31"),
        statistics_start_date=pandas.Timestamp("2024-01-01"),
        statistics_end_date=pandas.Timestamp("2024-12-31"),
        expectancy_gate=_gate_config(),
    )

    statistics_entries = _entry_details(simulation_metrics)
    assert [entry_detail.symbol for entry_detail in statistics_entries] == [
        "STATISTICS"
    ]
    assert statistics_entries[0].expectancy_gated is True


def test_multi_bucket_csv_reports_expectancy_gate_separately(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Enabled multi-bucket CSV output should include the dedicated flag."""

    data_directory = tmp_path / "prices"
    data_directory.mkdir()
    config_path = tmp_path / "expectancy_gate.json"
    config_path.write_text(
        json.dumps(
            {
                "max_position_count": 10,
                "starting_cash": 1000,
                "start_date": "2024-01-01",
                "data_source": "test",
                "expectancy_gate": {
                    "enabled": True,
                    "window": 2,
                    "baseline_mean": 0.0,
                    "baseline_sigma": 0.05,
                    "sigma_multiplier": 1.0,
                    "cold_start": "open",
                },
                "buckets": [
                    {
                        "label": "all_trades",
                        "strategy_id": "fixture",
                        "dollar_volume_filter": "dollar_volume>1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    artifacts = _build_artifacts(
        [
            _build_trade("2024-01-01", "2024-01-02", -0.10, "LOSS_ONE"),
            _build_trade("2024-01-01", "2024-01-03", -0.10, "LOSS_TWO"),
            _build_trade("2024-01-04", "2024-01-05", 0.10, "GATED"),
        ]
    )
    monkeypatch.setattr(manage, "DATA_SOURCE_PATHS", {"test": data_directory})
    monkeypatch.setattr(
        manage,
        "load_strategy_set_mapping",
        lambda: {"fixture": ("fixture", "fixture")},
    )
    monkeypatch.setattr(manage, "load_strategy_entry_filters", lambda: {})
    monkeypatch.setattr(
        strategy,
        "_generate_strategy_evaluation_artifacts",
        lambda *arguments, **keyword_arguments: artifacts,
    )
    _stub_money_simulations(monkeypatch)
    monkeypatch.chdir(tmp_path)
    output_buffer = io.StringIO()

    shell = manage.StockShell(stdout=output_buffer)
    shell.onecmd(f"multi_bucket_simulation {config_path}")

    output_files = list(
        (tmp_path / "logs" / "multi_bucket_simulation_result").glob("*.csv")
    )
    assert len(output_files) == 1
    output_frame = pandas.read_csv(output_files[0])
    assert output_frame.columns.get_loc("expectancy_gated") == (
        output_frame.columns.get_loc("phantom") + 1
    )
    gated_row = output_frame.loc[output_frame["symbol"] == "GATED"].iloc[0]
    assert bool(gated_row["expectancy_gated"]) is True
    assert bool(gated_row["phantom"]) is False
    output_text = output_buffer.getvalue()
    assert "Expectancy gate: window=2" in output_text
    assert "threshold=-0.050000" in output_text
    assert "closed_episodes=1 (2024-01-04 to 2024-01-04)" in output_text


def test_multi_bucket_exports_ft_all_signal_counterfactual_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The research export should preserve both entry and causal exit dates."""

    data_directory = tmp_path / "prices"
    data_directory.mkdir()
    config_path = tmp_path / "ft_expectancy_gate.json"
    config_path.write_text(
        json.dumps(
            {
                "max_position_count": 10,
                "starting_cash": 1000,
                "start_date": "2024-01-01",
                "data_source": "test",
                "buckets": [
                    {
                        "label": "fish_tail_production",
                        "strategy_id": "fixture",
                        "dollar_volume_filter": "dollar_volume>1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    artifacts = _build_artifacts(
        [
            _build_trade("2024-01-01", "2024-01-03", -0.10, "LOSS"),
            _build_trade("2024-01-02", "2024-01-04", 0.20, "WIN"),
        ]
    )
    monkeypatch.setattr(manage, "DATA_SOURCE_PATHS", {"test": data_directory})
    monkeypatch.setattr(
        manage,
        "load_strategy_set_mapping",
        lambda: {"fixture": ("fixture", "fixture")},
    )
    monkeypatch.setattr(manage, "load_strategy_entry_filters", lambda: {})
    monkeypatch.setattr(
        strategy,
        "_generate_strategy_evaluation_artifacts",
        lambda *arguments, **keyword_arguments: artifacts,
    )
    _stub_money_simulations(monkeypatch)
    monkeypatch.chdir(tmp_path)
    export_path = tmp_path / "research" / "ft_signal_outcomes.csv"
    output_buffer = io.StringIO()

    shell = manage.StockShell(stdout=output_buffer)
    shell.onecmd(
        f"multi_bucket_simulation {config_path} "
        f"--export-ft-signal-outcomes {export_path}"
    )

    exported_frame = pandas.read_csv(export_path)
    assert exported_frame[
        [
            "entry_date",
            "exit_date",
            "percentage_change",
            "exit_reason",
            "holding_bars",
        ]
    ].to_dict(orient="records") == [
        {
            "entry_date": "2024-01-01",
            "exit_date": "2024-01-03",
            "percentage_change": pytest.approx(-0.10),
            "exit_reason": "signal",
            "holding_bars": 2,
        },
        {
            "entry_date": "2024-01-02",
            "exit_date": "2024-01-04",
            "percentage_change": pytest.approx(0.20),
            "exit_reason": "signal",
            "holding_bars": 2,
        },
    ]
    assert exported_frame["max_favorable_excursion_pct"].isna().all()
    assert exported_frame["max_adverse_excursion_pct"].isna().all()
    assert "FT all-signal counterfactual outcomes saved" in (
        output_buffer.getvalue()
    )


@pytest.mark.parametrize(
    ("invalid_override", "expected_message"),
    [
        ({"window": 1}, "expectancy_gate.window must be an integer >= 2"),
        ({"window": 2.5}, "expectancy_gate.window must be an integer >= 2"),
        ({"baseline_sigma": 0}, "expectancy_gate.baseline_sigma must be > 0"),
        ({"sigma_multiplier": 0}, "expectancy_gate.sigma_multiplier must be > 0"),
        ({"baseline_mean": float("inf")}, "expectancy_gate.baseline_mean must be finite"),
        ({"cold_start": "warm"}, "expectancy_gate.cold_start must be 'open' or 'closed'"),
        ({"unexpected": 1}, "expectancy_gate contains unknown key(s): unexpected"),
    ],
)
def test_multi_bucket_expectancy_config_validation_aborts_run(
    tmp_path: Path,
    invalid_override: dict[str, object],
    expected_message: str,
) -> None:
    """Invalid enabled blocks should abort the command with a clear error."""

    expectancy_gate_block: dict[str, object] = {
        "enabled": True,
        "window": 20,
        "baseline_mean": 0.0087,
        "baseline_sigma": 0.0130,
        "sigma_multiplier": 3.0,
        "cold_start": "open",
    }
    expectancy_gate_block.update(invalid_override)
    config_path = tmp_path / "invalid_expectancy_gate.json"
    config_path.write_text(
        json.dumps(
            {
                "max_position_count": 1,
                "expectancy_gate": expectancy_gate_block,
                "buckets": [{}],
            }
        ),
        encoding="utf-8",
    )
    output_buffer = io.StringIO()

    shell = manage.StockShell(stdout=output_buffer)
    shell.onecmd(f"multi_bucket_simulation {config_path}")

    assert expected_message in output_buffer.getvalue()


def test_parse_ft_family_expectancy_gate_config_validation() -> None:
    parse = strategy.parse_ft_family_expectancy_gate_config
    assert parse(None) is None
    assert parse({"enabled": False, "window": 2}) is None
    gate_config = parse(
        {
            "enabled": True,
            "window": 12,
            "baseline_mean": 0.0043,
            "baseline_sigma": 0.011,
            "sigma_multiplier": 1.5,
        }
    )
    assert gate_config is not None
    assert gate_config.sensor_buckets == (
        "fish_tail_production",
        "fish_tail_squeeze",
    )
    assert gate_config.decision_mode == "rolling_trades"
    assert gate_config.threshold == pytest.approx(0.0043 - 1.5 * 0.011)
    period_gate_config = parse(
        {
            "enabled": True,
            "window": 12,
            "baseline_mean": 0.0043,
            "baseline_sigma": 0.011,
            "sigma_multiplier": 1.5,
            "decision_mode": "previous_calendar_period",
            "period_months": 6,
            "period_decision_threshold": 0.0,
            "period_minimum_samples": 20,
        }
    )
    assert period_gate_config is not None
    assert period_gate_config.period_months == 6
    assert period_gate_config.period_minimum_samples == 20
    assert period_gate_config.threshold == 0.0
    with pytest.raises(ValueError, match="unknown key"):
        parse({"window": 12, "baseline_mean": 0.0, "baseline_sigma": 0.01,
               "sigma_multiplier": 1.0, "extra": 1})
    with pytest.raises(ValueError, match="baseline_sigma"):
        parse({"window": 12, "baseline_mean": 0.0, "baseline_sigma": 0,
               "sigma_multiplier": 1.0})
    with pytest.raises(ValueError, match="is required"):
        parse({"window": 12, "baseline_mean": 0.0, "baseline_sigma": 0.01})
    with pytest.raises(ValueError, match="non-empty list"):
        parse({"window": 12, "baseline_mean": 0.0, "baseline_sigma": 0.01,
               "sigma_multiplier": 1.0, "gated_buckets": []})
    with pytest.raises(ValueError, match="positive integer divisor of 12"):
        parse({"window": 12, "baseline_mean": 0.0, "baseline_sigma": 0.01,
               "sigma_multiplier": 1.0,
               "decision_mode": "previous_calendar_period",
               "period_months": 5, "period_decision_threshold": 0.0})
    with pytest.raises(ValueError, match="period fields require"):
        parse({"window": 12, "baseline_mean": 0.0, "baseline_sigma": 0.01,
               "sigma_multiplier": 1.0, "period_months": 6})


def _ft_gate_config_for_fixture() -> strategy.FTFamilyExpectancyGateConfig:
    """window 2, threshold -0.05, scoped to the fixture's only set."""

    return strategy.FTFamilyExpectancyGateConfig(
        window=2,
        baseline_mean=0.0,
        baseline_sigma=0.05,
        sigma_multiplier=1.0,
        sensor_buckets=("all_trades",),
        gated_buckets=("all_trades",),
    )


def test_previous_calendar_period_gate_freezes_decision_and_exit_sample(
) -> None:
    """The prior half's exit returns must set one fixed next-half decision."""

    gate_config = strategy.FTFamilyExpectancyGateConfig(
        window=2,
        baseline_mean=0.0,
        baseline_sigma=0.05,
        sigma_multiplier=1.0,
        decision_mode="previous_calendar_period",
        period_months=6,
        period_decision_threshold=0.0,
        period_minimum_samples=2,
        sensor_buckets=("all_trades",),
        gated_buckets=("all_trades",),
    )
    gate = strategy.PreviousCalendarPeriodExpectancyGate(gate_config)
    outcomes = [
        _build_trade("2024-01-02", "2024-02-01", -0.10, "H1_LOSS_ONE")[0],
        _build_trade("2024-06-01", "2024-06-30", -0.10, "H1_LOSS_TWO")[0],
        _build_trade("2024-06-02", "2024-07-01", 0.50, "BOUNDARY_WIN")[0],
        _build_trade("2024-07-02", "2024-08-01", 0.10, "H2_WIN_ONE")[0],
        _build_trade("2024-08-02", "2024-09-01", 0.10, "H2_WIN_TWO")[0],
    ]
    for stable_outcome_order, completed_trade in enumerate(outcomes):
        gate.sensor.schedule_outcome(
            id(completed_trade),
            completed_trade,
            stable_outcome_order,
        )

    assert gate.evaluate_accepted_entry(pandas.Timestamp("2024-07-01")) is True
    assert gate.evaluate_accepted_entry(pandas.Timestamp("2024-12-01")) is True
    assert gate.evaluate_accepted_entry(pandas.Timestamp("2025-01-01")) is False
    assert gate.expectancy_gated_trade_count == 2


def test_previous_calendar_period_gate_stays_open_without_minimum_sample(
) -> None:
    """A sparse prior period must not close the next period."""

    gate_config = strategy.FTFamilyExpectancyGateConfig(
        window=2,
        baseline_mean=0.0,
        baseline_sigma=0.05,
        sigma_multiplier=1.0,
        decision_mode="previous_calendar_period",
        period_months=6,
        period_decision_threshold=0.0,
        period_minimum_samples=2,
        sensor_buckets=("all_trades",),
        gated_buckets=("all_trades",),
    )
    gate = strategy.PreviousCalendarPeriodExpectancyGate(gate_config)
    completed_trade = _build_trade(
        "2024-01-02",
        "2024-06-30",
        -0.10,
        "ONLY_SAMPLE",
    )[0]
    gate.sensor.schedule_outcome(id(completed_trade), completed_trade, 0)

    assert gate.evaluate_accepted_entry(pandas.Timestamp("2024-07-01")) is False


def test_ft_expectancy_gate_phantoms_gated_entry_with_zero_capital(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two ft losses below the band phantom the next gated entry only."""

    trades_with_details = [
        _build_trade("2024-01-01", "2024-01-02", -0.10, "LOSS_ONE"),
        _build_trade("2024-01-01", "2024-01-03", -0.10, "LOSS_TWO"),
        _build_trade("2024-01-08", "2024-01-09", 0.10, "GATED"),
        _build_trade("2024-01-16", "2024-01-17", 0.10, "REOPENED"),
    ]
    completed_trades = [
        completed_trade
        for completed_trade, _detail_pair in trades_with_details
    ]
    artifacts = _build_artifacts(trades_with_details)
    call_records = _stub_money_simulations(monkeypatch)
    monkeypatch.setattr(
        strategy,
        "_generate_strategy_evaluation_artifacts",
        lambda *arguments, **keyword_arguments: artifacts,
    )
    definitions = {
        "all_trades": strategy.ComplexStrategySetDefinition(
            label="all_trades",
            buy_strategy_name="fixture",
            sell_strategy_name="fixture",
        )
    }
    simulation_metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=10,
        multi_bucket_mode=True,
        ft_expectancy_gate=_ft_gate_config_for_fixture(),
    )

    entry_details_by_symbol = {
        entry_detail.symbol: entry_detail
        for entry_detail in _entry_details(simulation_metrics)
    }
    # Two -10% closes drag the 2-trade mean to -0.10 < -0.05: the next
    # gated entry phantoms. Its own +10% close then lifts the mean back
    # to 0.0, so the following entry deploys capital normally.
    assert entry_details_by_symbol["LOSS_ONE"].ft_expectancy_gated is False
    assert entry_details_by_symbol["LOSS_TWO"].ft_expectancy_gated is False
    assert entry_details_by_symbol["GATED"].ft_expectancy_gated is True
    assert entry_details_by_symbol["REOPENED"].ft_expectancy_gated is False
    assert simulation_metrics.ft_expectancy_gated_trade_count == 1
    assert len(simulation_metrics.ft_expectancy_gate_closed_episodes) == 1
    expected_zero_capital_identifiers = {id(completed_trades[2])}
    for money_simulation_calls in call_records.values():
        for _passed_trades, keyword_arguments in money_simulation_calls:
            assert keyword_arguments["gated_trade_ids"] == (
                expected_zero_capital_identifiers
            )


def test_ft_previous_period_gate_phantoms_next_half_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulation dispatch must apply H1's mean throughout H2."""

    trades_with_details = [
        _build_trade("2024-01-02", "2024-02-01", -0.10, "H1_LOSS_ONE"),
        _build_trade("2024-05-01", "2024-06-01", -0.10, "H1_LOSS_TWO"),
        _build_trade("2024-07-02", "2024-08-01", 0.10, "H2_GATED"),
    ]
    artifacts = _build_artifacts(trades_with_details)
    monkeypatch.setattr(
        strategy,
        "_generate_strategy_evaluation_artifacts",
        lambda *arguments, **keyword_arguments: artifacts,
    )
    _stub_money_simulations(monkeypatch)
    definitions = {
        "all_trades": strategy.ComplexStrategySetDefinition(
            label="all_trades",
            buy_strategy_name="fixture",
            sell_strategy_name="fixture",
        )
    }
    gate_config = strategy.FTFamilyExpectancyGateConfig(
        window=2,
        baseline_mean=0.0,
        baseline_sigma=0.05,
        sigma_multiplier=1.0,
        decision_mode="previous_calendar_period",
        period_months=6,
        period_decision_threshold=0.0,
        period_minimum_samples=2,
        sensor_buckets=("all_trades",),
        gated_buckets=("all_trades",),
    )

    simulation_metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=10,
        multi_bucket_mode=True,
        ft_expectancy_gate=gate_config,
    )

    entry_details_by_symbol = {
        entry_detail.symbol: entry_detail
        for entry_detail in _entry_details(simulation_metrics)
    }
    assert entry_details_by_symbol["H2_GATED"].ft_expectancy_gated is True
    assert simulation_metrics.ft_expectancy_gated_trade_count == 1


@pytest.mark.parametrize(
    "adaptive_tp_sl_config",
    [None, strategy.AdaptiveTPSLConfig()],
    ids=("non_adaptive", "adaptive"),
)
def test_ft_expectancy_sensor_includes_signals_blocked_by_fh_allocation(
    monkeypatch: pytest.MonkeyPatch,
    adaptive_tp_sl_config: strategy.AdaptiveTPSLConfig | None,
) -> None:
    """FH slot occupancy must not remove FT signals from the sensor."""

    fish_head_artifacts = _build_artifacts(
        [
            _build_trade(
                "2024-01-01",
                "2024-01-10",
                0.0,
                "FH_BLOCKER",
            )
        ]
    )
    fish_tail_artifacts = _build_artifacts(
        [
            _build_trade(
                "2024-01-02",
                "2024-01-03",
                -0.10,
                "FT_BLOCKED_LOSS_ONE",
            ),
            _build_trade(
                "2024-01-04",
                "2024-01-05",
                -0.10,
                "FT_BLOCKED_LOSS_TWO",
            ),
            _build_trade(
                "2024-01-11",
                "2024-01-12",
                0.10,
                "FT_CANDIDATE",
            ),
        ]
    )
    artifacts_by_buy_strategy = {
        "fish_head_fixture": fish_head_artifacts,
        "fish_tail_fixture": fish_tail_artifacts,
    }

    def fake_generate_strategy_evaluation_artifacts(
        data_directory: Path,
        buy_strategy_name: str,
        *unused_arguments: object,
        **unused_keyword_arguments: object,
    ) -> strategy.StrategyEvaluationArtifacts:
        """Return the fixture selected by its configured buy strategy."""

        del data_directory, unused_arguments, unused_keyword_arguments
        return artifacts_by_buy_strategy[buy_strategy_name]

    monkeypatch.setattr(
        strategy,
        "_generate_strategy_evaluation_artifacts",
        fake_generate_strategy_evaluation_artifacts,
    )
    _stub_money_simulations(monkeypatch)
    definitions = {
        "fish_head": strategy.ComplexStrategySetDefinition(
            label="fish_head",
            buy_strategy_name="fish_head_fixture",
            sell_strategy_name="fixture",
            entry_priority=1,
        ),
        "fish_tail": strategy.ComplexStrategySetDefinition(
            label="fish_tail",
            buy_strategy_name="fish_tail_fixture",
            sell_strategy_name="fixture",
            entry_priority=2,
        ),
    }
    gate_config = strategy.FTFamilyExpectancyGateConfig(
        window=2,
        baseline_mean=0.0,
        baseline_sigma=0.05,
        sigma_multiplier=1.0,
        sensor_buckets=("fish_tail",),
        gated_buckets=("fish_tail",),
    )

    simulation_metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=1,
        multi_bucket_mode=True,
        adaptive_tp_sl=adaptive_tp_sl_config,
        ft_expectancy_gate=gate_config,
    )

    entry_details_by_symbol = {
        entry_detail.symbol: entry_detail
        for entry_detail in _entry_details(simulation_metrics)
    }
    assert set(entry_details_by_symbol) == {"FH_BLOCKER", "FT_CANDIDATE"}
    assert entry_details_by_symbol["FT_CANDIDATE"].ft_expectancy_gated is True
    assert simulation_metrics.ft_expectancy_gated_trade_count == 1
    assert [
        outcome.entry_date
        for outcome in simulation_metrics.ft_expectancy_sensor_outcomes
    ] == [
        pandas.Timestamp("2024-01-02"),
        pandas.Timestamp("2024-01-04"),
        pandas.Timestamp("2024-01-11"),
    ]


def test_ft_expectancy_gate_rejects_unknown_bucket_label() -> None:
    definitions = {
        "all_trades": strategy.ComplexStrategySetDefinition(
            label="all_trades",
            buy_strategy_name="fixture",
            sell_strategy_name="fixture",
        )
    }
    bad_gate = strategy.FTFamilyExpectancyGateConfig(
        window=2,
        baseline_mean=0.0,
        baseline_sigma=0.05,
        sigma_multiplier=1.0,
        sensor_buckets=("missing_bucket",),
        gated_buckets=("all_trades",),
    )
    with pytest.raises(ValueError, match="unknown bucket label"):
        strategy.run_complex_simulation(
            Path("/tmp"),
            definitions,
            maximum_position_count=10,
            multi_bucket_mode=True,
            ft_expectancy_gate=bad_gate,
        )
