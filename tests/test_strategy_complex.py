"""Tests for complex simulation shared position management."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas
import pytest

from stock_indicator import strategy


def _build_trade(
    entry_date: str,
    exit_date: str,
    *,
    entry_price: float = 10.0,
    exit_price: float = 11.0,
    profit: float = 1.0,
    symbol: str = "AAA",
    near_price_volume_ratio: float | None = None,
    above_price_volume_ratio: float | None = None,
) -> tuple[strategy.Trade, tuple[strategy.TradeDetail, strategy.TradeDetail]]:
    """Create a trade and associated detail records for testing."""

    entry_timestamp = pandas.Timestamp(entry_date)
    exit_timestamp = pandas.Timestamp(exit_date)
    holding_period = (exit_timestamp - entry_timestamp).days
    trade = strategy.Trade(
        entry_date=entry_timestamp,
        exit_date=exit_timestamp,
        entry_price=entry_price,
        exit_price=exit_price,
        profit=profit,
        holding_period=holding_period,
        exit_reason="signal",
    )
    entry_detail = strategy.TradeDetail(
        date=entry_timestamp,
        symbol=symbol,
        action="open",
        price=entry_price,
        simple_moving_average_dollar_volume=0.0,
        total_simple_moving_average_dollar_volume=0.0,
        simple_moving_average_dollar_volume_ratio=0.0,
        near_price_volume_ratio=near_price_volume_ratio,
        above_price_volume_ratio=above_price_volume_ratio,
    )
    exit_detail = strategy.TradeDetail(
        date=exit_timestamp,
        symbol=symbol,
        action="close",
        price=exit_price,
        simple_moving_average_dollar_volume=0.0,
        total_simple_moving_average_dollar_volume=0.0,
        simple_moving_average_dollar_volume_ratio=0.0,
        result="win",
        percentage_change=profit / entry_price,
    )
    return trade, (entry_detail, exit_detail)


def _build_artifacts(
    trades_with_details: list[tuple[strategy.Trade, tuple[strategy.TradeDetail, strategy.TradeDetail]]],
) -> strategy.StrategyEvaluationArtifacts:
    """Create evaluation artifacts from prepared trades."""

    trades = [trade for trade, _ in trades_with_details]
    trade_symbol_lookup = {trade: detail_pair[0].symbol for trade, detail_pair in trades_with_details}
    closing_price_series_by_symbol = {
        detail_pair[0].symbol: pandas.Series(
            [detail_pair[0].price, detail_pair[1].price],
            index=[detail_pair[0].date, detail_pair[1].date],
        )
        for _, detail_pair in trades_with_details
    }
    trade_detail_pairs = {trade: detail_pair for trade, detail_pair in trades_with_details}
    simulation_results = [
        strategy.SimulationResult(
            trades=trades,
            total_profit=sum(current_trade.profit for current_trade in trades),
        )
    ]
    earliest_entry = min((trade.entry_date for trade in trades), default=None)
    return strategy.StrategyEvaluationArtifacts(
        trades=trades,
        simulation_results=simulation_results,
        trade_symbol_lookup=trade_symbol_lookup,
        closing_price_series_by_symbol=closing_price_series_by_symbol,
        trade_detail_pairs=trade_detail_pairs,
        simulation_start_date=earliest_entry,
    )


def _stub_metrics_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, list[int]]:
    """Replace expensive metric helpers with deterministic stubs."""

    call_records: dict[str, list[int]] = {
        "simulate_portfolio_balance": [],
        "calculate_max_drawdown": [],
    }

    monkeypatch.setattr(
        strategy, "calculate_annual_returns", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        strategy, "calculate_annual_trade_counts", lambda trades: {}
    )

    def fake_simulate_portfolio_balance(
        trades: list[strategy.Trade],
        starting_cash: float,
        maximum_position_count: int,
        *args: object,
        **kwargs: object,
    ) -> float:
        call_records["simulate_portfolio_balance"].append(maximum_position_count)
        return float(starting_cash)

    def fake_calculate_max_drawdown(
        trades: list[strategy.Trade],
        starting_cash: float,
        maximum_position_count: int,
        *args: object,
        **kwargs: object,
    ) -> float:
        call_records["calculate_max_drawdown"].append(maximum_position_count)
        return 0.0

    monkeypatch.setattr(
        strategy, "simulate_portfolio_balance", fake_simulate_portfolio_balance
    )
    monkeypatch.setattr(strategy, "calculate_max_drawdown", fake_calculate_max_drawdown)

    return call_records


def test_resolve_trade_decision_dates_non_pending() -> None:
    """Non-pending mode: signal_date is one trading bar before fill,
    confirmation_date coincides with signal_date (no separate B-layer
    bar). Regression for the off-by-one bug where signal_date was
    pulled to entry_date - 2 regardless of mode."""

    run_index = pandas.bdate_range("2024-01-02", periods=10)
    entry_date = run_index[5]
    signal_date, confirmation_date = strategy._resolve_trade_decision_dates(
        run_index, entry_date, use_confirmation_angle=False
    )
    assert signal_date == run_index[4]
    assert confirmation_date == run_index[4]


def test_resolve_trade_decision_dates_pending() -> None:
    """Pending mode: signal_date is two trading bars before fill (T),
    confirmation_date is one bar before fill (T+1)."""

    run_index = pandas.bdate_range("2024-01-02", periods=10)
    entry_date = run_index[5]
    signal_date, confirmation_date = strategy._resolve_trade_decision_dates(
        run_index, entry_date, use_confirmation_angle=True
    )
    assert signal_date == run_index[3]
    assert confirmation_date == run_index[4]


def test_resolve_trade_decision_dates_edge_at_start() -> None:
    """At the start of the run frame both modes degrade gracefully:
    fall back to whatever earlier bar exists, never overshoot
    backwards. entry_date itself is the worst-case fallback."""

    run_index = pandas.bdate_range("2024-01-02", periods=10)
    # Position 0 — no prior bar in either mode.
    signal_date, confirmation_date = strategy._resolve_trade_decision_dates(
        run_index, run_index[0], use_confirmation_angle=False
    )
    assert signal_date == run_index[0]
    assert confirmation_date == run_index[0]
    # Position 1 in pending mode — only one bar back available.
    signal_date, confirmation_date = strategy._resolve_trade_decision_dates(
        run_index, run_index[1], use_confirmation_angle=True
    )
    assert signal_date == run_index[0]
    assert confirmation_date == run_index[0]
    # Position 1 in non-pending mode — entry_date - 1 is well-defined.
    signal_date, confirmation_date = strategy._resolve_trade_decision_dates(
        run_index, run_index[1], use_confirmation_angle=False
    )
    assert signal_date == run_index[0]
    assert confirmation_date == run_index[0]


def test_resolve_trade_decision_dates_pre_cross_lookback_non_pending() -> None:
    """``pre_cross_signal_lookback`` shifts non-pending signal_date back
    one ADDITIONAL bar (lands on the bar BEFORE the cross detection bar).
    confirmation_date stays put — only the A-layer read shifts.

    This is the deliberate knob fish_head_vacuum_turn uses: the cross
    bar already includes the first reaction-up tick, so its slope_60 /
    near_delta no longer reflect the extreme-low context the narrative
    depends on. Reading the pre-cross bar restores that context."""

    run_index = pandas.bdate_range("2024-01-02", periods=10)
    entry_date = run_index[5]
    signal_date, confirmation_date = strategy._resolve_trade_decision_dates(
        run_index,
        entry_date,
        use_confirmation_angle=False,
        pre_cross_signal_lookback=True,
    )
    # Non-pending baseline puts signal at run_index[4] (= entry_date - 1
    # = the cross bar). The pre-cross flag pulls it back another bar to
    # run_index[3].
    assert signal_date == run_index[3]
    assert confirmation_date == run_index[4]


def test_resolve_trade_decision_dates_pre_cross_lookback_pending() -> None:
    """In pending mode ``pre_cross_signal_lookback`` also shifts signal
    back one bar (entry_date - 3 trading bars total). Same semantic:
    capture the bar before the cross, regardless of fill-mode timeline."""

    run_index = pandas.bdate_range("2024-01-02", periods=10)
    entry_date = run_index[5]
    signal_date, confirmation_date = strategy._resolve_trade_decision_dates(
        run_index,
        entry_date,
        use_confirmation_angle=True,
        pre_cross_signal_lookback=True,
    )
    # Pending baseline: signal=run_index[3] (entry-2). Flag pulls back
    # to run_index[2] (entry-3). confirmation_date unchanged.
    assert signal_date == run_index[2]
    assert confirmation_date == run_index[4]


def test_resolve_trade_decision_dates_pre_cross_lookback_edge_at_start() -> None:
    """At the start of the run frame ``pre_cross_signal_lookback`` cannot
    pull signal_date back further; degrade gracefully to whatever
    earlier bar is available."""

    run_index = pandas.bdate_range("2024-01-02", periods=10)
    # Non-pending at pos=1: baseline signal=run_index[0]; flag has
    # nowhere to pull back, must stay at run_index[0].
    signal_date, _ = strategy._resolve_trade_decision_dates(
        run_index,
        run_index[1],
        use_confirmation_angle=False,
        pre_cross_signal_lookback=True,
    )
    assert signal_date == run_index[0]


def test_resolve_trade_decision_dates_aapl_2018_12_27_non_pending() -> None:
    """For the V-cross trade entering AAPL 2018-12-27 in non-pending
    mode (cross detected at 2018-12-26), signal_date must be 2018-12-26
    (one trading bar before fill), not 2018-12-24 (two bars back as
    the buggy implementation produced).

    End-to-end check: the chip metric the artifact generator records
    on TradeDetail.above_price_volume_ratio is computed from
    ``calculate_chip_concentration_metrics(frame.loc[: signal_date], ...)``
    which means a correct signal_date routes the read to the cross
    day. Verify the chip value at 12-26 lands BELOW the V threshold
    (0.973), while the value the buggy 12-24 path would have read is
    ABOVE the threshold — the same disparity that caused V trades to
    silently disappear from the multi-bucket backtest."""

    aapl_csv = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "stock_data_2010_yf_clean"
        / "AAPL.csv"
    )
    if not aapl_csv.exists():
        pytest.skip("AAPL fixture not available in this checkout")
    from stock_indicator.chip_filter import calculate_chip_concentration_metrics

    frame = strategy.load_price_data(aapl_csv)
    run_index = frame.index
    entry_date = pandas.Timestamp("2018-12-27")
    signal_date, confirmation_date = strategy._resolve_trade_decision_dates(
        run_index, entry_date, use_confirmation_angle=False
    )
    assert signal_date == pandas.Timestamp("2018-12-26"), (
        f"non-pending signal_date should be 12-26 (cross day), "
        f"got {signal_date.date()}"
    )
    assert confirmation_date == pandas.Timestamp("2018-12-26")

    fixed_above_pv = calculate_chip_concentration_metrics(
        frame.loc[:signal_date],
        lookback_window_size=60,
        include_volume_profile=False,
    )["above_price_volume_ratio"]
    assert fixed_above_pv == pytest.approx(0.924, abs=0.01)
    assert fixed_above_pv < 0.973, (
        "12-26 above_pv must be below the V threshold post-fix"
    )

    # Pending mode on the same trade pulls signal back two bars (T+2 fill).
    pending_signal, pending_confirmation = strategy._resolve_trade_decision_dates(
        run_index, entry_date, use_confirmation_angle=True
    )
    assert pending_signal == pandas.Timestamp("2018-12-24")
    assert pending_confirmation == pandas.Timestamp("2018-12-26")
    buggy_above_pv = calculate_chip_concentration_metrics(
        frame.loc[:pending_signal],
        lookback_window_size=60,
        include_volume_profile=False,
    )["above_price_volume_ratio"]
    assert buggy_above_pv == pytest.approx(1.0, abs=1e-6), (
        "12-24 above_pv must be ~1.0 — this is the value the buggy "
        "non-pending code path was reading"
    )
    assert buggy_above_pv > 0.973, (
        "the buggy date's above_pv was above V threshold, which is "
        "why V trades silently disappeared on Boxing Day in the "
        "multi-bucket backtest"
    )


def test_adaptive_rolling_update_waits_for_raw_exit_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adaptive rolling stats must not learn raw signal results before raw exit."""

    trades_with_details = []
    for trade_index, symbol_name in enumerate(["AAA", "BBB", "CCC"]):
        entry_day = 1 + trade_index * 2
        adaptive_close_day = entry_day + 1
        trade, detail_pair = _build_trade(
            f"2024-01-{entry_day:02d}",
            f"2024-01-{20 + trade_index:02d}",
            entry_price=10.0,
            exit_price=11.0,
            profit=1.0,
            symbol=symbol_name,
        )
        trade = strategy.replace(
            trade,
            bar_excursions=[
                (
                    pandas.Timestamp(f"2024-01-{adaptive_close_day:02d}"),
                    0.02,
                    0.00,
                    0.00,
                ),
            ],
        )
        trades_with_details.append((trade, detail_pair))

    probe_trade, probe_detail_pair = _build_trade(
        "2024-01-08",
        "2024-01-12",
        entry_price=10.0,
        exit_price=10.5,
        profit=0.5,
        symbol="DDD",
    )
    probe_trade = strategy.replace(
        probe_trade,
        bar_excursions=[
            (pandas.Timestamp("2024-01-09"), 0.025, 0.00, 0.00),
            (pandas.Timestamp("2024-01-10"), 0.025, 0.00, 0.00),
            (pandas.Timestamp("2024-01-11"), 0.025, 0.00, 0.00),
            (pandas.Timestamp("2024-01-12"), 0.025, 0.00, 0.00),
        ],
    )
    trades_with_details.append((probe_trade, probe_detail_pair))
    artifacts = _build_artifacts(trades_with_details)

    def fake_generate(
        *args: object,
        **kwargs: object,
    ) -> strategy.StrategyEvaluationArtifacts:
        return artifacts

    monkeypatch.setattr(
        strategy,
        "_generate_strategy_evaluation_artifacts",
        fake_generate,
    )
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
        ),
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=1,
        adaptive_tp_sl=strategy.AdaptiveTPSLConfig(
            min_samples=1,
            sigma_multiplier=0.0,
            delayed_rolling_update=True,
        ),
    )

    probe_close_detail = next(
        detail
        for detail_list in metrics.overall_metrics.trade_details_by_year.values()
        for detail in detail_list
        if detail.action == "close" and detail.symbol == "DDD"
    )

    assert probe_close_detail.adaptive_tp_pct == pytest.approx(0.02)


def test_adaptive_stop_loss_uses_rolling_signal_losses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rolling SL should come from recent signal losses, not TP / target_r."""

    trades_with_details = []
    prior_trade_specs = [
        ("2024-01-01", "2024-01-02", "AAA", 0.30),
        ("2024-01-02", "2024-01-03", "BBB", 0.30),
        ("2024-01-03", "2024-01-04", "CCC", 0.30),
        ("2024-01-04", "2024-01-05", "DDD", -0.04),
        ("2024-01-05", "2024-01-06", "EEE", -0.04),
        ("2024-01-06", "2024-01-07", "FFF", -0.04),
    ]
    for entry_date, exit_date, symbol_name, raw_return in prior_trade_specs:
        exit_price = 10.0 * (1.0 + raw_return)
        trade, detail_pair = _build_trade(
            entry_date,
            exit_date,
            entry_price=10.0,
            exit_price=exit_price,
            profit=exit_price - 10.0,
            symbol=symbol_name,
        )
        trades_with_details.append((trade, detail_pair))

    probe_trade, probe_detail_pair = _build_trade(
        "2024-01-10",
        "2024-01-15",
        entry_price=10.0,
        exit_price=11.0,
        profit=1.0,
        symbol="ZZZ",
    )
    probe_trade = strategy.replace(
        probe_trade,
        bar_excursions=[
            (pandas.Timestamp("2024-01-11"), 0.00, -0.05, 0.00),
        ],
    )
    trades_with_details.append((probe_trade, probe_detail_pair))
    artifacts = _build_artifacts(trades_with_details)

    def fake_generate(
        *args: object,
        **kwargs: object,
    ) -> strategy.StrategyEvaluationArtifacts:
        return artifacts

    monkeypatch.setattr(
        strategy,
        "_generate_strategy_evaluation_artifacts",
        fake_generate,
    )
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
        ),
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=6,
        adaptive_tp_sl=strategy.AdaptiveTPSLConfig(
            min_samples=6,
            sigma_multiplier=0.0,
            min_sl=0.01,
        ),
    )

    probe_close_detail = next(
        detail
        for detail_list in metrics.overall_metrics.trade_details_by_year.values()
        for detail in detail_list
        if detail.action == "close" and detail.symbol == "ZZZ"
    )

    assert probe_close_detail.exit_reason == "adaptive_stop_loss"
    assert probe_close_detail.adaptive_tp_pct == pytest.approx(0.30)
    assert probe_close_detail.adaptive_sl_pct == pytest.approx(0.04)


def test_adaptive_stop_loss_uses_gap_down_open_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adaptive SL should fill at open when the bar opens below stop."""

    trade, detail_pair = _build_trade(
        "2024-01-01",
        "2024-01-05",
        entry_price=10.0,
        exit_price=10.5,
        profit=0.5,
        symbol="AAA",
    )
    trade = strategy.replace(
        trade,
        bar_excursions=[
            (pandas.Timestamp("2024-01-02"), 0.00, -0.04, -0.04),
        ],
    )
    artifacts = _build_artifacts([(trade, detail_pair)])

    def fake_generate(
        *args: object,
        **kwargs: object,
    ) -> strategy.StrategyEvaluationArtifacts:
        return artifacts

    monkeypatch.setattr(
        strategy,
        "_generate_strategy_evaluation_artifacts",
        fake_generate,
    )
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
        ),
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=1,
        adaptive_tp_sl=strategy.AdaptiveTPSLConfig(
            min_sl=0.03,
            min_tp=0.06,
        ),
    )

    close_detail = next(
        detail
        for detail_list in metrics.overall_metrics.trade_details_by_year.values()
        for detail in detail_list
        if detail.action == "close"
    )

    assert close_detail.exit_reason == "adaptive_stop_loss"
    assert close_detail.price == pytest.approx(9.6)
    assert close_detail.percentage_change == pytest.approx(-0.04)
    assert metrics.overall_metrics.mean_loss_percentage == pytest.approx(0.04)


def test_early_adaptive_stop_loss_keeps_old_min_hold_slot_rhythm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Early SL should affect P/L while slot timing follows min_hold replay."""

    stopped_trade, stopped_details = _build_trade(
        "2024-01-01",
        "2024-01-10",
        entry_price=10.0,
        exit_price=10.0,
        profit=0.0,
        symbol="AAA",
    )
    stopped_trade = strategy.replace(
        stopped_trade,
        bar_excursions=[
            (pandas.Timestamp("2024-01-02"), 0.00, -0.03, 0.00),
            (pandas.Timestamp("2024-01-03"), 0.00, 0.00, 0.00),
            (pandas.Timestamp("2024-01-04"), 0.00, 0.00, 0.00),
            (pandas.Timestamp("2024-01-05"), 0.00, 0.00, 0.00),
            (pandas.Timestamp("2024-01-08"), 0.00, 0.00, 0.00),
            (pandas.Timestamp("2024-01-09"), 0.00, 0.00, 0.00),
            (pandas.Timestamp("2024-01-10"), 0.00, 0.00, 0.00),
        ],
    )
    blocked_trade, blocked_details = _build_trade(
        "2024-01-09",
        "2024-01-10",
        symbol="BBB",
    )

    artifacts = _build_artifacts(
        [
            (stopped_trade, stopped_details),
            (blocked_trade, blocked_details),
        ],
    )

    def fake_generate(
        *args: object,
        **kwargs: object,
    ) -> strategy.StrategyEvaluationArtifacts:
        return artifacts

    monkeypatch.setattr(
        strategy,
        "_generate_strategy_evaluation_artifacts",
        fake_generate,
    )
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
        ),
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=1,
        minimum_holding_bars=5,
        adaptive_tp_sl=strategy.AdaptiveTPSLConfig(
            # Stress regime: SL > TP → R = TP/SL = 0.67 → dynamic lock = round(5 * 1.5) = 8 bars
            # AAA bar 1 low = -0.03 ≤ -0.03 SL → SL fires bar 1.
            # Shadow lock extends past BBB entry (Jan 9 = bar 6) → BBB blocked.
            min_sl=0.03,
            min_tp=0.02,
            override_min_hold_sl_only=True,
            min_hold_sl=1,
        ),
    )

    close_details = [
        detail
        for detail_list in metrics.overall_metrics.trade_details_by_year.values()
        for detail in detail_list
        if detail.action == "close"
    ]

    assert [detail.symbol for detail in close_details] == ["AAA"]
    assert close_details[0].date == pandas.Timestamp("2024-01-02")
    assert close_details[0].exit_reason == "adaptive_stop_loss"


def test_early_adaptive_stop_loss_uses_slot_trade_for_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capital metrics should use shadow slot timing, not accounting SL timing."""

    stopped_trade, stopped_details = _build_trade(
        "2024-01-01",
        "2024-01-10",
        entry_price=10.0,
        exit_price=10.0,
        profit=0.0,
        symbol="AAA",
    )
    stopped_trade = strategy.replace(
        stopped_trade,
        bar_excursions=[
            (pandas.Timestamp("2024-01-02"), 0.00, -0.03, 0.00),
            (pandas.Timestamp("2024-01-03"), 0.00, 0.00, 0.00),
            (pandas.Timestamp("2024-01-04"), 0.00, 0.00, 0.00),
            (pandas.Timestamp("2024-01-05"), 0.00, 0.00, 0.00),
            (pandas.Timestamp("2024-01-08"), 0.00, 0.00, 0.00),
            (pandas.Timestamp("2024-01-09"), 0.00, 0.00, 0.00),
            (pandas.Timestamp("2024-01-10"), 0.00, 0.00, 0.00),
        ],
    )
    overlapping_trade, overlapping_details = _build_trade(
        "2024-01-03",
        "2024-01-04",
        symbol="BBB",
    )

    artifacts = _build_artifacts(
        [
            (stopped_trade, stopped_details),
            (overlapping_trade, overlapping_details),
        ],
    )

    def fake_generate(
        *args: object,
        **kwargs: object,
    ) -> strategy.StrategyEvaluationArtifacts:
        return artifacts

    monkeypatch.setattr(
        strategy,
        "_generate_strategy_evaluation_artifacts",
        fake_generate,
    )
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
        ),
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=2,
        minimum_holding_bars=5,
        adaptive_tp_sl=strategy.AdaptiveTPSLConfig(
            # Choppy regime: TP = SL → R = 1 → dynamic lock = round(5 * 1) = 5 bars
            # AAA bar 1 low = -0.03 ≤ -0.03 SL → SL fires bar 1, slot locked to bar 5 (Jan 8).
            # BBB (Jan 3-4) overlaps with AAA slot lock.
            # max_concurrent = 2 (AAA locked slot + BBB active).
            min_sl=0.03,
            min_tp=0.03,
            override_min_hold_sl_only=True,
            min_hold_sl=1,
        ),
    )

    close_details = [
        detail
        for detail_list in metrics.overall_metrics.trade_details_by_year.values()
        for detail in detail_list
        if detail.action == "close"
    ]

    assert metrics.overall_metrics.total_trades == 2
    assert metrics.overall_metrics.maximum_concurrent_positions == 2
    assert close_details[0].symbol == "AAA"
    assert close_details[0].date == pandas.Timestamp("2024-01-02")


def test_dynamic_min_hold_releases_before_raw_signal_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dynamic slot lock should not hold until raw signal exit by default."""

    stopped_trade, stopped_details = _build_trade(
        "2024-01-01",
        "2024-01-10",
        entry_price=10.0,
        exit_price=10.0,
        profit=0.0,
        symbol="AAA",
    )
    stopped_trade = strategy.replace(
        stopped_trade,
        bar_excursions=[
            (pandas.Timestamp("2024-01-02"), 0.00, -0.03, 0.00),
            (pandas.Timestamp("2024-01-03"), 0.00, 0.00, 0.00),
            (pandas.Timestamp("2024-01-04"), 0.00, 0.00, 0.00),
            (pandas.Timestamp("2024-01-05"), 0.00, 0.00, 0.00),
            (pandas.Timestamp("2024-01-08"), 0.00, 0.00, 0.00),
            (pandas.Timestamp("2024-01-09"), 0.00, 0.00, 0.00),
            (pandas.Timestamp("2024-01-10"), 0.00, 0.00, 0.00),
        ],
    )
    next_trade, next_details = _build_trade(
        "2024-01-09",
        "2024-01-10",
        symbol="BBB",
    )

    artifacts = _build_artifacts(
        [
            (stopped_trade, stopped_details),
            (next_trade, next_details),
        ],
    )

    def fake_generate(
        *args: object,
        **kwargs: object,
    ) -> strategy.StrategyEvaluationArtifacts:
        return artifacts

    monkeypatch.setattr(
        strategy,
        "_generate_strategy_evaluation_artifacts",
        fake_generate,
    )
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
        ),
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=1,
        minimum_holding_bars=5,
        adaptive_tp_sl=strategy.AdaptiveTPSLConfig(
            min_sl=0.03,
            min_tp=0.03,
            override_min_hold_sl_only=True,
            min_hold_sl=1,
        ),
    )

    close_symbols = [
        detail.symbol
        for detail_list in metrics.overall_metrics.trade_details_by_year.values()
        for detail in detail_list
        if detail.action == "close"
    ]

    assert close_symbols == ["AAA", "BBB"]


def test_dynamic_min_hold_extends_slot_after_signal_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dynamic slot lock should use the price calendar beyond raw exit."""

    throttled_trade, throttled_details = _build_trade(
        "2024-01-01",
        "2024-01-03",
        entry_price=10.0,
        exit_price=10.0,
        profit=0.0,
        symbol="AAA",
    )
    throttled_trade = strategy.replace(
        throttled_trade,
        bar_excursions=[
            (pandas.Timestamp("2024-01-02"), 0.00, 0.00, 0.00),
            (pandas.Timestamp("2024-01-03"), 0.00, 0.00, 0.00),
        ],
    )
    blocked_trade, blocked_details = _build_trade(
        "2024-01-08",
        "2024-01-09",
        symbol="BBB",
    )

    artifacts = _build_artifacts(
        [
            (throttled_trade, throttled_details),
            (blocked_trade, blocked_details),
        ],
    )
    full_calendar = pandas.bdate_range("2024-01-01", "2024-01-12")
    artifacts.closing_price_series_by_symbol["AAA"] = pandas.Series(
        [10.0] * len(full_calendar),
        index=full_calendar,
    )

    def fake_generate(
        *args: object,
        **kwargs: object,
    ) -> strategy.StrategyEvaluationArtifacts:
        return artifacts

    monkeypatch.setattr(
        strategy,
        "_generate_strategy_evaluation_artifacts",
        fake_generate,
    )
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
        ),
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=1,
        minimum_holding_bars=5,
        adaptive_tp_sl=strategy.AdaptiveTPSLConfig(
            min_tp=0.02,
            min_sl=0.03,
            disable_sl_trigger=True,
        ),
    )

    close_details = [
        detail
        for detail_list in metrics.overall_metrics.trade_details_by_year.values()
        for detail in detail_list
        if detail.action == "close"
    ]

    assert [detail.symbol for detail in close_details] == ["AAA"]
    assert close_details[0].date == pandas.Timestamp("2024-01-03")
    assert close_details[0].exit_reason == "signal"


def test_early_adaptive_take_profit_releases_slot_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TP should not inherit the outer min_hold slot delay."""

    winning_trade, winning_details = _build_trade(
        "2024-01-01",
        "2024-01-10",
        entry_price=10.0,
        exit_price=10.0,
        profit=0.0,
        symbol="AAA",
    )
    winning_trade = strategy.replace(
        winning_trade,
        bar_excursions=[
            (pandas.Timestamp("2024-01-02"), 0.03, 0.00, 0.00),
            (pandas.Timestamp("2024-01-03"), 0.00, 0.00, 0.00),
            (pandas.Timestamp("2024-01-04"), 0.00, 0.00, 0.00),
            (pandas.Timestamp("2024-01-05"), 0.00, 0.00, 0.00),
            (pandas.Timestamp("2024-01-08"), 0.00, 0.00, 0.00),
        ],
    )
    next_trade, next_details = _build_trade(
        "2024-01-03",
        "2024-01-05",
        symbol="BBB",
    )

    artifacts = _build_artifacts(
        [
            (winning_trade, winning_details),
            (next_trade, next_details),
        ],
    )

    def fake_generate(
        *args: object,
        **kwargs: object,
    ) -> strategy.StrategyEvaluationArtifacts:
        return artifacts

    monkeypatch.setattr(
        strategy,
        "_generate_strategy_evaluation_artifacts",
        fake_generate,
    )
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
        ),
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=1,
        minimum_holding_bars=5,
        adaptive_tp_sl=strategy.AdaptiveTPSLConfig(
            min_sl=0.50,
            min_tp=0.03,
            override_min_hold_tp_only=True,
            min_hold_tp=1,
        ),
    )

    close_details = [
        detail
        for detail_list in metrics.overall_metrics.trade_details_by_year.values()
        for detail in detail_list
        if detail.action == "close"
    ]

    assert [detail.symbol for detail in close_details] == ["AAA", "BBB"]
    assert close_details[0].exit_reason == "adaptive_take_profit"


def test_run_complex_simulation_enforces_shared_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared position cap should reject excess entries across strategy sets."""

    trade_a1 = _build_trade("2024-01-01", "2024-01-03", symbol="AAA")
    trade_a2 = _build_trade("2024-01-02", "2024-01-04", symbol="AAB")
    trade_b1 = _build_trade("2024-01-02", "2024-01-05", symbol="BAA")

    artifacts_a = _build_artifacts([trade_a1, trade_a2])
    artifacts_b = _build_artifacts([trade_b1])

    artifact_map = {
        "set_a": artifacts_a,
        "set_b": artifacts_b,
    }

    def fake_generate(*args: object, **kwargs: object) -> strategy.StrategyEvaluationArtifacts:
        buy_name = kwargs.get("buy_strategy_name") or args[1]
        return artifact_map[str(buy_name)]

    monkeypatch.setattr(
        strategy,
        "_generate_strategy_evaluation_artifacts",
        fake_generate,
    )
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
        ),
        "B": strategy.ComplexStrategySetDefinition(
            label="B",
            buy_strategy_name="set_b",
            sell_strategy_name="set_b",
        ),
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=2,
    )

    assert metrics.metrics_by_set["A"].total_trades == 2
    assert metrics.metrics_by_set["B"].total_trades == 0
    assert metrics.overall_metrics.total_trades == 2
    assert metrics.overall_metrics.maximum_concurrent_positions == 2


def test_evict_oldest_keeps_evicted_exit_from_far_future_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Far-future close events must not overwrite an already evicted trade."""

    first_trade, first_details = _build_trade(
        "2024-01-01",
        "2024-01-12",
        symbol="AAA",
    )
    first_trade = strategy.replace(
        first_trade,
        bar_excursions=[
            (pandas.Timestamp("2024-01-02"), 0.01, 0.00, 0.00),
            (pandas.Timestamp("2024-01-03"), 0.01, 0.00, 0.00),
            (pandas.Timestamp("2024-01-04"), 0.01, 0.00, 0.00),
            (pandas.Timestamp("2024-01-05"), 0.01, 0.00, 0.00),
            (pandas.Timestamp("2024-01-08"), 0.01, 0.00, 0.02),
        ],
    )
    second_trade, second_details = _build_trade(
        "2024-01-08",
        "2024-01-12",
        symbol="BBB",
    )
    second_trade = strategy.replace(
        second_trade,
        bar_excursions=[
            (pandas.Timestamp("2024-01-09"), 0.01, 0.00, 0.00),
            (pandas.Timestamp("2024-01-10"), 0.01, 0.00, 0.00),
        ],
    )

    artifacts = _build_artifacts(
        [
            (first_trade, first_details),
            (second_trade, second_details),
        ],
    )

    def fake_generate(
        *args: object,
        **kwargs: object,
    ) -> strategy.StrategyEvaluationArtifacts:
        return artifacts

    monkeypatch.setattr(
        strategy,
        "_generate_strategy_evaluation_artifacts",
        fake_generate,
    )
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
        ),
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=1,
        minimum_holding_bars=5,
        adaptive_tp_sl=strategy.AdaptiveTPSLConfig(
            fixed_sl=0.03,
            evict_oldest=True,
        ),
    )

    close_details = [
        detail
        for detail_list in metrics.overall_metrics.trade_details_by_year.values()
        for detail in detail_list
        if detail.action == "close"
    ]
    first_close = next(
        detail for detail in close_details if detail.symbol == "AAA"
    )

    assert first_close.date == pandas.Timestamp("2024-01-08")
    assert first_close.exit_reason == "evicted"
    assert metrics.overall_metrics.maximum_concurrent_positions == 1


def test_same_symbol_refresh_uses_applied_stop_loss_not_fixed_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same-symbol refresh should respect the SL already applied to the trade."""

    first_signal_trade, first_signal_details = _build_trade(
        "2024-01-01",
        "2024-01-08",
        entry_price=10.0,
        exit_price=12.0,
        profit=2.0,
        symbol="AAA",
    )
    first_signal_trade = strategy.replace(
        first_signal_trade,
        exit_reason="reentry",
        bar_excursions=[
            (pandas.Timestamp("2024-01-02"), 0.01, 0.00, 0.00),
        ],
    )
    refreshed_signal_trade, refreshed_signal_details = _build_trade(
        "2024-01-03",
        "2024-01-08",
        entry_price=10.0,
        exit_price=12.0,
        profit=2.0,
        symbol="AAA",
    )
    refreshed_signal_trade = strategy.replace(
        refreshed_signal_trade,
        bar_excursions=[
            (pandas.Timestamp("2024-01-04"), 0.01, -0.02, 0.00),
        ],
    )

    artifacts = _build_artifacts(
        [
            (first_signal_trade, first_signal_details),
            (refreshed_signal_trade, refreshed_signal_details),
        ],
    )

    def fake_generate(
        *args: object,
        **kwargs: object,
    ) -> strategy.StrategyEvaluationArtifacts:
        return artifacts

    monkeypatch.setattr(strategy, "_generate_strategy_evaluation_artifacts", fake_generate)
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
        ),
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=1,
        minimum_holding_bars=1,
        adaptive_tp_sl=strategy.AdaptiveTPSLConfig(
            fixed_sl=0.03,
            evict_oldest=True,
        ),
    )

    close_details = [
        detail
        for detail_list in metrics.overall_metrics.trade_details_by_year.values()
        for detail in detail_list
        if detail.action == "close"
    ]

    assert len(close_details) == 1
    assert close_details[0].symbol == "AAA"
    assert close_details[0].date == pandas.Timestamp("2024-01-04")
    assert close_details[0].exit_reason == "adaptive_stop_loss"
    assert close_details[0].price == pytest.approx(9.9)
    assert close_details[0].adaptive_sl_pct == pytest.approx(0.01)


def test_evict_oldest_refreshes_same_symbol_signal_age_without_reentry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same-symbol signals refresh eviction age without adding a new position."""

    first_signal_trade, first_signal_details = _build_trade(
        "2024-01-01",
        "2024-01-08",
        entry_price=10.0,
        exit_price=12.0,
        profit=2.0,
        symbol="AAA",
    )
    first_signal_trade = strategy.replace(
        first_signal_trade,
        exit_reason="reentry",
        bar_excursions=[
            (pandas.Timestamp("2024-01-02"), 0.10, 0.00, 0.00),
            (pandas.Timestamp("2024-01-03"), 0.10, 0.00, 0.00),
            (pandas.Timestamp("2024-01-04"), 0.10, 0.00, 0.00),
            (pandas.Timestamp("2024-01-05"), 0.10, 0.00, 0.00),
            (pandas.Timestamp("2024-01-08"), 0.20, 0.10, 0.20),
        ],
    )
    refreshed_signal_trade, refreshed_signal_details = _build_trade(
        "2024-01-08",
        "2024-01-12",
        entry_price=12.0,
        exit_price=14.0,
        profit=2.0,
        symbol="AAA",
    )
    refreshed_signal_trade = strategy.replace(
        refreshed_signal_trade,
        bar_excursions=[
            (pandas.Timestamp("2024-01-09"), 0.25, 0.00, 0.00),
            (pandas.Timestamp("2024-01-10"), 0.25, 0.00, 0.00),
            (pandas.Timestamp("2024-01-11"), 0.25, 0.00, 0.00),
            (pandas.Timestamp("2024-01-12"), 0.25, 0.00, 0.00),
        ],
    )
    competing_trade, competing_details = _build_trade(
        "2024-01-10",
        "2024-01-12",
        symbol="BBB",
    )
    competing_trade = strategy.replace(
        competing_trade,
        bar_excursions=[
            (pandas.Timestamp("2024-01-11"), 0.01, 0.00, 0.00),
            (pandas.Timestamp("2024-01-12"), 0.01, 0.00, 0.00),
        ],
    )

    artifacts = _build_artifacts(
        [
            (first_signal_trade, first_signal_details),
            (refreshed_signal_trade, refreshed_signal_details),
            (competing_trade, competing_details),
        ],
    )

    def fake_generate(
        *args: object,
        **kwargs: object,
    ) -> strategy.StrategyEvaluationArtifacts:
        return artifacts

    monkeypatch.setattr(
        strategy,
        "_generate_strategy_evaluation_artifacts",
        fake_generate,
    )
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
        ),
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=1,
        minimum_holding_bars=5,
        adaptive_tp_sl=strategy.AdaptiveTPSLConfig(
            fixed_sl=0.03,
            evict_oldest=True,
        ),
    )

    close_details = [
        detail
        for detail_list in metrics.overall_metrics.trade_details_by_year.values()
        for detail in detail_list
        if detail.action == "close"
    ]

    assert [detail.symbol for detail in close_details] == ["AAA"]
    assert close_details[0].date == pandas.Timestamp("2024-01-12")
    assert close_details[0].exit_reason == "end_of_data"
    assert close_details[0].max_favorable_excursion_pct == pytest.approx(0.5)
    assert metrics.overall_metrics.total_trades == 1
    assert metrics.overall_metrics.maximum_concurrent_positions == 1


def test_run_complex_simulation_assigns_global_position_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Global position counts should reflect open positions across all sets."""

    trade_a = _build_trade("2024-01-01", "2024-01-04", symbol="AAA")
    trade_b = _build_trade("2024-01-02", "2024-01-05", symbol="BBB")

    artifacts_a = _build_artifacts([trade_a])
    artifacts_b = _build_artifacts([trade_b])

    artifact_map = {
        "set_a": artifacts_a,
        "set_b": artifacts_b,
    }

    def fake_generate(*args: object, **kwargs: object) -> strategy.StrategyEvaluationArtifacts:
        buy_name = kwargs.get("buy_strategy_name") or args[1]
        return artifact_map[str(buy_name)]

    monkeypatch.setattr(strategy, "_generate_strategy_evaluation_artifacts", fake_generate)
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
        ),
        "B": strategy.ComplexStrategySetDefinition(
            label="B",
            buy_strategy_name="set_b",
            sell_strategy_name="set_b",
        ),
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=4,
    )

    overall_details = [
        detail
        for detail_list in metrics.overall_metrics.trade_details_by_year.values()
        for detail in detail_list
    ]
    assert overall_details, "expected trade details for global count verification"

    def find_count(symbol: str, action: str, date: str) -> int | None:
        for detail in overall_details:
            if (
                detail.symbol == symbol
                and detail.action == action
                and detail.date == pandas.Timestamp(date)
            ):
                return detail.global_concurrent_position_count
        return None

    assert find_count("AAA", "open", "2024-01-01") == 1
    assert find_count("BBB", "open", "2024-01-02") == 2
    assert find_count("AAA", "close", "2024-01-04") == 1
    assert find_count("BBB", "close", "2024-01-05") == 0


def test_run_complex_simulation_allows_two_b_positions_when_limit_rounds_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set B should receive the rounded-up half of the shared position cap."""

    trade_a1 = _build_trade("2024-01-03", "2024-01-05", symbol="AAA")
    trade_b1 = _build_trade("2024-01-01", "2024-01-03", symbol="BAA")
    trade_b2 = _build_trade("2024-01-02", "2024-01-04", symbol="BAB")

    artifacts_a = _build_artifacts([trade_a1])
    artifacts_b = _build_artifacts([trade_b1, trade_b2])

    artifact_map = {
        "set_a": artifacts_a,
        "set_b": artifacts_b,
    }

    def fake_generate(*args: object, **kwargs: object) -> strategy.StrategyEvaluationArtifacts:
        buy_name = kwargs.get("buy_strategy_name") or args[1]
        return artifact_map[str(buy_name)]

    monkeypatch.setattr(strategy, "_generate_strategy_evaluation_artifacts", fake_generate)
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
        ),
        "B": strategy.ComplexStrategySetDefinition(
            label="B",
            buy_strategy_name="set_b",
            sell_strategy_name="set_b",
        ),
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=3,
    )

    assert metrics.metrics_by_set["B"].total_trades == 2
    assert metrics.overall_metrics.total_trades == 3


def test_run_complex_simulation_skips_b_when_global_open_count_reaches_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set B entries should be rejected once the shared open count hits its quota."""

    trade_a1 = _build_trade("2024-01-01", "2024-01-10", symbol="AAA")
    trade_a2 = _build_trade("2024-01-02", "2024-01-11", symbol="AAB")
    trade_b1 = _build_trade("2024-01-03", "2024-01-05", symbol="BAA")
    trade_b2 = _build_trade("2024-01-03", "2024-01-06", symbol="BAB")

    artifacts_a = _build_artifacts([trade_a1, trade_a2])
    artifacts_b = _build_artifacts([trade_b1, trade_b2])

    artifact_map = {
        "set_a": artifacts_a,
        "set_b": artifacts_b,
    }

    def fake_generate(*args: object, **kwargs: object) -> strategy.StrategyEvaluationArtifacts:
        buy_name = kwargs.get("buy_strategy_name") or args[1]
        return artifact_map[str(buy_name)]

    monkeypatch.setattr(strategy, "_generate_strategy_evaluation_artifacts", fake_generate)
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
        ),
        "B": strategy.ComplexStrategySetDefinition(
            label="B",
            buy_strategy_name="set_b",
            sell_strategy_name="set_b",
        ),
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=4,
    )

    assert metrics.metrics_by_set["A"].total_trades == 2
    assert metrics.metrics_by_set["B"].total_trades == 0
    assert metrics.overall_metrics.total_trades == 2


def test_run_complex_simulation_overall_metrics_use_global_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overall metric helpers should receive the shared position cap."""

    trade_a = _build_trade("2024-01-01", "2024-01-05", symbol="AAA")
    trade_b = _build_trade("2024-01-02", "2024-01-06", symbol="BBB")

    artifacts_a = _build_artifacts([trade_a])
    artifacts_b = _build_artifacts([trade_b])

    def fake_generate(*args: object, **kwargs: object) -> strategy.StrategyEvaluationArtifacts:
        buy_name = kwargs.get("buy_strategy_name") or args[1]
        return artifacts_a if buy_name == "set_a" else artifacts_b

    monkeypatch.setattr(strategy, "_generate_strategy_evaluation_artifacts", fake_generate)
    call_records = _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
        ),
        "B": strategy.ComplexStrategySetDefinition(
            label="B",
            buy_strategy_name="set_b",
            sell_strategy_name="set_b",
        ),
    }

    strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=3,
    )

    assert call_records["simulate_portfolio_balance"][-1] == 3
    assert call_records["calculate_max_drawdown"][-1] == 3


def test_run_complex_simulation_prioritizes_low_above_ratio_for_s4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set definitions linked to s4 favor lower above-ratio entries."""

    lower_ratio_trade = _build_trade(
        "2024-01-01",
        "2024-01-05",
        symbol="LOW",
        above_price_volume_ratio=0.5,
    )
    higher_ratio_trade = _build_trade(
        "2024-01-01",
        "2024-01-06",
        symbol="HIGH",
        above_price_volume_ratio=1.2,
    )

    artifacts = _build_artifacts([lower_ratio_trade, higher_ratio_trade])

    def fake_generate(*args: object, **kwargs: object) -> strategy.StrategyEvaluationArtifacts:
        return artifacts

    monkeypatch.setattr(
        strategy, "_generate_strategy_evaluation_artifacts", fake_generate
    )
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
            strategy_identifier="s4",
        )
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=1,
    )

    assert metrics.metrics_by_set["A"].total_trades == 1
    entry_details = [
        detail
        for detail in metrics.metrics_by_set["A"].trade_details_by_year.get(2024, [])
        if detail.action == "open"
    ]
    assert len(entry_details) == 1
    assert entry_details[0].symbol == "LOW"


def test_run_complex_simulation_prioritizes_low_near_ratio_for_s6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set definitions linked to s6 favor lower near-ratio entries."""

    higher_ratio_trade = _build_trade(
        "2024-01-01",
        "2024-01-05",
        symbol="HIGH",
        near_price_volume_ratio=0.8,
    )
    lower_ratio_trade = _build_trade(
        "2024-01-01",
        "2024-01-06",
        symbol="LOW",
        near_price_volume_ratio=0.2,
    )

    artifacts = _build_artifacts([higher_ratio_trade, lower_ratio_trade])

    def fake_generate(*args: object, **kwargs: object) -> strategy.StrategyEvaluationArtifacts:
        return artifacts

    monkeypatch.setattr(
        strategy, "_generate_strategy_evaluation_artifacts", fake_generate
    )
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
            strategy_identifier="s6",
        )
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=1,
    )

    assert metrics.metrics_by_set["A"].total_trades == 1
    entry_details = [
        detail
        for detail in metrics.metrics_by_set["A"].trade_details_by_year.get(2024, [])
        if detail.action == "open"
    ]
    assert len(entry_details) == 1
    assert entry_details[0].symbol == "LOW"


def _build_flat_replay_trade(
    num_bars: int,
) -> strategy.Trade:
    """Build a Trade with flat-price bar_excursions for replay tests."""
    start = pandas.Timestamp("2024-01-02")
    excursions = [
        (start + pandas.Timedelta(days=offset), 0.0, 0.0, 0.0)
        for offset in range(num_bars)
    ]
    return strategy.Trade(
        entry_date=pandas.Timestamp("2024-01-01"),
        exit_date=excursions[-1][0],
        entry_price=100.0,
        exit_price=100.0,
        profit=0.0,
        holding_period=num_bars,
        exit_reason="signal",
        bar_excursions=excursions,
    )


def test_replay_max_hold_fires_without_refire_reset() -> None:
    """Without reset flag, max_hold cuts at the original bar (control)."""
    trade = _build_flat_replay_trade(num_bars=20)
    adjusted = strategy._replay_trade_with_adaptive_tp_sl(
        trade,
        tp_pct=0.50,
        sl_pct=0.50,
        minimum_holding_bars=0,
        disable_sl_trigger=True,
        max_hold_bars=5,
        reset_hold_on_reentry_signal=False,
        re_fire_dates=None,
    )
    assert adjusted.exit_reason == "max_hold"
    assert adjusted.holding_period == 5
    assert adjusted.exit_date == trade.bar_excursions[5][0]


def test_replay_refire_resets_bars_since_anchor_extends_max_hold() -> None:
    """A re-fire on bar 5 should restart max_hold counting from that bar."""
    trade = _build_flat_replay_trade(num_bars=20)
    re_fire_dates = {trade.bar_excursions[4][0]}
    adjusted = strategy._replay_trade_with_adaptive_tp_sl(
        trade,
        tp_pct=0.50,
        sl_pct=0.50,
        minimum_holding_bars=0,
        disable_sl_trigger=True,
        max_hold_bars=5,
        reset_hold_on_reentry_signal=True,
        re_fire_dates=re_fire_dates,
    )
    assert adjusted.exit_reason == "max_hold"
    assert adjusted.holding_period == 10
    assert adjusted.exit_date == trade.bar_excursions[10][0]


def test_replay_refire_disabled_when_flag_false_even_with_dates() -> None:
    """When flag is False, re_fire_dates must be ignored (regression guard)."""
    trade = _build_flat_replay_trade(num_bars=20)
    re_fire_dates = {trade.bar_excursions[4][0]}
    adjusted = strategy._replay_trade_with_adaptive_tp_sl(
        trade,
        tp_pct=0.50,
        sl_pct=0.50,
        minimum_holding_bars=0,
        disable_sl_trigger=True,
        max_hold_bars=5,
        reset_hold_on_reentry_signal=False,
        re_fire_dates=re_fire_dates,
    )
    assert adjusted.holding_period == 5
    assert adjusted.exit_date == trade.bar_excursions[5][0]


def test_replay_refire_blocks_sl_during_new_min_hold_window() -> None:
    """Re-fire resets min_hold gate; SL cannot fire on the re-fire bar.

    Bar 3 drops -10% and is also the re-fire bar. Without re-fire, min_hold_sl=4
    would let SL fire here (holding=4). With re-fire, bars_since_anchor resets
    to 0 on bar 3, so SL is gated until bars_since_anchor reaches 4 again — i.e.
    bar 7 (also dropped -10% so SL has a trigger there).
    """
    start = pandas.Timestamp("2024-01-02")
    bars: list[tuple[pandas.Timestamp, float, float, float]] = []
    for offset in range(10):
        low_pct = -0.10 if offset in {3, 7} else 0.0
        bars.append((start + pandas.Timedelta(days=offset), 0.0, low_pct, 0.0))
    trade = strategy.Trade(
        entry_date=pandas.Timestamp("2024-01-01"),
        exit_date=bars[-1][0],
        entry_price=100.0,
        exit_price=100.0,
        profit=0.0,
        holding_period=10,
        exit_reason="signal",
        bar_excursions=bars,
    )
    re_fire_dates = {bars[3][0]}
    adjusted = strategy._replay_trade_with_adaptive_tp_sl(
        trade,
        tp_pct=0.50,
        sl_pct=0.10,
        minimum_holding_bars=4,
        minimum_holding_bars_sl=4,
        disable_sl_trigger=False,
        reset_hold_on_reentry_signal=True,
        re_fire_dates=re_fire_dates,
    )
    assert adjusted.exit_reason == "adaptive_stop_loss"
    assert adjusted.holding_period == 8
    assert adjusted.exit_date == bars[7][0]


def test_risk_score_stop_mask_zeroes_target_months() -> None:
    """Test the gate-mask logic in isolation — month-keyed mask must
    AND-zero entry signals for any bar in a stop month.

    Replicates the masking applied inside
    _generate_strategy_evaluation_artifacts after _combined_buy_entry
    is built. Keeping this as a focused unit test avoids the full
    artifacts-pipeline machinery (eligibility mask, dollar-volume
    filter, etc.) and pins down the only logic this gate adds.
    """
    bars = pandas.date_range("2010-01-04", "2010-02-26", freq="B")
    combined_buy_entry = pandas.Series(True, index=bars)
    risk_score_stop_months = {"2010-01"}

    year_month_series = bars.strftime("%Y-%m")
    stop_mask = pandas.Series(
        year_month_series, index=bars
    ).isin(risk_score_stop_months)
    gated = combined_buy_entry & (~stop_mask)

    # All 2010-01 bars must be False; all 2010-02 bars must remain True.
    jan_bars = gated[gated.index.strftime("%Y-%m") == "2010-01"]
    feb_bars = gated[gated.index.strftime("%Y-%m") == "2010-02"]
    assert not jan_bars.any()
    assert feb_bars.all()


def test_risk_score_stop_mask_noop_when_set_empty() -> None:
    """Empty or None stop_months set must not change entry signal."""
    bars = pandas.date_range("2010-01-04", "2010-02-26", freq="B")
    combined_buy_entry = pandas.Series(True, index=bars)

    # Mimic the early-return in production code: when set is falsy, the
    # masking block is skipped entirely. Verify gated == original.
    risk_score_stop_months: set[str] = set()
    if risk_score_stop_months:
        year_month_series = bars.strftime("%Y-%m")
        stop_mask = pandas.Series(
            year_month_series, index=bars
        ).isin(risk_score_stop_months)
        gated = combined_buy_entry & (~stop_mask)
    else:
        gated = combined_buy_entry.copy()
    assert gated.all()
