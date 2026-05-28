"""Tests for the cron-facing multi-bucket daily signal contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas

from stock_indicator import multi_bucket_today, strategy, symbol_seasoning


def test_load_multi_bucket_config_preserves_bucket_sigma_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Live cron config loader should not lose per-bucket TP sigma knobs."""
    config_path = tmp_path / "multi_bucket_config.json"
    config_path.write_text(
        json.dumps(
            {
                "max_position_count": 6,
                "starting_cash": 60_000,
                "margin": 1.5,
                "withdraw": 0,
                "ff12_data_path": "data/research_new_symbols_with_sector.parquet",
                "symbol_seasoning": {
                    "enabled": True,
                    "eligibility_path": "data/production_symbol_eligibility.csv",
                    "default_new_symbol_quarantine_days": 365,
                },
                "adaptive_tp_sl": {"window": 20, "sigma": 0.5},
                "buckets": [
                    {
                        "label": "fish_head_production",
                        "strategy_id": "fish_head_vacuum_turn",
                        "dollar_volume_filter": "dollar_volume>0.02%,Top500,Pick5",
                        "sigma": 0.75,
                        "skip_ff12_groups": [9, "7", 5],
                    },
                    {
                        "label": "fish_tail_explore",
                        "strategy_id": "fish_tail_blow_off_top",
                        "dollar_volume_filter": "dollar_volume>0.02%,Top500,Pick5",
                        "sigma": 0.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        multi_bucket_today,
        "load_strategy_set_mapping",
        lambda: {
            "fish_head_vacuum_turn": ("fish_head_buy", "fish_head_sell"),
            "fish_tail_blow_off_top": ("fish_tail_buy", "fish_tail_sell"),
        },
    )
    monkeypatch.setattr(
        multi_bucket_today,
        "load_strategy_entry_filters",
        lambda: {},
    )

    loaded_config = multi_bucket_today.load_multi_bucket_config(config_path)

    assert loaded_config.bucket_definitions["fish_head_production"].sigma == 0.75
    assert (
        loaded_config.bucket_definitions[
            "fish_head_production"
        ].skipped_fama_french_groups
        == {5, 7, 9}
    )
    assert loaded_config.bucket_definitions["fish_tail_explore"].sigma == 0.0
    assert (
        loaded_config.ff12_data_path_text
        == "data/research_new_symbols_with_sector.parquet"
    )
    assert loaded_config.symbol_seasoning is not None
    assert loaded_config.symbol_seasoning.enabled is True
    assert (
        loaded_config.symbol_seasoning.eligibility_path
        == "data/production_symbol_eligibility.csv"
    )


def _build_test_bucket(
    *,
    bucket_label: str,
    strategy_identifier: str,
) -> strategy.ComplexStrategySetDefinition:
    """Create a minimal bucket definition for cron contract tests."""

    return strategy.ComplexStrategySetDefinition(
        label=bucket_label,
        buy_strategy_name=f"{bucket_label}_buy",
        sell_strategy_name=f"{bucket_label}_sell",
        strategy_identifier=strategy_identifier,
        entry_priority=1,
        maximum_positions=6,
    )


def _build_test_config() -> multi_bucket_today.MultiBucketRunConfig:
    """Create a minimal multi-bucket config for daily cron tests."""

    first_bucket = _build_test_bucket(
        bucket_label="fish_head_production",
        strategy_identifier="fish_head_vacuum_turn",
    )
    second_bucket = _build_test_bucket(
        bucket_label="fish_tail_explore",
        strategy_identifier="fish_tail_blow_off_top",
    )
    return multi_bucket_today.MultiBucketRunConfig(
        bucket_definitions={
            first_bucket.label: first_bucket,
            second_bucket.label: second_bucket,
        },
        adaptive_tp_sl=strategy.AdaptiveTPSLConfig(
            window=20,
            min_hold_tp=1,
            disable_sl_trigger=True,
            delayed_rolling_update=True,
        ),
        maximum_position_count=6,
        starting_cash=60_000.0,
        withdraw_amount=0.0,
        margin_multiplier=1.5,
        minimum_holding_bars=5,
        show_trade_details=False,
        start_date_string=None,
        confirmation_mode=None,
        use_confirmation_angle=False,
        confirmation_entry_mode="close",
        confirmation_sma_angle_range=None,
        data_source_name="daily",
        symbol_list_name="test",
        ff12_data_path_text=None,
        max_same_symbol=1,
        raw_document={},
    )


def test_compute_today_signals_rejects_ineligible_symbol_before_slots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Seasoning should reject entry candidates before slot competition."""

    signal_results_by_strategy: dict[str, dict[str, Any]] = {
        "fish_head_production_buy": {
            "filtered_symbols": [("VST", 1)],
            "entry_signals": ["VST"],
            "exit_signals": [],
        },
        "fish_tail_explore_buy": {
            "filtered_symbols": [("AMZN", 1)],
            "entry_signals": ["AMZN"],
            "exit_signals": [],
        },
    }

    def fake_compute_signals_for_date(
        *,
        buy_strategy_name: str,
        **_: Any,
    ) -> dict[str, Any]:
        """Return deterministic bucket signals without reading stock data."""

        return signal_results_by_strategy[buy_strategy_name]

    monkeypatch.setattr(
        strategy,
        "compute_signals_for_date",
        fake_compute_signals_for_date,
    )
    monkeypatch.setattr(
        multi_bucket_today.daily_job,
        "filter_debug_values",
        lambda *arguments, **keyword_arguments: {
            "slope_60": 0.1,
            "near_delta": 0.2,
        },
    )
    monkeypatch.setattr(
        multi_bucket_today,
        "passes_per_bucket_entry_filters",
        lambda *arguments, **keyword_arguments: True,
    )
    monkeypatch.setattr(
        strategy,
        "compute_frozen_tp_sl_for_bucket",
        lambda *arguments, **keyword_arguments: (0.05, 0.03, 0.04, -0.02),
    )

    config = _build_test_config()
    config.symbol_seasoning = symbol_seasoning.SymbolSeasoningConfig(
        enabled=True,
        eligibility_path="eligibility.csv",
    )
    state: dict[str, Any] = {
        "schema_version": multi_bucket_today.SCHEMA_VERSION,
        "winners": [],
        "losers": [],
        "pending_rolling": [],
        "closed_trades": [],
        "accepted_entries": [],
    }

    result = multi_bucket_today.compute_today_signals(
        config=config,
        eval_date=pandas.Timestamp("2026-05-14"),
        held_positions={},
        state=state,
        data_directory=tmp_path,
        allowed_symbols=None,
        symbol_first_eligible_trade_dates={
            "VST": pandas.Timestamp("2026-05-15").date(),
            "AMZN": pandas.Timestamp("2026-05-14").date(),
        },
    )

    assert [(record.symbol, reason) for record, reason in result.rejected_records] == [
        ("VST", "symbol_seasoning")
    ]
    assert [record.symbol for record in result.accepted_records] == ["AMZN"]
    assert "('VST', 'fish_head_production', 'symbol_seasoning')" in next(
        log_line for log_line in result.log_lines if log_line.startswith("rejected:")
    )


def test_compute_today_signals_emits_all_dashboard_exit_signals(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Cron should log pure exit signals even with no virtual held position."""

    signal_results_by_strategy: dict[str, dict[str, Any]] = {
        "fish_head_production_buy": {
            "filtered_symbols": [("VST", 1), ("T", 2), ("ABT", 3)],
            "entry_signals": ["VST"],
            "exit_signals": ["T", "ABT"],
        },
        "fish_tail_explore_buy": {
            "filtered_symbols": [("AMZN", 1), ("T", 2)],
            "entry_signals": ["AMZN"],
            "exit_signals": ["T"],
        },
    }

    def fake_compute_signals_for_date(
        *,
        buy_strategy_name: str,
        **_: Any,
    ) -> dict[str, Any]:
        """Return deterministic bucket signals without reading stock data."""

        return signal_results_by_strategy[buy_strategy_name]

    def fake_filter_debug_values(*_: Any, **__: Any) -> dict[str, float]:
        """Return entry diagnostics that pass patched bucket filters."""

        return {"slope_60": 0.1, "near_delta": 0.2}

    def fake_compute_frozen_tp_sl_for_bucket(*_: Any, **__: Any) -> tuple[float, float, float, float]:
        """Return deterministic frozen TP/SL values for accepted entries."""

        return (0.05, 0.03, 0.04, -0.02)

    monkeypatch.setattr(
        strategy,
        "compute_signals_for_date",
        fake_compute_signals_for_date,
    )
    monkeypatch.setattr(
        multi_bucket_today.daily_job,
        "filter_debug_values",
        fake_filter_debug_values,
    )
    monkeypatch.setattr(
        multi_bucket_today,
        "passes_per_bucket_entry_filters",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        strategy,
        "compute_frozen_tp_sl_for_bucket",
        fake_compute_frozen_tp_sl_for_bucket,
    )

    state: dict[str, Any] = {
        "schema_version": multi_bucket_today.SCHEMA_VERSION,
        "winners": [0.04],
        "losers": [-0.02],
        "pending_rolling": [],
        "closed_trades": [],
        "accepted_entries": [],
    }

    result = multi_bucket_today.compute_today_signals(
        config=_build_test_config(),
        eval_date=pandas.Timestamp("2026-05-14"),
        held_positions={},
        state=state,
        data_directory=tmp_path,
        allowed_symbols=None,
    )

    exit_signal_lines = [
        log_line
        for log_line in result.log_lines
        if log_line.startswith("[EXIT_SIGNAL]")
    ]
    entry_signal_lines = [
        log_line
        for log_line in result.log_lines
        if log_line.startswith("[ENTRY_SIGNAL]")
    ]

    assert exit_signal_lines == [
        (
            "[EXIT_SIGNAL] symbol=T "
            "buckets=fish_head_production,fish_tail_explore "
            "strategies=fish_head_vacuum_turn,fish_tail_blow_off_top"
        ),
        (
            "[EXIT_SIGNAL] symbol=ABT "
            "buckets=fish_head_production "
            "strategies=fish_head_vacuum_turn"
        ),
    ]
    assert entry_signal_lines == [
        (
            "[ENTRY_SIGNAL] bucket=fish_head_production "
            "strategy_id=fish_head_vacuum_turn symbol=VST"
        ),
        (
            "[ENTRY_SIGNAL] bucket=fish_tail_explore "
            "strategy_id=fish_tail_blow_off_top symbol=AMZN"
        ),
    ]
    assert any(
        log_line.startswith("[ROLLING_TP_SL_STATE] winners=1 losers=1")
        for log_line in result.log_lines
    )
    assert sum(
        log_line.startswith("[BUCKET_TP_SL]")
        for log_line in result.log_lines
    ) == 2
    assert sum(
        log_line.startswith("[FROZEN_TP_SL]")
        for log_line in result.log_lines
    ) == 2
    assert [
        entry["disable_sl_trigger"]
        for entry in state["accepted_entries"]
    ] == [True, True]


def test_exit_alpha_factor_uses_recursive_simulation_formula() -> None:
    """Raw exit should use the same recursive EMA(alpha) as simulation."""

    date_index = pandas.to_datetime(
        [
            "2026-04-29",
            "2026-04-30",
            "2026-05-01",
            "2026-05-04",
            "2026-05-05",
            "2026-05-06",
            "2026-05-07",
            "2026-05-08",
            "2026-05-11",
            "2026-05-12",
            "2026-05-13",
            "2026-05-14",
        ]
    )
    close_values = pandas.Series(
        [
            291.76,
            292.07,
            286.64,
            284.10,
            285.17,
            284.10,
            283.70,
            275.75,
            274.60,
            274.84,
            275.70,
            274.97,
        ],
        index=date_index,
    )
    price_frame = pandas.DataFrame(
        {
            "open": close_values,
            "high": close_values + 1.0,
            "low": close_values - 1.0,
            "close": close_values,
            "volume": [1_000_000] * len(close_values),
        },
        index=date_index,
    )

    strategy.attach_ema_sma_cross_testing_signals(
        price_frame,
        window_size=3,
        angle_range=(-0.01, 65),
        near_range=(-10.0, 10.0),
        above_range=(0.78, 1.0),
        include_raw_signals=True,
        exit_alpha_factor=3,
    )

    moving_average = close_values.rolling(3).mean()
    recursive_exit_ema = close_values.round(3).ewm(
        alpha=3 / (3 + 1),
        adjust=False,
    ).mean()
    recursive_exit_signal = (
        (recursive_exit_ema.shift(1) >= moving_average.shift(1))
        & (recursive_exit_ema < moving_average)
    )

    assert bool(recursive_exit_signal.loc["2026-05-14"]) is True
    assert bool(
        price_frame.loc[
            "2026-05-14",
            "ema_sma_cross_testing_raw_exit_signal",
        ]
    ) is True
