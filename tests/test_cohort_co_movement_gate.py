import pandas
import pytest

from stock_indicator import strategy


def make_trade_detail(
    *,
    symbol_return: float | None,
    cohort_return: float | None,
    gap: float | None,
    peer_count: int,
) -> strategy.TradeDetail:
    return strategy.TradeDetail(
        date=pandas.Timestamp("2026-01-02"),
        symbol="TEST",
        action="open",
        price=10.0,
        simple_moving_average_dollar_volume=1_000_000.0,
        total_simple_moving_average_dollar_volume=10_000_000.0,
        simple_moving_average_dollar_volume_ratio=0.10,
        cohort_symbol_lookback_return=symbol_return,
        cohort_median_lookback_return=cohort_return,
        cohort_idiosyncratic_gap=gap,
        cohort_peer_count=peer_count,
    )


def test_cohort_co_movement_gate_skips_isolated_deep_drawdown() -> None:
    gate_config = strategy.CohortCoMovementGateConfig()
    entry_detail = make_trade_detail(
        symbol_return=-0.13,
        cohort_return=-0.004,
        gap=-0.126,
        peer_count=20,
    )

    assert strategy.should_skip_for_cohort_co_movement_gate(
        entry_detail,
        gate_config,
    )


def test_cohort_co_movement_gate_keeps_confirmed_cohort_selloff() -> None:
    gate_config = strategy.CohortCoMovementGateConfig()
    entry_detail = make_trade_detail(
        symbol_return=-0.13,
        cohort_return=-0.06,
        gap=-0.07,
        peer_count=20,
    )

    assert not strategy.should_skip_for_cohort_co_movement_gate(
        entry_detail,
        gate_config,
    )


def test_parse_cohort_co_movement_gate_rejects_invalid_peer_count() -> None:
    with pytest.raises(ValueError, match="minimum_peer_count"):
        strategy.parse_cohort_co_movement_gate_config(
            {"minimum_peer_count": 0},
            bucket_label="fish_head_production",
        )
