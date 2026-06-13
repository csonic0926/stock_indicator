"""Unit tests for WR-synced sizing margin override computation."""

import pandas as pd
import pytest

from stock_indicator.simulator import Trade
from stock_indicator.strategy import (
    WRSyncedSizingConfig,
    compute_wr_synced_margin_overrides,
)


def make_trade(entry: str, exit_: str, win: bool) -> Trade:
    entry_price = 10.0
    exit_price = 11.0 if win else 9.0
    return Trade(
        entry_date=pd.Timestamp(entry),
        exit_date=pd.Timestamp(exit_),
        entry_price=entry_price,
        exit_price=exit_price,
        profit=exit_price - entry_price,
        holding_period=5,
    )


def daily_trades(start: str, outcomes: list[bool]) -> list[Trade]:
    """One trade closing per business day starting at `start`."""
    dates = pd.bdate_range(start, periods=len(outcomes) + 1)
    return [
        make_trade(str(dates[i].date()), str(dates[i + 1].date()), outcome)
        for i, outcome in enumerate(outcomes)
    ]


def test_warmup_returns_no_overrides() -> None:
    config = WRSyncedSizingConfig(window=40, wr_floor=0.45, wr_healthy=0.60)
    trades = daily_trades("2020-01-01", [False] * 39)
    assert compute_wr_synced_margin_overrides(trades, 1.5, config) == {}


def test_healthy_win_rate_returns_no_overrides() -> None:
    config = WRSyncedSizingConfig(window=40, wr_floor=0.45, wr_healthy=0.60)
    # 70% WR across 60 trades: every window of 40 stays >= healthy.
    outcomes = ([True] * 7 + [False] * 3) * 6
    trades = daily_trades("2020-01-01", outcomes)
    assert compute_wr_synced_margin_overrides(trades, 1.5, config) == {}


def test_degraded_win_rate_scales_margin_linearly() -> None:
    config = WRSyncedSizingConfig(window=40, wr_floor=0.45, wr_healthy=0.60)
    # Exactly 50% WR in the trailing 40-trade window before March; a probe
    # trade entering in March makes March an in-range entry month.
    outcomes = [True, False] * 20
    trades = daily_trades("2020-01-01", outcomes)  # closes span Jan-Feb
    trades.append(make_trade("2020-03-02", "2020-03-09", True))
    overrides = compute_wr_synced_margin_overrides(trades, 1.5, config)
    # multiplier = (0.50 - 0.45) / (0.60 - 0.45) = 1/3; the March probe
    # win must NOT count toward March's own margin (no lookahead).
    assert overrides["2020-03"] == pytest.approx(1.5 / 3)


def test_win_rate_at_floor_zeroes_margin() -> None:
    config = WRSyncedSizingConfig(window=40, wr_floor=0.45, wr_healthy=0.60)
    outcomes = [False] * 40
    trades = daily_trades("2020-01-01", outcomes)
    trades.append(make_trade("2020-03-02", "2020-03-09", True))
    overrides = compute_wr_synced_margin_overrides(trades, 1.5, config)
    assert overrides["2020-03"] == pytest.approx(0.0)


def test_no_lookahead_month_of_close_unaffected() -> None:
    config = WRSyncedSizingConfig(window=10, wr_floor=0.45, wr_healthy=0.60)
    # 10 losses all closing inside January: January itself must carry no
    # override (window only filled by trades closed BEFORE the month).
    trades = daily_trades("2020-01-02", [False] * 10)
    trades.append(make_trade("2020-02-03", "2020-02-10", True))
    overrides = compute_wr_synced_margin_overrides(trades, 1.5, config)
    assert "2020-01" not in overrides
    assert overrides["2020-02"] == pytest.approx(0.0)


def test_recovery_restores_full_margin() -> None:
    config = WRSyncedSizingConfig(window=10, wr_floor=0.45, wr_healthy=0.60)
    # 10 losses closing in January, then 10 wins closing in February.
    trades = daily_trades("2020-01-02", [False] * 10)
    trades.extend(daily_trades("2020-02-03", [True] * 10))
    trades.append(make_trade("2020-03-02", "2020-03-09", True))
    overrides = compute_wr_synced_margin_overrides(trades, 1.5, config)
    # February sized on January's 10 losses: zero margin.
    assert overrides["2020-02"] == pytest.approx(0.0)
    # March sized on February's 10 wins: full margin (absent from map).
    assert "2020-03" not in overrides


def test_invalid_anchor_ordering_raises() -> None:
    config = WRSyncedSizingConfig(window=10, wr_floor=0.60, wr_healthy=0.45)
    trades = daily_trades("2020-01-02", [True] * 12)
    with pytest.raises(ValueError):
        compute_wr_synced_margin_overrides(trades, 1.5, config)


def test_z_curve_ignores_insignificant_dips() -> None:
    config = WRSyncedSizingConfig(window=40, curve="z_score")
    # WR 0.45 over 40 trades: z = (-0.05)*sqrt(40)/0.5 = -0.63 — noise.
    outcomes = [True] * 18 + [False] * 22
    trades = daily_trades("2020-01-01", outcomes)
    trades.append(make_trade("2020-03-02", "2020-03-09", True))
    assert compute_wr_synced_margin_overrides(trades, 1.5, config) == {}


def test_z_curve_scales_on_significant_degradation() -> None:
    config = WRSyncedSizingConfig(window=40, curve="z_score")
    # WR 0.30 over 40 trades: z = (-0.20)*sqrt(40)/0.5 = -2.5298
    # multiplier = (z - (-3.0)) / (-1.5 - (-3.0)) = 0.4701/1.5 = 0.3134
    outcomes = [True] * 12 + [False] * 28
    trades = daily_trades("2020-01-01", outcomes)
    trades.append(make_trade("2020-03-02", "2020-03-09", True))
    overrides = compute_wr_synced_margin_overrides(trades, 1.5, config)
    import math
    z = (0.30 - 0.5) * math.sqrt(40) / 0.5
    expected = 1.5 * (z - (-3.0)) / 1.5
    assert overrides["2020-03"] == pytest.approx(expected)


def test_z_curve_zeroes_below_floor() -> None:
    config = WRSyncedSizingConfig(window=40, curve="z_score")
    # WR 0.20: z = -3.79 < z_floor -> margin 0.
    outcomes = [True] * 8 + [False] * 32
    trades = daily_trades("2020-01-01", outcomes)
    trades.append(make_trade("2020-03-02", "2020-03-09", True))
    overrides = compute_wr_synced_margin_overrides(trades, 1.5, config)
    assert overrides["2020-03"] == pytest.approx(0.0)


def test_invalid_curve_name_raises() -> None:
    config = WRSyncedSizingConfig(window=10, curve="sigmoid")
    trades = daily_trades("2020-01-02", [True] * 12)
    with pytest.raises(ValueError):
        compute_wr_synced_margin_overrides(trades, 1.5, config)


def test_expectancy_z_deaf_to_frequency_but_hears_magnitude() -> None:
    # 252 baseline trades of small returns, then 40 trades whose LOSSES
    # are 5x bigger while WR stays ~50% — a magnitude event.
    config = WRSyncedSizingConfig(
        window=40, curve="expectancy_z", sigma_ref_window=252
    )
    trades = daily_trades("2018-01-01", [True, False] * 126)  # baseline
    big_loss_phase = []
    dates = pd.bdate_range("2019-01-01", periods=41)
    for i in range(40):
        win = i % 2 == 0
        entry, exit_ = 10.0, (10.5 if win else 5.0)  # +5% wins, -50% losses
        big_loss_phase.append(
            Trade(
                entry_date=dates[i],
                exit_date=dates[i + 1],
                entry_price=entry,
                exit_price=exit_,
                profit=exit_ - entry,
                holding_period=5,
            )
        )
    trades.extend(big_loss_phase)
    trades.append(make_trade("2019-04-01", "2019-04-08", True))
    overrides = compute_wr_synced_margin_overrides(trades, 1.5, config)
    # Magnitude collapse must zero the margin even though WR = 50%.
    assert overrides["2019-04"] == pytest.approx(0.0)


def test_dual_z_takes_minimum_of_channels() -> None:
    # Frequency event: WR collapses to 0.25 with normal magnitudes.
    # WR-z fires hard; expectancy-z may or may not — dual must be <= WR-z.
    config_dual = WRSyncedSizingConfig(window=40, curve="dual_z", sigma_ref_window=252)
    config_wr = WRSyncedSizingConfig(window=40, curve="z_score")
    trades = daily_trades("2018-01-01", [True, False] * 126)
    trades.extend(daily_trades("2019-01-01", [True] + [False] * 3) * 0 or
                  daily_trades("2019-01-01", ([True] + [False] * 3) * 10))
    trades.append(make_trade("2019-04-01", "2019-04-08", True))
    dual = compute_wr_synced_margin_overrides(trades, 1.5, config_dual)
    wr_only = compute_wr_synced_margin_overrides(trades, 1.5, config_wr)
    assert dual["2019-04"] <= wr_only["2019-04"] + 1e-9


def test_expectancy_z_warmup_needs_reference_window() -> None:
    # Only 100 trades of history (< window + sigma_ref_window): the
    # expectancy channel must stay silent even through a magnitude crash.
    config = WRSyncedSizingConfig(
        window=40, curve="expectancy_z", sigma_ref_window=252
    )
    trades = daily_trades("2020-01-01", [False] * 100)
    trades.append(make_trade("2020-07-01", "2020-07-08", True))
    assert compute_wr_synced_margin_overrides(trades, 1.5, config) == {}
