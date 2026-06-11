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
