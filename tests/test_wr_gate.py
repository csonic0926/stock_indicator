"""Unit tests for the ft-family regime phantom score."""

import pytest

from stock_indicator.strategy import (
    WRGateConfig,
    compute_wr_gate_score,
)

CFG = WRGateConfig()  # w=(0.5, 0.5, 0), threshold 0.5


def outcomes(wins: int, tps: int, mhs: int, total: int = 12):
    """Build (win, exit_reason) tuples: tps TP exits, mhs max_hold, rest signal."""
    result = []
    for i in range(total):
        reason = (
            "adaptive_take_profit" if i < tps
            else "max_hold" if i < tps + mhs
            else "signal"
        )
        result.append((i < wins, reason))
    return result


def test_healthy_window_scores_zero() -> None:
    # WR 8/12=0.67, TP 3/12 -> noTP 0.75 < 0.80 anchor.
    assert compute_wr_gate_score(outcomes(8, 3, 4), CFG) == 0.0


def test_2002_signature_saturates() -> None:
    # Dec-2001 type: WR 3/12=0.25, zero TP, 5 max_hold.
    score = compute_wr_gate_score(outcomes(3, 0, 5), CFG)
    # WR term: clip((0.60-0.25)*10)=1.0 -> 0.5; noTP: (1.0-0.8)/0.2=1.0 -> 0.5
    assert score == pytest.approx(1.0)


def test_partial_degradation_crosses_threshold() -> None:
    # WR 6/12=0.50 (WR term 1.0 -> 0.5), TP 1/12 -> noTP 0.917 -> term
    # 0.583 -> 0.29. Total ~0.79 > 0.5 threshold.
    score = compute_wr_gate_score(outcomes(6, 1, 4), CFG)
    assert score == pytest.approx(0.5 + 0.5 * ((11 / 12 - 0.8) / 0.2))
    assert score > CFG.score_threshold


def test_mild_dip_stays_below_threshold() -> None:
    # WR 7/12=0.583 (term 0.167 -> 0.083), TP 2/12 -> noTP 0.833 (term
    # 0.167 -> 0.083). Total 0.167 — noise-level dip does not gate.
    score = compute_wr_gate_score(outcomes(7, 2, 3), CFG)
    assert score < CFG.score_threshold


def test_max_hold_term_inactive_at_zero_weight() -> None:
    base = compute_wr_gate_score(outcomes(8, 3, 0), CFG)
    heavy_mh = compute_wr_gate_score(outcomes(8, 3, 9), CFG)
    assert base == heavy_mh  # w3=0: mh share irrelevant


def test_ablation_weights_activate_max_hold() -> None:
    cfg = WRGateConfig(
        weight_wr=0.4, weight_no_tp=0.4, weight_max_hold=0.2
    )
    low_mh = compute_wr_gate_score(outcomes(8, 3, 0), cfg)
    high_mh = compute_wr_gate_score(outcomes(8, 3, 9), cfg)
    # mh 9/12 = 0.75 -> term (0.75-0.40)/0.20 capped at 1.0 -> +0.2
    assert high_mh - low_mh == pytest.approx(0.2)


def test_empty_window_scores_zero() -> None:
    assert compute_wr_gate_score([], CFG) == 0.0


def test_wr_cross_curve_validation() -> None:
    import pytest as _pytest
    from stock_indicator import strategy as _strategy
    cfg = WRGateConfig(curve="nonsense")
    with _pytest.raises(ValueError):
        _strategy.run_complex_simulation(
            data_directory=None,  # never reached: validation fires first
            set_definitions={"x": _strategy.ComplexStrategySetDefinition(
                label="x", buy_strategy_name="b", sell_strategy_name="s",
                strategy_identifier="fish_tail_blow_off_top")},
            maximum_position_count=7,
            adaptive_tp_sl=_strategy.AdaptiveTPSLConfig(),
            wr_gate=cfg,
        )


def test_wr_cross_ema_sma_semantics_match_pandas() -> None:
    """The incremental EMA/SMA must match pandas ewm/rolling conventions."""
    import pandas as _pd
    wins = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    n = 5
    series = _pd.Series([float(w) for w in wins])
    expected_ema = series.ewm(span=n, adjust=False).mean()
    expected_sma = series.rolling(n).mean()
    # Replicate the in-loop incremental computation.
    alpha = 2.0 / (n + 1.0)
    ema = None
    window: list[float] = []
    for i, w in enumerate(wins):
        v = float(w)
        ema = v if ema is None else alpha * v + (1.0 - alpha) * ema
        window.append(v)
        window = window[-n:]
        assert ema == pytest.approx(expected_ema.iloc[i])
        if len(window) >= n:
            assert sum(window) / len(window) == pytest.approx(
                expected_sma.iloc[i]
            )


def test_dynamic_breakeven_moves_with_payoff_ratio() -> None:
    from collections import deque
    from stock_indicator.strategy import compute_dynamic_breakeven_win_rate
    # P/L = 1.0 -> breakeven 0.50 (the old static line is the special case)
    even = compute_dynamic_breakeven_win_rate(
        deque([0.04] * 10), deque([0.04] * 10), 10
    )
    assert even == pytest.approx(0.50)
    # P/L = 1.5 -> breakeven 0.40: WR 0.45 is ALIVE in a fat-payoff market
    fat = compute_dynamic_breakeven_win_rate(
        deque([0.06] * 10), deque([0.04] * 10), 10
    )
    assert fat == pytest.approx(0.40)
    # P/L = 0.8 -> raw breakeven 0.556 but the greedy cap holds at 0.50:
    # the line relaxes in fat regimes, never tightens above coin-flip.
    thin = compute_dynamic_breakeven_win_rate(
        deque([0.04] * 10), deque([0.05] * 10), 10
    )
    assert thin == pytest.approx(0.50)
    # Warmup: either side short -> None (floor stays dark, no static fallback)
    assert compute_dynamic_breakeven_win_rate(
        deque([0.04] * 9), deque([0.04] * 10), 10
    ) is None
