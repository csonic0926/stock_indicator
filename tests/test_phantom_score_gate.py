"""Unit tests for the ft-family regime phantom score."""

import pytest

from stock_indicator.strategy import (
    PhantomScoreGateConfig,
    compute_regime_phantom_score,
)

CFG = PhantomScoreGateConfig()  # w=(0.5, 0.5, 0), threshold 0.5


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
    assert compute_regime_phantom_score(outcomes(8, 3, 4), CFG) == 0.0


def test_2002_signature_saturates() -> None:
    # Dec-2001 type: WR 3/12=0.25, zero TP, 5 max_hold.
    score = compute_regime_phantom_score(outcomes(3, 0, 5), CFG)
    # WR term: clip((0.60-0.25)*10)=1.0 -> 0.5; noTP: (1.0-0.8)/0.2=1.0 -> 0.5
    assert score == pytest.approx(1.0)


def test_partial_degradation_crosses_threshold() -> None:
    # WR 6/12=0.50 (WR term 1.0 -> 0.5), TP 1/12 -> noTP 0.917 -> term
    # 0.583 -> 0.29. Total ~0.79 > 0.5 threshold.
    score = compute_regime_phantom_score(outcomes(6, 1, 4), CFG)
    assert score == pytest.approx(0.5 + 0.5 * ((11 / 12 - 0.8) / 0.2))
    assert score > CFG.score_threshold


def test_mild_dip_stays_below_threshold() -> None:
    # WR 7/12=0.583 (term 0.167 -> 0.083), TP 2/12 -> noTP 0.833 (term
    # 0.167 -> 0.083). Total 0.167 — noise-level dip does not gate.
    score = compute_regime_phantom_score(outcomes(7, 2, 3), CFG)
    assert score < CFG.score_threshold


def test_max_hold_term_inactive_at_zero_weight() -> None:
    base = compute_regime_phantom_score(outcomes(8, 3, 0), CFG)
    heavy_mh = compute_regime_phantom_score(outcomes(8, 3, 9), CFG)
    assert base == heavy_mh  # w3=0: mh share irrelevant


def test_ablation_weights_activate_max_hold() -> None:
    cfg = PhantomScoreGateConfig(
        weight_wr=0.4, weight_no_tp=0.4, weight_max_hold=0.2
    )
    low_mh = compute_regime_phantom_score(outcomes(8, 3, 0), cfg)
    high_mh = compute_regime_phantom_score(outcomes(8, 3, 9), cfg)
    # mh 9/12 = 0.75 -> term (0.75-0.40)/0.20 capped at 1.0 -> +0.2
    assert high_mh - low_mh == pytest.approx(0.2)


def test_empty_window_scores_zero() -> None:
    assert compute_regime_phantom_score([], CFG) == 0.0
