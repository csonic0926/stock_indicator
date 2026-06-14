"""Unit tests for the ft-family regime phantom score."""

import pandas as pd
import pytest

from stock_indicator import strategy
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


def test_risk_score_activation_threshold_field() -> None:
    # Default is None (always-on); explicit threshold stored.
    assert WRGateConfig().risk_score_activation_threshold is None
    assert WRGateConfig(
        risk_score_activation_threshold=50
    ).risk_score_activation_threshold == 50


def test_live_sensor_functions_match_simulator_close_handler() -> None:
    """update_wr_gate_sensor_state must reproduce the simulator's in-loop
    EMA/SMA/magnitude update byte-for-byte, so a live-path sensor seeded
    from the export and fed adaptive ft closes equals the simulator."""
    from collections import deque as _deque
    from stock_indicator import strategy as _s
    window = 12
    # Replicate the simulator's in-loop variables.
    sim_ema = None
    sim_win = _deque(maxlen=window)
    sim_winp = _deque(maxlen=window)
    sim_losp = _deque(maxlen=window)
    # Live sensor state dict.
    live = {"cross_ema": None, "cross_window": [], "winner_pcts": [], "loser_pcts": []}
    import random as _r
    _r.seed(7)
    closes = [(_r.random() > 0.42, _r.uniform(-0.08, 0.10)) for _ in range(60)]
    for win, pct in closes:
        # simulator in-loop update (copied semantics)
        wv = 1.0 if win else 0.0
        a = 2.0 / (window + 1.0)
        sim_ema = wv if sim_ema is None else a * wv + (1.0 - a) * sim_ema
        sim_win.append(wv)
        if pct > 0: sim_winp.append(pct)
        elif pct < 0: sim_losp.append(abs(pct))
        # live update
        _s.update_wr_gate_sensor_state(live, win, pct, window)
        # must match
        assert live["cross_ema"] == pytest.approx(sim_ema)
        assert live["cross_window"] == list(sim_win)
        assert live["winner_pcts"] == list(sim_winp)
        assert live["loser_pcts"] == list(sim_losp)


def test_evaluate_wr_gate_phantom_matches_entry_gate_logic() -> None:
    from stock_indicator import strategy as _s
    cfg = _s.WRGateConfig(window=12, curve="wr_cross")
    # Warm-up: short window -> no phantom.
    short = {"cross_ema": 0.3, "cross_window": [1.0]*5, "winner_pcts": [], "loser_pcts": []}
    assert _s.evaluate_wr_gate_phantom(short, cfg) is False
    # EMA below SMA (degrading) -> phantom.
    deg = {"cross_ema": 0.30, "cross_window": [0.0]*4 + [1.0]*8,
           "winner_pcts": [0.05]*12, "loser_pcts": [0.04]*12}
    assert _s.evaluate_wr_gate_phantom(deg, cfg) is True
    # EMA above SMA (improving), SMA 0.50 above breakeven 0.444 -> no phantom.
    healthy = {"cross_ema": 0.70, "cross_window": [0.0]*6 + [1.0]*6,
               "winner_pcts": [0.05]*12, "loser_pcts": [0.04]*12}
    assert _s.evaluate_wr_gate_phantom(healthy, cfg) is False
    # EMA above SMA but SMA below dynamic breakeven (thin payoff) -> phantom.
    thin = {"cross_ema": 0.50, "cross_window": [0.0]*7 + [1.0]*5,
            "winner_pcts": [0.03]*12, "loser_pcts": [0.06]*12}
    assert _s.evaluate_wr_gate_phantom(thin, cfg) is True


def test_compute_adaptive_ft_close_matches_simulator() -> None:
    """The live adaptive-exit helper reproduces the simulator's recorded
    ft adaptive exit (TP/max_hold) byte-for-byte on real trades."""
    import glob
    from pathlib import Path
    from stock_indicator import multi_bucket_today as mbt
    csvs = glob.glob(
        "logs/multi_bucket_simulation_result/*rs25_2010*.csv"
    )
    if not csvs:
        pytest.skip("no rs25 2010 trade-detail CSV available")
    df = pd.read_csv(csvs[0], parse_dates=["entry_date", "exit_date"])
    data_dir = Path("data/stock_data_2010_yf_clean")
    ft = df[
        df.bucket.isin(["fish_tail_production", "fish_tail_squeeze"])
        & df.exit_reason.isin(["adaptive_take_profit", "max_hold"])
    ].head(120)
    checked = 0
    for row in ft.itertuples():
        result = mbt.compute_adaptive_ft_close(
            data_dir,
            row.symbol,
            str(row.entry_date.date()),
            float(row.entry_price),
            str(row.exit_date.date()),
            float(row.adaptive_tp_pct),
            min_hold_tp=1,
            max_hold=7,
        )
        if result is None:
            continue
        win, pct, reason, adaptive_date = result
        checked += 1
        assert reason == row.exit_reason, (row.symbol, row.entry_date)
        assert pct == pytest.approx(row.percentage_change, abs=1e-4)
        assert win == (row.percentage_change > 0)
    assert checked >= 50


def test_daily_flow_sensor_matches_direct_feed() -> None:
    """The daily-flow sensor maintenance (register + advance over business
    days) must reproduce a direct sensor-bucket adaptive-close feed —
    validating the pending-list bookkeeping, same-date ordering, signal vs
    TP/max_hold classification, and no-lookahead timing end-to-end. The
    direct feed (sensor bucket, exit-then-entry order) is itself the order
    that matches the simulator's exported sensor byte-for-byte."""
    import glob
    import pathlib
    from stock_indicator import multi_bucket_today as mbt
    csvs = glob.glob("logs/multi_bucket_simulation_result/*rs25_2010*.csv")
    if not csvs:
        pytest.skip("no rs25 2010 trade-detail CSV available")
    df = pd.read_csv(csvs[0], parse_dates=["entry_date", "exit_date"])
    data_dir = pathlib.Path("data/stock_data_2010_yf_clean")
    cfg = strategy.WRGateConfig(window=12, curve="wr_cross")
    sb = "fish_tail_production"
    start, cut = pd.Timestamp("2010-01-01"), pd.Timestamp("2013-01-01")
    bucket_df = df[df.bucket == sb].copy()
    bucket_df["win"] = bucket_df.result == "win"

    # Reference: direct feed, exit<cut, (exit_date, entry_date) order.
    reference = {"cross_ema": None, "cross_window": [], "winner_pcts": [], "loser_pcts": []}
    for row in bucket_df[bucket_df.exit_date < cut].sort_values(
        ["exit_date", "entry_date"]
    ).itertuples():
        strategy.update_wr_gate_sensor_state(
            reference, bool(row.win), float(row.percentage_change), 12
        )

    # Daily flow.
    state = {
        "wr_gate_sensor": {"cross_ema": None, "cross_window": [], "winner_pcts": [], "loser_pcts": []},
        "wr_gate_pending_ft": [],
        "closed_trades": [],
    }
    by_signal: dict = {}
    sig_closes: dict = {}
    for row in bucket_df.itertuples():
        signal = (row.entry_date - pd.offsets.BDay(1)).normalize()
        by_signal.setdefault(signal, []).append(row)
        if row.exit_reason == "signal":
            sig_closes.setdefault(row.exit_date.normalize(), []).append(
                (row.symbol, str(signal.date()), str(row.exit_date.date()),
                 float(row.percentage_change))
            )
    for day in pd.bdate_range(start, cut):
        for symbol, sig_d, exit_d, raw_pct in sig_closes.get(day.normalize(), []):
            state["closed_trades"].append({
                "symbol": symbol, "bucket": sb, "entry_date": sig_d,
                "exit_date": exit_d, "raw_pct": raw_pct,
            })
        for row in by_signal.get(day.normalize(), []):
            mbt.register_wr_gate_pending_entry(
                state, cfg, row.bucket, row.symbol,
                str((row.entry_date - pd.offsets.BDay(1)).date()),
                float(row.adaptive_tp_pct), 1, 7,
            )
        mbt.advance_wr_gate_sensor(state, cfg, str(day.date()), data_dir)

    sensor = state["wr_gate_sensor"]
    assert sensor["cross_ema"] == pytest.approx(reference["cross_ema"], abs=1e-9)
    assert sensor["cross_window"] == reference["cross_window"]
    # winner/loser magnitudes within CSV rounding (6 dp).
    assert len(sensor["winner_pcts"]) == len(reference["winner_pcts"])
    for produced, expected in zip(sensor["winner_pcts"], reference["winner_pcts"]):
        assert produced == pytest.approx(expected, abs=5e-6)
