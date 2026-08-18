# Expectancy Gates — Current Mechanism Reference

This document describes the current quantitative gates. It is a mechanism
reference, not a performance claim.

Scope:

- the **global expectancy stop** runs in simulation and production;
- the **FT-family expectancy gate** is a simulation research mechanism;
- the **FT-family win-rate gate** remains research-only and is absent from
  production and canonical simulation configs.

Current bucket allocation is static in both simulation and production:

| Bucket | Priority |
|---|---:|
| `fish_head_production` | 1 |
| `fish_tail_squeeze` | 2 |
| `fish_tail_production` | 3 |
| `fish_head_b30_35` | 4 |

Lower numbers win contested slots.

## 1. Global expectancy stop

The global sensor is one FIFO deque shared by all buckets.

| Property | Definition |
|---|---|
| Sample | Final signed trade return, `profit / entry_price` |
| Window | Latest `window` observable outcomes |
| Reading | Arithmetic mean |
| Threshold | `baseline_mean - sigma_multiplier * baseline_sigma` |

Every **accepted** trade can feed the sensor, including funded trades and
zero-capital phantoms. TP, SL, max-hold and signal exits all count because the
sensor records the final adjusted outcome rather than only signal exits.

An outcome can affect entries only when its exit date is strictly before the
candidate entry date. An exit on day D cannot affect a decision on day D. This
no-lookahead ordering is deterministic even when several outcomes become
observable together.

Decision:

- full window and mean below the threshold: accept the entry as a zero-capital
  phantom that still occupies a slot;
- full window and mean at or above the threshold: fund normally;
- incomplete window: follow `cold_start` (`open` or `closed`).

The comparison is strictly `<`; equality is open. Phantom outcomes continue to
feed the sensor, allowing the gate to reopen without manual intervention.

### Configuration

```json
"expectancy_gate": {
  "enabled": true,
  "window": 20,
  "baseline_mean": 0.0087,
  "baseline_sigma": 0.013,
  "sigma_multiplier": 3.0,
  "cold_start": "open"
}
```

The production threshold is:

```text
0.0087 - 3.0 * 0.013 = -0.0303
```

Unknown keys and invalid values fail validation instead of silently changing
behaviour.

## 2. Production state and causality

Production stores the global sensor ledger in
`data/live_state/expectancy_gate_state.json`.

- The dashboard registers a server-preview-selected BUY before broker
  submission.
- Reconfirming the same `(signal_date, bucket, strategy_id, symbol)` is
  idempotent.
- The cron resolves final TP/SL/max-hold/signal outcomes from the frozen entry
  metadata and daily prices.
- State writes are locked and atomic.
- Missing, corrupt, schema-mismatched or config-mismatched state blocks the
  production path rather than guessing.
- The dashboard validates that the persisted evaluation date and cron heartbeat
  match the current signal date and config.

The heartbeat shape is:

```text
[EXPECTANCY_GATE_SENSOR] status=ready mean=-0.030300 stop_threshold=-0.030300 gate_closed=False window=20/20 window_full=True open_pending=N fed_this_run=N closed_episodes=N expectancy_gated_trades=N
```

## 3. FT-family expectancy gate (simulation research)

This sensor observes every eligible signal from `sensor_buckets`, including
signals later blocked by shared capacity, per-bucket capacity, same-symbol
limits, static allocation priority or another gate. Each signal gets an
independent counterfactual trade replay with TP/SL state frozen at its entry.
Its outcome becomes observable only after the replay exit date.

Therefore FH allocation cannot reduce the FT sensor sample. Entry filters,
cohort filters and symbol-seasoning eligibility still determine whether a
candidate is an eligible FT signal.

Default FT sensor and gated families are:

```json
"sensor_buckets": ["fish_tail_production", "fish_tail_squeeze"],
"gated_buckets": ["fish_tail_production", "fish_tail_squeeze"]
```

Accepted entries in a closed FT gate become zero-capital, slot-holding
phantoms. The independent CSV flag is `ft_expectancy_gated`.

### Rolling-trade mode

```json
"ft_family_expectancy_gate": {
  "enabled": true,
  "window": 20,
  "baseline_mean": 0.0043,
  "baseline_sigma": 0.011,
  "sigma_multiplier": 1.5,
  "decision_mode": "rolling_trades"
}
```

The decision uses the latest full FIFO and the same strict threshold form as
the global gate.

### Previous-calendar-period mode

```json
"ft_family_expectancy_gate": {
  "enabled": true,
  "window": 20,
  "baseline_mean": 0.0043,
  "baseline_sigma": 0.011,
  "sigma_multiplier": 1.5,
  "decision_mode": "previous_calendar_period",
  "period_months": 6,
  "period_decision_threshold": 0.0,
  "period_minimum_samples": 20
}
```

At each calendar boundary, outcomes whose exit dates fall in the immediately
preceding period set one decision for the entire current period. Insufficient
prior samples leave the gate open. `period_months` must divide 12, so months,
quarters, halves and years remain calendar-aligned.

This mode is deliberately causal and coarse: the current period cannot alter
its own decision.

## 4. FT-family win-rate gate (research only)

The implementation remains available for isolated research, but every current
production and simulation config omits it. Therefore it has no effect on live
orders or the canonical FT baseline while FT-family improvement is unresolved.

When an explicit research config enables it, the FT-WR sensor is separate from
both expectancy sensors. For `curve=wr_cross`, a full-window degrading reading
directly makes configured FT-family entries zero-capital, slot-holding phantoms.
There is no second condition joining the WR reading to the global expectancy
mean.

```json
"ft_family_wr_gate": {
  "enabled": true,
  "sensor_bucket": "fish_tail_production",
  "gated_buckets": [
    "fish_tail_production",
    "fish_tail_squeeze"
  ],
  "window": 12,
  "curve": "wr_cross"
}
```

Only an explicit research config with `enabled: true` starts the WR statistical
state and phantom path. A missing block—or a block without that explicit
opt-in—is off.

## 5. Phantom union and reporting

The global expectancy stop, FT-family expectancy gate and FT-WR gate are
independent triggers sharing one phantom union. A trade triggered by more than
one mechanism:

- is zeroed only once;
- occupies one slot;
- follows the ordinary exit rules;
- remains observable to the relevant statistical sensors.

Simulation reporting includes:

```text
Expectancy gate summary: closed_episodes=N (...), expectancy_gated_trades=N
FT expectancy gate summary: closed_episodes=N (...), ft_expectancy_gated_trades=N
```

When enabled, trade-detail CSVs add `expectancy_gated` and/or
`ft_expectancy_gated`. The existing `phantom` field continues to identify the
WR mechanism.

## 6. Code and tests

Implementation:

- `src/stock_indicator/strategy.py`
- `src/stock_indicator/live_expectancy_gate.py`
- `src/stock_indicator/multi_bucket_today.py`
- `src/stock_indicator/manage.py`
- `src/stock_indicator/dashboard.py`

Focused tests:

- `tests/test_expectancy_gate.py`
- `tests/test_live_expectancy_gate.py`
- `tests/test_wr_gate.py`
- `tests/test_dashboard_risk_gate.py`
- `tests/test_multi_bucket_today_cron.py`
