# Rolling Expectancy Gate — Mechanism Reference

This document records the mechanism of the rolling expectancy gate and its
soft-tier priority override. It describes behaviour only; it makes no claim
about performance.

Scope: simulation (`multi_bucket_simulation`) and production allocation
(`multi_bucket_daily_signal` → dashboard confirmation).

Code: `src/stock_indicator/strategy.py` (simulation sensor, gate and event
re-ranking), `src/stock_indicator/live_expectancy_gate.py` (production sensor
ledger), `src/stock_indicator/multi_bucket_today.py` (frozen live outcome
metadata), `src/stock_indicator/manage.py` (cron advancement and simulation
reporting), and `src/stock_indicator/dashboard.py` (live selection and phantom
execution). Tests: `tests/test_expectancy_gate.py`,
`tests/test_expectancy_priority_override.py`, and
`tests/test_live_expectancy_gate.py`.

---

## 1. The sensor

One global deque shared by every bucket.

| Property | Value |
|---|---|
| Contents | Signed per-trade `percentage_change` (`profit / entry_price`) |
| Length | `window` (default 20), FIFO |
| Reading | Arithmetic mean of the deque |
| Unit | Fraction per trade (e.g. `-0.0303` = −3.03% per trade) |

**What feeds it.** Every accepted closed trade, without exception:

- ordinary funded trades,
- WR-gate phantom trades,
- expectancy-gate phantom trades (see §2).

The outcome recorded is the **final adjusted** result — the same value that
reaches trade statistics and the trade-detail CSV — so adaptive TP/SL exits,
stop-loss exits, signal exits and max-hold exits all enter on the same footing.

In production, “accepted” means a BUY selected in the fresh server-side
preview and confirmed by the operator. The order layer registers that
allocation before contacting Futu. Broker submission status does not change
the sensor sample: the ledger follows the accepted counterfactual trade, just
as it follows either kind of zero-capital phantom. Re-confirming the same
`(signal_date, bucket, strategy_id, symbol)` is idempotent.

**Causality rule.** Outcomes are scheduled when a trade is accepted, but become
observable only for candidate entries whose entry date is **strictly after** the
trade's exit date. A trade exiting on day *D* cannot influence any decision made
on day *D*. Consumption order is a heap on
`(exit_date, entry_date, stable_trade_detail_order)`, so the deque contents are
deterministic regardless of the order in which trades were accepted.

When adaptive TP/SL replays a trade, the outcome is re-scheduled; only the
latest scheduled version is consumed, and a trade already consumed is never
re-consumed.

Production stores the same causal objects in
`data/live_state/expectancy_gate_state.json`. Dashboard acceptance and cron
advancement take one inter-process lock; writes are atomic. The cron resolves
accepted trades against the separate ADAPTIVE TP/SL virtual history and daily
price cache, sorts all newly observable outcomes by exit date, entry date and
stable bucket/acceptance order, then truncates the global FIFO to `window`.
For buckets with `reset_hold_on_reentry_signal`, the replay reads the union of
raw entry dates across every bucket for that symbol and re-anchors min-hold and
max-hold on those dates, matching simulator replay.
Missing state is a valid cold start only for cron initialization. Corrupt,
schema-mismatched, config-mismatched or missing-at-confirmation state blocks
the production path.

**Warm-up.** In date-range runs (`multi_bucket_simulation CONFIG START END`),
warm-up-year trades feed the sensor, so it is already warm at the start of the
statistics window. Production does not backfill unaccepted historical trades:
its state starts from confirmed live allocations and follows `cold_start` until
the deque fills.

---

## 2. Tier 1 — the stop

**Threshold**

```
stop_threshold = baseline_mean − sigma_multiplier × baseline_sigma
```

All three terms come from the config. Nothing is calibrated inside the run.

**Decision.** For each candidate entry, read the sensor as of that entry date
(§1 causality rule):

| Sensor reading | Result |
|---|---|
| `≥ stop_threshold` | Entry is funded normally |
| `< stop_threshold` | Entry becomes a **phantom-slot trade** |
| Deque not yet full | Governed by `cold_start` (`open` = funded, `closed` = phantom) |

Equality is *open*: the comparison is strictly `<`.

**Phantom-slot trade.** The entry is fully accepted and behaves normally in
every respect except capital:

- it occupies its position slot, so slot contention is unchanged;
- it runs to its exit under the usual TP/SL/min-hold/max-hold rules;
- its outcome feeds the sensor, adaptive TP/SL state, trade statistics and CSV;
- it carries **zero capital weight** in `simulate_portfolio_balance`,
  `calculate_annual_returns` and `calculate_max_drawdown`.

Zero-capital enforcement is a single union of trade identifiers shared with the
WR-gate phantom mechanism (`phantom or expectancy_gated`), so a trade marked by
either mechanism is zeroed exactly once and no other trade's sizing is touched.
In production the same union is one shared `phantom_positions.json` slot
ledger; an expectancy phantom, a WR phantom, or an entry marked by both creates
one slot and no broker order.

**Reopening.** Automatic and unconditional: phantom outcomes keep updating the
deque while the gate is closed, and the next candidate entry is funded as soon
as the reading returns to `≥ stop_threshold`. There is no hysteresis, no minimum
closure length, and no manual re-arm. The phantom stream is what makes reopening
possible — without it a closed gate would receive no new data.

---

## 3. Tier 2 — the priority override

An optional soft tier sitting **above** the stop threshold. It does not stop
trading; it changes which bucket wins a contested slot.

**Threshold**

```
soft_threshold = baseline_mean − priority_override.sigma_multiplier × baseline_sigma
```

`priority_override.sigma_multiplier` must be smaller than the stop's
`sigma_multiplier`, so the soft threshold always sits above the stop threshold.

**Evaluation granularity — once per entry day.** Bucket priority is a same-day
contest, so all candidates on one date must be ranked under one regime. The
sensor reading for a date is evaluated once and cached for that date. Because
the sensor only consumes exits strictly before the date, the answer is
independent of when during that day it is asked.

Production cron publishes one validated daily sensor heartbeat. The dashboard
uses that single decision for every candidate in the log; it never recomputes
the regime per candidate.

The soft tier is **always inactive until the deque is full**, regardless of the
stop's `cold_start` setting: a defensive reordering requires evidence.

**Effect.** On an active day, the day's entry candidates are ranked by
`priority_override.priorities` instead of the buckets' configured
`entry_priority`. Lower number = stronger claim on the slot. Ranking within a
tier is unchanged: the sort key is
`(override_bucket_priority, entry_priority, insertion_counter)`.

The squeeze-fuel contention boost (`fuel_priority_threshold`, promote one tier)
applies to the override priority exactly as it applies to the default priority.

**Interaction with the stop.** The two tiers are independent readings of the
same sensor. Below the stop threshold the reading is also below the soft
threshold, so entries are phantom **and** the day's ranking uses the override
priorities — slot contention among phantom entries stays consistent with what
funded entries would have done.

**Note on selection.** Unlike the stop — which preserves the trade set because
phantom trades still occupy their slots — the priority override *does* change
which trades are selected on contested days. Two runs differing only in this
tier can have different trade counts.

---

## 4. Configuration

```json
"expectancy_gate": {
  "enabled": true,
  "window": 20,
  "baseline_mean": 0.0087,
  "baseline_sigma": 0.0130,
  "sigma_multiplier": 3.0,
  "cold_start": "open",
  "priority_override": {
    "enabled": true,
    "sigma_multiplier": 1.5,
    "priorities": {
      "fish_head_production": 1,
      "fish_tail_squeeze": 2,
      "fish_tail_production": 3,
      "fish_head_b30_35": 4
    }
  }
}
```

| Field | Meaning |
|---|---|
| `enabled` | `false` or block absent → the whole mechanism is inert |
| `window` | Deque length, integer ≥ 2 |
| `baseline_mean` | Reference expectancy per trade |
| `baseline_sigma` | Reference dispersion used as the threshold unit |
| `sigma_multiplier` | Stop tier distance below `baseline_mean` |
| `cold_start` | `open` or `closed`; applies to the stop tier only |
| `priority_override.enabled` | `false` or block absent → soft tier inert |
| `priority_override.sigma_multiplier` | Soft tier distance; must be `< sigma_multiplier` |
| `priority_override.priorities` | Bucket label → priority integer; must cover every configured bucket exactly |

With the values above: stop at `0.0087 − 3.0 × 0.0130 = −0.0303`, soft tier at
`0.0087 − 1.5 × 0.0130 = −0.0108`.

**Validation — fail closed.** An enabled block aborts the run on: unknown keys;
`window < 2`; non-finite `baseline_mean`; `baseline_sigma ≤ 0`;
`sigma_multiplier ≤ 0`; `cold_start` outside `{open, closed}`; soft
`sigma_multiplier` not `> 0` or not `< sigma_multiplier`; non-integer or empty
`priorities`; `priorities` not covering the configured buckets exactly (unknown
or missing labels are both reported); `priority_override` present while the gate
itself is disabled.

Production also blocks BUY preview or confirmation when the cron heartbeat is
missing/not ready, contains non-finite values, disagrees with the configured
window or thresholds, activates the soft tier before a full window, or when
the live sensor state is missing/corrupt/incompatible. The persisted sensor's
evaluation date must equal the daily log date, so a cron/dashboard race cannot
apply a stale regime.

**Inertness.** With the block absent or `enabled: false`, results are identical
to a run without the feature, and the extra CSV columns are not emitted.

---

## 5. Composition with existing mechanisms

| Mechanism | Relationship |
|---|---|
| WR-gate phantom | Independent trigger, shared zero-capital union. A trade phantomed by either stays phantom. |
| `risk_score_priority_overrides` (legacy/research configs) | **Union** with the soft tier when configured: the override applies when either the month qualifies or the day's sensor reading is below the soft threshold. The LLM condition is exact membership in its `scores` list, not a `≥` comparison. Live production omits this block. |
| Risk-score stop gate (`stop_threshold`) | Independent when configured. Live production omits the block, so the historical risk-score CSV cannot stop BUY orders. |
| WR-gate `risk_score_activation_threshold` | Optional legacy/research condition. Live production omits it, so `wr_degrading=True` directly activates the WR phantom path. |

---

## 6. Observability

**Startup lines**

```
Expectancy gate: window=20 baseline_mean=0.008700 baseline_sigma=0.013000 sigma_multiplier=3.000000 threshold=-0.030300
Expectancy priority override: sigma_multiplier=1.500000 soft_threshold=-0.010800
```

**Post-run summary**

```
Expectancy gate summary: closed_episodes=N (first to last; ...), expectancy_gated_trades=N
Expectancy priority override summary: active_days=N (YYYY-MM:count  ...), override_entries=N
```

**Production cron heartbeat**

```
[EXPECTANCY_GATE_SENSOR] status=ready mean=-0.030300 stop_threshold=-0.030300 soft_threshold=-0.010800 gate_closed=False priority_override_active=True window=20/20 window_full=True open_pending=N fed_this_run=N closed_episodes=N expectancy_gated_trades=N priority_override_entries=N
```

The dashboard validates this line against the current production config and
returns the parsed state under `expectancy_gate` in order preview. Equality at
the stop threshold remains open. Decision fields stay frozen at the cron
evaluation, while acceptance counters such as `open_pending` are refreshed
from the current persisted ledger after confirmed dashboard allocations.

A closed episode is a contiguous run of gated entries — consecutive in accepted
entry order, not in calendar time.

**Simulation trade-detail CSV** — emitted only when the corresponding block is enabled,
inserted immediately after `phantom`:

| Column | Meaning |
|---|---|
| `expectancy_gated` | Entry was phantom due to the stop tier |
| `expectancy_priority_override` | Entry day was ranked under the soft tier |

`phantom` keeps its original WR-gate meaning; the columns are independent and a
trade can carry both.

---

## 7. Reference configs

| File | Contents |
|---|---|
| `data/multi_bucket_production_quad_1994_ibkr_expgate.json` | Stop tier only; retains `risk_score_priority_overrides` |
| `data/multi_bucket_production_quad_1994_ibkr_expgate_prio.json` | Stop tier + soft tier; `risk_score_priority_overrides` removed (pure mechanical trigger) |
| `data/multi_bucket_production.json` | Live cron/dashboard config; stop tier + soft tier enabled, with all runtime LLM risk-score hooks removed |

Production initializes `data/live_state/expectancy_gate_state.json` on the
first cron run after deployment. No broker order is sent by initialization.
