# Production Runtime Configuration Contract

This document records the current production JSON contract so the live config
does not drift through accidental `.json` edits.

Last aligned with `data/multi_bucket_production.json`: 2026-08-18.

---

## Live production entry point

Live daily signal generation uses:

```bash
multi_bucket_daily_signal data/multi_bucket_production.json
```

`run_daily_job.sh`, `dashboard.py`, and `place_tp_sl.py` all point at
`data/multi_bucket_production.json` as the production config.

---

## Locked production universe settings

These fields define the live universe contract and should not be changed during
strategy tuning:

| Field | Production value | Reason |
|---|---:|---|
| `data_source` | `daily` | Live signal generation must read the daily Yahoo cache. |
| `symbol_list` | `production` | Live signals must consume `data/production_symbols.txt`. |
| `production_symbol_status_path` | `data/production_symbol_status.csv` | Non-active symbols stay in production history but cannot open new positions. Exit signals remain observable. |
| `symbol_exclude_list` | `crypto_proxy_blocked` | Policy-blocked symbols remain observable for exits and are blocked only from new entries. |
| `ff12_data_path` | `data/production_symbols_with_sector.parquet` | Pick-N sector balancing must use production sector rows. |
| `symbol_seasoning.enabled` | `true` | Newly promoted symbols must pass the live-entry seasoning gate. |
| `symbol_seasoning.eligibility_path` | `data/production_symbol_eligibility.csv` | Missing eligibility rows fail closed. |
| `symbol_seasoning.default_new_symbol_quarantine_days` | `365` | Default helper seasons promoted symbols for 365 calendar days. |
| `symbol_seasoning.eligibility_source` | omitted / `csv` in live configs | Live cron reads the audited CSV. Historical simulations may use `price_history` to recompute first-bar quarantine from the selected data source. |
| `symbol_seasoning.quarantine_trading_bars` | `252` when `eligibility_source=price_history` | Blocks the first 252 observed price bars; combined with the 365-day gate by taking the later date. |

Do **not** point the live config at candidate symbol files. Candidate outputs
are staging inputs for audited promotion only.

`production_symbols.txt` is append-preserving: delisting, non-common-security,
and price-source failures are recorded in `production_symbol_status.csv`
instead of deleting the symbol. `active` rows may enter; `inactive` and
`price_unavailable` rows are excluded before Top-N/Pick-N selection.
`price_unavailable` rows remain in the Yahoo refresh set so they can recover.
Seasoning-ineligible rows are also removed before Top-N/Pick-N selection, so a
new or unaudited symbol cannot consume a slot that an eligible symbol should
backfill.

---

## Top-level portfolio settings

| Field | Production value |
|---|---:|
| `max_position_count` | `7` |
| `starting_cash` | `70000` |
| `margin` | `1.5` |
| `withdraw` | `0` |
| `start_date` | `2010-01-01` |
| `min_hold` | `5` |
| `show_trade_details` | `false` |
| `broker_cost_model` | `futu_hk` |

### `broker_cost_model`

Selects which broker's schedule prices the run. It sets the commission
function and the margin interest rate together, so a run is costed end to end
at one broker. Registered in `simulator.BROKER_COST_MODELS`; an unrecognised
name aborts the run rather than falling back to a default.

| Value | Schedule |
|---|---|
| `futu_hk` | Production. `0.0049`/share commission (min `0.99`) + `0.005`/share platform fee (min `1.00`) + `0.003`/share settlement; margin `4.80%` |
| `futu_hk_legacy` | `futu_hk` with the regulatory rates frozen before 2026-08-08. Reproduces result files from earlier runs; not for new work |
| `ibkr_fixed` | IBKR HK Fixed: `0.005`/share (min `1.00`, max 1% of trade value), exchange and clearing already included; margin `5.13%` |
| `ibkr_tiered` | IBKR HK Tiered: `0.0035`/share (min `0.35`) plus an assumed `0.0030`/share liquidity-removing exchange fee and `0.0002`/share clearing; margin `5.13%` |

US regulatory fees (SEC Section 31, FINRA TAF, FINRA CAT) are statutory
pass-throughs charged identically by every broker, so they sit outside the
per-broker functions and cancel in any broker-to-broker comparison.

The whole backtest window is priced with the schedule in force today, not with
a date-indexed history of past rates: a run answers "what would this system
cost under today's fees", which is the question a broker decision turns on.

To add another broker, register a `BrokerCostModel` in
`simulator.BROKER_COST_MODELS`; no other module needs to change.

Adding a broker cost model is sim-only. The live order path does not compute
commissions, so this field does not reach `multi_bucket_today` or the
dashboard.

---

## Adaptive TP/SL defaults

These defaults apply unless a bucket overrides the same behavior.

| Field | Production value |
|---|---:|
| `window` | `20` |
| `sigma` | `0.5` |
| `sl_sigma` | `1.0` |
| `min_sl` | `0.03` |
| `fixed_sl` | `null` |
| `override_min_hold_tp_only` | `true` |
| `min_hold_tp` | `1` |
| `override_min_hold_sl_only` | `true` |
| `min_hold_sl` | `1` |
| `disable_sl_trigger` | `true` |
| `delayed_rolling_update` | `true` |
| `tp_regime_adjust` | `false` |
| `tp_regime_ratio_min` | `0.5` |
| `tp_regime_ratio_max` | `2.0` |

---

## Active production buckets

All active buckets use:

```text
dollar_volume>0.02%,Top500,Pick5
```

| Label | Strategy | Priority | Max positions | Fill remaining | Max hold | Sigma | Min SL | Reset hold on re-entry |
|---|---|---:|---:|---|---:|---:|---:|---|
| `fish_head_production` | `fish_head_vacuum_turn` | `1` | `7` | `false` | default engine behavior | `0.75` | `0.01` | `false` |
| `fish_tail_squeeze` | `fish_tail_blow_off_top` | `2` | `2` | `false` | `7` | `0.75` | `0.01` | `false` |
| `fish_tail_production` | `fish_tail_blow_off_top` | `3` | `4` | `false` | `7` | `0.0` | `0.01` | `false` |
| `fish_head_b30_35` | `fish_head_b30_35` | `4` | `2` | `true` | `14` | `0.75` | `0.01` | `true` |

Additional bucket-level settings:

| Label | Setting | Production value |
|---|---|---:|
| `fish_head_production` | `exit_alpha_factor` | `3` |
| `fish_head_production` | `free_fall_slope` | `-0.2` |
| `fish_head_production` | `free_fall_near_delta` | `-0.05` |
| `fish_head_production` | `pre_cross_signal_lookback` | `true` |
| `fish_tail_squeeze` | `exit_alpha_factor` | `3` |
| `fish_tail_squeeze` | `fuel_drawdown_max` | `-0.15` |
| `fish_tail_squeeze` | `tp_regime_adjust` | `true` |
| `fish_tail_squeeze` | `tp_slope_amplify` | `true` |
| `fish_tail_production` | `exit_alpha_factor` | `3` |
| `fish_tail_production` | `tp_regime_adjust` | `true` |
| `fish_tail_production` | `tp_slope_amplify` | `true` |
| `fish_head_b30_35` | `exit_alpha_factor` | `3` |
| `fish_head_b30_35` | `free_fall_slope` | `-0.2` |
| `fish_head_b30_35` | `free_fall_near_delta` | `-0.05` |
| `fish_head_b30_35` | `pre_cross_signal_lookback` | `true` |

Buckets may also set a per-bucket `min_hold` (signal-exit minimum holding
bars). Omitted, a bucket inherits the top-level `min_hold`; when set, it
governs that bucket's signal-exit lockout window (signals firing inside the
window are permanently invalidated), the slot-eviction gate, and the
dashboard's SELL `min_hold_block` check (entry bucket recovered from the Futu
remark), and serves as the inherit base for the bucket's TP/SL min-hold
gates. Production currently sets none — all buckets inherit `min_hold: 5`.

The `fish_tail_production` label is an active production bucket label. Do not
interpret older `fish_tail_explore` log/state entries as permission to use
candidate symbol files in live production.

---

## Rolling expectancy gate

The live cron advances `data/live_state/expectancy_gate_state.json` from
dashboard-confirmed allocations. The dashboard validates the cron heartbeat,
then uses the stop for zero-capital phantom allocation. It never changes the
fixed bucket order.

| Field | Production value |
|---|---:|
| `expectancy_gate.enabled` | `true` |
| `expectancy_gate.window` | `20` |
| `expectancy_gate.baseline_mean` | `0.0087` |
| `expectancy_gate.baseline_sigma` | `0.0130` |
| `expectancy_gate.sigma_multiplier` | `3.0` |
| `expectancy_gate.cold_start` | `open` |

The stop threshold is `-0.0303`; equality is open.

| Bucket label | Fixed priority |
|---|---:|
| `fish_head_production` | `1` |
| `fish_tail_squeeze` | `2` |
| `fish_tail_production` | `3` |
| `fish_head_b30_35` | `4` |

See `Docs/expectancy_gate.md` for state ownership, causality, phantom union and
fail-closed rules.

---

## Active gates and fixed allocation

The live LLM risk-score CSV has no causal path to BUY blocking or bucket order.

The FT-WR gate is **not configured** in production. The cron does not advance
its sensor, and the dashboard cannot create an FT-WR phantom. Current
production-equivalent simulation configs omit it as well, so FT-family changes
are measured without this protection while FT performance is still under
research.

The only active dynamic gate is the global expectancy hard stop described
above. Bucket allocation remains FH=1, squeeze=2, FT=3 and B30=4 in every
regime. The caps are FH=7, squeeze=2, FT=4 and B30=2; this is the tested
Calmar-above-1 allocation promoted on 2026-08-18.

---

## Related configs are not the live contract

| Config | Status |
|---|---|
| `data/multi_bucket_production_2010_baseline.json` | Full production-equivalent baseline. It differs only by `data_source=2010_yf_clean` and explicit `broker_cost_model=futu_hk`. Run this for the current 2010 baseline. |
| `data/multi_bucket_triple_explore.json` | Candidate-universe exploration config. It uses `symbol_list=production_candidate` and candidate sector rows. Do not use it for live signals. |

Current baseline command:

```bash
multi_bucket_simulation data/multi_bucket_production_2010_baseline.json
```

---

## Safe change protocol

Before changing any production JSON:

1. Confirm whether the change is a universe change, a risk-gate change, or a
   strategy/bucket change.
2. For universe changes, do **not** edit the live JSON. Follow the official
   add-symbol policy in `Docs/universe_pipeline.md`.
3. For strategy or risk-gate changes, edit the tracked config directly, inspect
   the git diff, and update this document in the same change. Do not create
   backup copies or backup branches.
4. Verify the locked universe fields above still point at production files.
5. Run the focused tests that cover the touched area before committing.

Minimum focused checks after production-config edits:

```bash
venv/bin/python -m pytest \
  tests/test_universe_aliases.py \
  tests/test_symbol_seasoning.py \
  tests/test_live_expectancy_gate.py \
  tests/test_expectancy_gate.py \
  tests/test_multi_bucket_today_cron.py \
  tests/test_dashboard_risk_gate.py \
  tests/test_cron.py
```
