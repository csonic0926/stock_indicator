# Production Runtime Configuration Contract

This document records the current production JSON contract so the live config
does not drift through accidental `.json` edits.

Last aligned with `data/multi_bucket_production.json`: 2026-05-28.

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
| `ff12_data_path` | `data/production_symbols_with_sector.parquet` | Pick-N sector balancing must use production sector rows. |
| `symbol_seasoning.enabled` | `true` | Newly promoted symbols must pass the live-entry seasoning gate. |
| `symbol_seasoning.eligibility_path` | `data/production_symbol_eligibility.csv` | Missing eligibility rows fail closed. |
| `symbol_seasoning.default_new_symbol_quarantine_days` | `365` | Default helper seasons promoted symbols for 365 calendar days. |
| `symbol_seasoning.eligibility_source` | omitted / `csv` in live configs | Live cron reads the audited CSV. Historical simulations may use `price_history` to recompute first-bar quarantine from the selected data source. |
| `symbol_seasoning.quarantine_trading_bars` | `252` when `eligibility_source=price_history` | Blocks the first 252 observed price bars; combined with the 365-day gate by taking the later date. |

Do **not** point the live config at candidate symbol files. Candidate outputs
are staging inputs for audited promotion only.

---

## Top-level portfolio settings

| Field | Production value |
|---|---:|
| `max_position_count` | `6` |
| `starting_cash` | `60000` |
| `margin` | `1.5` |
| `withdraw` | `0` |
| `start_date` | `2010-01-01` |
| `min_hold` | `5` |
| `show_trade_details` | `false` |

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
| `fish_head_production` | `fish_head_vacuum_turn` | `1` | `6` | `false` | default engine behavior | `0.75` | `0.01` | `false` |
| `fish_tail_explore` | `fish_tail_blow_off_top` | `1` | `6` | `false` | `7` | `0.0` | `0.01` | `false` |
| `fish_head_b30_35` | `fish_head_b30_35` | `2` | `2` | `true` | `14` | `0.75` | `0.01` | `true` |

Additional bucket-level settings:

| Label | Setting | Production value |
|---|---|---:|
| `fish_head_production` | `exit_alpha_factor` | `3` |
| `fish_head_production` | `free_fall_slope` | `-0.2` |
| `fish_head_production` | `free_fall_near_delta` | `-0.05` |
| `fish_head_production` | `pre_cross_signal_lookback` | `true` |
| `fish_tail_explore` | `exit_alpha_factor` | `3` |
| `fish_tail_explore` | `tp_regime_adjust` | `true` |
| `fish_tail_explore` | `tp_slope_amplify` | `true` |
| `fish_head_b30_35` | `exit_alpha_factor` | `3` |
| `fish_head_b30_35` | `free_fall_slope` | `-0.2` |
| `fish_head_b30_35` | `free_fall_near_delta` | `-0.05` |
| `fish_head_b30_35` | `pre_cross_signal_lookback` | `true` |

The `fish_tail_explore` label is currently an active production bucket label.
Do not interpret that label as permission to use candidate symbol files in live
production.

---

## Risk score gate

| Field | Production value |
|---|---:|
| `risk_score_gate.csv_path` | `data/historical_risk_scores.csv` |
| `risk_score_gate.stop_threshold` | `75` |

Priority overrides are active for risk scores `25` and `50`:

| Bucket label | Override priority |
|---|---:|
| `fish_head_production` | `1` |
| `fish_tail_explore` | `2` |
| `fish_head_b30_35` | `3` |

---

## Related configs are not the live contract

| Config | Status |
|---|---|
| `data/multi_bucket_production_test.json` | Test/backtest helper. It uses `data_source=2010`, the production symbol list, and only the first two buckets. |
| `data/multi_bucket_triple_explore.json` | Candidate-universe exploration config. It uses `symbol_list=production_candidate` and candidate sector rows. Do not use it for live signals. |
| Historical max-hold variant configs | Scenario files only. Inspect their JSON fields before use; the filename alone is not a production contract. |

---

## Safe change protocol

Before changing any production JSON:

1. Confirm whether the change is a universe change, a risk-gate change, or a
   strategy/bucket change.
2. For universe changes, do **not** edit the live JSON. Follow the official
   add-symbol policy in `Docs/universe_pipeline.md`.
3. For strategy or risk-gate changes, edit a draft copy first, compare the diff,
   and update this document in the same change.
4. Verify the locked universe fields above still point at production files.
5. Run the focused tests that cover the touched area before committing.

Minimum focused checks after production-config edits:

```bash
venv/bin/python -m pytest \
  tests/test_universe_aliases.py \
  tests/test_symbol_seasoning.py \
  tests/test_multi_bucket_today_cron.py \
  tests/test_cron.py
```
