# Usage

To evaluate the FTD EMA and SMA cross strategy in the management shell, call:

```
start_simulate dollar_volume>50 ftd_ema_sma_cross ftd_ema_sma_cross
```

To restrict simulation to the six symbols with the highest 50-day average dollar
volume, use:

```
start_simulate dollar_volume=6th ftd_ema_sma_cross ftd_ema_sma_cross
```

To apply both a minimum dollar volume and a ranking filter, combine them:

```
start_simulate dollar_volume>10000,6th ftd_ema_sma_cross ftd_ema_sma_cross
```

The `dollar_volume` clause accepts a `>` threshold and an `=Nth` ranking. When both
are separated by a comma, the parser applies them sequentially. The command
above first filters symbols to those whose 50-day average dollar volume exceeds
10,000 million and then selects the six symbols with the highest remaining
averages. The tests `tests/test_manage.py::test_start_simulate_dollar_volume_threshold_and_rank`
and `tests/test_strategy.py::test_evaluate_combined_strategy_dollar_volume_filter_and_rank`
exercise this combined syntax.

The previous `start_ftd_ema_sma_cross` command has been removed.
Use `start_simulate` with `ftd_ema_sma_cross` for both the buying and
selling strategies instead.

## Available strategies

The `start_simulate` command accepts the following strategies:

* `ema_sma_cross`
* `20_50_sma_cross`
* `ema_sma_cross_and_rsi`
* `ftd_ema_sma_cross`
* `ema_sma_cross_with_slope`
* `ema_sma_double_cross`
* `kalman_filtering` *(sell only)*

Not every strategy supports both buying and selling. Only the first six
strategies in the list can be used for buying. All seven strategies can be
used for selling.

## Annual withdrawal simulation

To run a simulation that withdraws a fixed amount of cash at the end of each
year, use the `start_simulate_withdraw` command:

```
start_simulate_withdraw=5000,1000
```

The first number specifies the starting cash and the second number is the cash
withdrawn at each year end. The command evaluates the default `ema_sma_cross`
strategy for both entry and exit signals and reports the resulting metrics.
