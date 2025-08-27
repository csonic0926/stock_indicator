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
start_simulate starting_cash=5000 withdraw=1000 dollar_volume>10000,6th ftd_ema_sma_cross ftd_ema_sma_cross
```

The optional `starting_cash` argument sets the initial portfolio balance, and
`withdraw` deducts a fixed amount at each year end. The `dollar_volume` clause
accepts a `>` threshold and an `=Nth` ranking. When both are separated by a
comma, the parser applies them sequentially. The command above first filters
symbols to those whose 50-day average dollar volume exceeds 10,000 million and
then selects the six symbols with the highest remaining averages. The tests
`tests/test_manage.py::test_start_simulate_dollar_volume_threshold_and_rank` and
`tests/test_strategy.py::test_evaluate_combined_strategy_dollar_volume_filter_and_rank`
exercise this combined syntax.

The previous `start_ftd_ema_sma_cross` command has been removed.
Use `start_simulate` with `ftd_ema_sma_cross` for both the buying and
selling strategies instead.

## Recalculating signals

Each execution of the daily job records entry and exit signals in a log file in
the project's `logs` directory using the `<YYYY-MM-DD>.log` naming convention.
The `find_signal` command recalculates the signals for a specific date rather
than reading the log files. The management shell can compute signals for a
specific day with:

```
find_signal DATE DOLLAR_VOLUME_FILTER BUY_STRATEGY SELL_STRATEGY STOP_LOSS
```

The command prints the entry signal list on the first line and the exit signal
list on the second line. For example:

```
find_signal 2024-01-10 dollar_volume>1 ema_sma_cross ema_sma_cross 1.0
['AAA', 'BBB']
['CCC', 'DDD']
```

Developers may call `daily_job.find_signal("2024-01-10", "dollar_volume>1", "ema_sma_cross", "ema_sma_cross", 1.0)` to compute
the same values from Python code. This function also recalculates signals
instead of reading log files.

## Available strategies

The `start_simulate` command accepts the following strategies:

* `ema_sma_cross`
* `20_50_sma_cross`
* `ema_sma_cross_and_rsi`
* `ftd_ema_sma_cross`
* `ema_sma_cross_with_slope`
* `ema_sma_cross_with_slope_and_volume`
* `ema_sma_double_cross`
* `kalman_filtering` *(sell only)*

Not every strategy supports both buying and selling. Only the first seven
strategies in the list can be used for buying. All eight strategies can be
used for selling.
