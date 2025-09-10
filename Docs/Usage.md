# Usage

The daily job reads its ticker list from `data/symbols_daily_job.txt`. Run the
`reset_symbols_daily_job` command to recreate this file by copying
`data/symbols_yf.txt` whenever the list is missing or outdated. The command
prints a confirmation or an error message.

```
reset_symbols_daily_job
```

To evaluate the FTD EMA and SMA cross strategy in the management shell, call:

```
start_simulate start=1990-01-01 dollar_volume>50 ftd_ema_sma_cross ftd_ema_sma_cross
```

To require that a symbol's 50-day average dollar volume exceeds one percent of
the market total, use:

```
start_simulate dollar_volume>1% ftd_ema_sma_cross ftd_ema_sma_cross
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

The optional `start` argument sets the simulation start date, `starting_cash`
sets the initial portfolio balance, and `withdraw` deducts a fixed amount at
each year end. The `dollar_volume` clause accepts a `>` threshold expressed in
millions or as a percentage using `%`, and an `=Nth` ranking. When both are
separated by a comma, the parser applies them sequentially. A trailing
`PickM` segment allows up to `M` symbols from any single sector when a ranking
filter is used. The command above first filters symbols to those whose 50-day
average dollar volume exceeds 10,000 million and then selects the six symbols
with the highest remaining averages. The tests
`tests/test_manage.py::test_start_simulate_dollar_volume_threshold_and_rank` and
`tests/test_strategy.py::test_evaluate_combined_strategy_dollar_volume_filter_and_rank`
exercise this combined syntax.

A numeric stop loss and a trailing `True` or `False` flag may follow the
strategy names. The stop loss specifies the fractional decline that triggers an
exit on the next day's open, and the boolean flag controls whether individual
trade details are printed.

The simulation report lists the maximum drawdown alongside other metrics. This
percentage indicates the greatest decline from any previous portfolio peak.

The previous `start_ftd_ema_sma_cross` command has been removed.
Use `start_simulate` with `ftd_ema_sma_cross` for both the buying and
selling strategies instead.

## Recalculating signals

Each execution of the daily job records entry and exit signals in a log file in
the project's `logs` directory using the `<YYYY-MM-DD>.log` naming convention.
The filename reflects the **signal date**, and the log message notes the
corresponding **trade date** on which those signals will execute. The
`find_history_signal` command recalculates the signals for a specific date
rather than reading the log files. It reports the signals generated on the
supplied date without shifting them to the following trading day. Trading based
on those signals still occurs at the next trading day's open. Signal calculation
uses the same group dynamic ratio and Top-N rule as `start_simulate`.
The management shell can compute signals for a specific day with either form:

```
find_history_signal DATE DOLLAR_VOLUME_FILTER BUY_STRATEGY SELL_STRATEGY STOP_LOSS
find_history_signal DATE DOLLAR_VOLUME_FILTER STOP_LOSS strategy=ID
```

Accepted forms are `BUY SELL STOP_LOSS` or `STOP_LOSS strategy=ID`. The command
prints the entry signal list on the first line, the exit signal list on the
second line, and when portfolio status is available, a mapping of suggested
entry budgets on the third line. For example:

```
find_history_signal 2024-01-10 dollar_volume>1 1.0 strategy=default
entry signals: ['AAA', 'BBB']
exit signals: ['CCC', 'DDD']
budget suggestions: {'AAA': 500.0, 'BBB': 500.0}
```
A daily job run for the same signal date writes a log entry similar to:

```
Starting daily tasks for trade date 2024-01-11 using signals from 2024-01-10
```

In contrast, simulation commands operate on trade days. A signal produced on
`2024-01-10` triggers a simulated trade on `2024-01-11`, while
`find_history_signal 2024-01-10 ...` reports the signals for `2024-01-10`
itself. The daily job's log entry clarifies this relationship by recording both
dates, helping separate signal-day lookups from trade-day executions.

Developers may call
`daily_job.find_history_signal("2024-01-10", "dollar_volume>1", "ema_sma_cross", "ema_sma_cross", 1.0)`
to compute the same values from Python code. Passing ``None`` as the first
argument evaluates the most recent trading day. The function returns the entry
and exit signal lists along with the budget information when available, rather
than reading log files.

To refresh data for all cached symbols and compute today's signals, run
`find_history_signal` without a date. The command accepts the same argument
forms:

```
find_history_signal DOLLAR_VOLUME_FILTER BUY_STRATEGY SELL_STRATEGY STOP_LOSS
find_history_signal DOLLAR_VOLUME_FILTER STOP_LOSS strategy=ID
```

Example:

```
find_history_signal dollar_volume>1 ema_sma_cross ema_sma_cross 1.0
entry signals: ['AAA']
exit signals: ['BBB']
budget suggestions: {'AAA': 500.0}
```

## Available strategies

The `start_simulate` command accepts the following strategies:

* `ema_sma_cross`
* `20_50_sma_cross`
* `ema_sma_cross_and_rsi`
* `ftd_ema_sma_cross`
* `ema_sma_cross_with_slope` *(use `ema_sma_cross_with_slope_N` to set a custom EMA/SMA window size; `N` defaults to 40. Append `_LOWER_UPPER` to constrain the simple moving average angle to a range in degrees.)*
* `ema_sma_cross_with_slope_and_volume`
* `ema_sma_cross_testing`
* `ema_sma_double_cross`
* `kalman_filtering` *(sell only)*

To change the EMA and SMA window size, append `_N` to `ema_sma_cross_with_slope`, where `N` sets the number of days and defaults to `40`:

```
start_simulate dollar_volume>1 ema_sma_cross_with_slope_40 ema_sma_cross_with_slope_40
```

To limit the angle of the simple moving average, add two numeric bounds after the optional window size. Both bounds may be negative or positive floating-point numbers representing degrees. These strategies follow the generic `ema_sma_signal_with_slope_n_k` pattern and use the format `ema_sma_cross_with_slope[_N]_LOWER_UPPER`:

```
start_simulate dollar_volume>1 ema_sma_cross_with_slope_-5.7_50.2 ema_sma_cross_with_slope_-5.7_50.2
```

The window size and slope range can be combined by placing the integer before the slope bounds:

```
start_simulate dollar_volume>1 ema_sma_cross_with_slope_40_-5.7_50.2 ema_sma_cross_with_slope_40_-5.7_50.2
```

The testing variant `ema_sma_cross_testing` accepts the same optional window
size and slope range suffixes. In addition to the EMA/SMA cross and slope
filters, it recalculates chip concentration metrics. Strategy names follow
`ema_sma_cross_testing_<window>_<lower>_<upper>_<near>_<above>`:

* `<window>` — EMA and SMA window size (default `40`).
* `<lower>` and `<upper>` — inclusive simple moving average angle bounds in degrees (defaults `-16.7` and `65`).
* `<near>` — maximum allowed near-price volume ratio (default `0.12`).
* `<above>` — maximum allowed above-price volume ratio (default `0.10`).

Example with custom chip concentration thresholds:

```
start_simulate dollar_volume>1 ema_sma_cross_testing_40_-16.7_65_0.2_0.15 ema_sma_cross_testing_40_-16.7_65_0.2_0.15
```

The tests `tests/test_manage.py::test_start_simulate_accepts_slope_range_strategy_names` and `tests/test_strategy.py::test_evaluate_combined_strategy_passes_slope_range` confirm this behavior. The management test shows the command accepts strategy names with slope bounds, and the strategy test verifies that the evaluation routine passes the bounds to the strategy implementation.

Not every strategy supports both buying and selling. Only the first seven
strategies in the list can be used for buying. All eight strategies can be
used for selling.
