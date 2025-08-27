# Usage

To evaluate the FTD EMA and SMA cross strategy in the management shell, call:

```
start_simulate start=1990-01-01 dollar_volume>2.4%,-0.2% ftd_ema_sma_cross ftd_ema_sma_cross
```

To require that a symbol's 50-day average dollar volume exceeds one percent of
the market total with a gradual reduction for earlier years, use:

```
start_simulate dollar_volume>1%,-0.2% ftd_ema_sma_cross ftd_ema_sma_cross
```

The optional `start` argument sets the simulation start date, `starting_cash`
sets the initial portfolio balance, and `withdraw` deducts a fixed amount at
each year end. The `dollar_volume` clause accepts a base percentage followed by
an adjustment applied every five years before 2021. For example,
`dollar_volume>2.4%,-0.2%` uses a 2.4% threshold from 2021–2025 and reduces it
to 2.2% for 2016–2020, 2.0% for 2011–2015, and so on.

The simulation report lists the maximum drawdown alongside other metrics. This
percentage indicates the greatest decline from any previous portfolio peak.

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
find_signal 2024-01-10 dollar_volume>1%,-0.2% ema_sma_cross ema_sma_cross 1.0
['AAA', 'BBB']
['CCC', 'DDD']
```

Developers may call `daily_job.find_signal("2024-01-10", "dollar_volume>1%,-0.2%", "ema_sma_cross", "ema_sma_cross", 1.0)` to compute
the same values from Python code. This function also recalculates signals
instead of reading log files.

## Available strategies

The `start_simulate` command accepts the following strategies:

* `ema_sma_cross`
* `20_50_sma_cross`
* `ema_sma_cross_and_rsi`
* `ftd_ema_sma_cross`
* `ema_sma_cross_with_slope` *(use `ema_sma_cross_with_slope_N` to set a custom EMA/SMA window size; `N` defaults to 40. Append `_LOWER_UPPER` to constrain the simple moving average slope to a range.)*
* `ema_sma_cross_with_slope_and_volume`
* `ema_sma_double_cross`
* `kalman_filtering` *(sell only)*

To change the EMA and SMA window size, append `_N` to `ema_sma_cross_with_slope`, where `N` sets the number of days and defaults to `40`:

```
start_simulate dollar_volume>1%,-0.2% ema_sma_cross_with_slope_40 ema_sma_cross_with_slope_40
```

To limit the slope of the simple moving average, add two numeric bounds after the optional window size. Both bounds may be negative or positive floating-point numbers. These strategies follow the generic `ema_sma_signal_with_slope_n_k` pattern and use the format `ema_sma_cross_with_slope[_N]_LOWER_UPPER`:

```
start_simulate dollar_volume>1%,-0.2% ema_sma_cross_with_slope_-0.1_1.2 ema_sma_cross_with_slope_-0.1_1.2
```

The window size and slope range can be combined by placing the integer before the slope bounds:

```
start_simulate dollar_volume>1%,-0.2% ema_sma_cross_with_slope_40_-0.1_1.2 ema_sma_cross_with_slope_40_-0.1_1.2
```

The tests `tests/test_manage.py::test_start_simulate_accepts_slope_range_strategy_names` and `tests/test_strategy.py::test_evaluate_combined_strategy_passes_slope_range` confirm this behavior. The management test shows the command accepts strategy names with slope bounds, and the strategy test verifies that the evaluation routine passes the bounds to the strategy implementation.

Not every strategy supports both buying and selling. Only the first seven
strategies in the list can be used for buying. All eight strategies can be
used for selling.
