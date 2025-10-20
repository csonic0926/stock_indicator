# Stock Indicator

## Project Description
Stock Indicator provides a collection of Python utilities for computing common technical indicators used in stock market analysis. The goal is to make it easy to calculate metrics such as moving averages, RSI, and MACD on historical price data.

## Quick Start

### Requirements
- Python 3.10+
- Internet connection for downloading market data from providers like [Yahoo Finance](https://finance.yahoo.com) or [Alpha Vantage](https://www.alphavantage.co/)
- Python packages listed in [`requirements.txt`](./requirements.txt)

### Installation
```bash
git clone https://github.com/yourusername/stock_indicator.git
cd stock_indicator
python -m venv venv
source venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

The requirements file installs the project's runtime dependencies, including
`pandas`, `numpy`, `yfinance`, and other libraries used by the CLI utilities.

### Example Usage
```python
from stock_indicator.data_loader import download_history
from stock_indicator.indicators import rsi

prices = download_history("AAPL", "2023-01-01", "2023-06-01")
prices["rsi_14"] = rsi(prices["close"], window=14)
print(prices[["close", "rsi_14"]].tail())
```

Downloaded data frames use lower-case ``snake_case`` column names. With
``yfinance`` version ``0.2.51`` and later, the ``close`` column already reflects
dividends and stock splits, so no separate adjusted closing price column is
provided. Downstream code should refer to columns using this standardized
style. The ``load_price_data`` helper applies the same normalization to CSV
files by converting headers to lowercase ``snake_case`` and stripping common
suffixes such as ``_price``. If multiple headers normalize to the same label,
only the first is preserved.

For example, a file with the header ``Date,Open,Close,Adj Close,Close Price``
loads as:

```python
from pathlib import Path
from stock_indicator.strategy import load_price_data

frame = load_price_data(Path("prices.csv"))
print(frame.columns)
# Index(['open', 'close', 'adj_close'], dtype='object')
```

Both ``Close`` and ``Close Price`` map to ``close``, so ``Close Price`` is
discarded.

### Command Line Example
Stock Indicator also includes a command line interface for generating trade signals from historical price data.

```bash
python -m stock_indicator.cli --symbol AAPL --start 2023-01-01 --end 2023-06-01 --strategy sma --output trades.csv
```

* `--symbol` — ticker symbol of the stock to analyze.
* `--start` — start date for the price history in `YYYY-MM-DD` format.
* `--end` — end date for the price history in `YYYY-MM-DD` format.
* `--strategy` — indicator or strategy to apply, such as `sma` for simple moving average.
* `--output` — file path for saving generated trades as a CSV file.

### Management Shell

The package provides an interactive shell for updating the symbol cache and
downloading historical price data. Daily tasks now build their ticker list
directly from the Yahoo Finance cache maintained by the shell commands, so no
manual text file management is required.

```bash
python -m stock_indicator.manage

(stock-indicator) update_symbols
(stock-indicator) update_data_from_yf AAPL 2024-01-01 2024-02-01
(stock-indicator) update_all_data_from_yf 2024-01-01 2024-02-01
(stock-indicator) exit
```

* `update_symbols` downloads the latest list of available ticker symbols from the SEC `company_tickers.json` dataset (via the sector pipeline integration) and writes `data/symbols.txt`.
* `update_data_from_yf SYMBOL START END` saves historical data for the given symbol to
  `data/<SYMBOL>.csv`. ``END`` is inclusive.
* `update_all_data_from_yf START END` performs the download for every cached symbol.
  ``END`` is inclusive.
* `find_history_signal [DATE] DOLLAR_VOLUME_FILTER BUY_STRATEGY SELL_STRATEGY
  STOP_LOSS` or `find_history_signal [DATE] DOLLAR_VOLUME_FILTER STOP_LOSS
  strategy=ID` recalculates the entry and exit signals for `DATE`. The first
  form supplies explicit buy and sell strategy names, while the second
  references a strategy set identifier (see Strategy Sets below). Omitting
  `DATE` refreshes data for all cached symbols and evaluates the most recent
  trading day. The command now reports the signals generated on the supplied
  day without shifting them to the next trading day. Trades based on those
  signals still execute at the following day's open. Signal calculation uses
  the same group dynamic ratio and Top-N rule as `start_simulate`.

For example:

```bash
(stock-indicator) find_history_signal 2024-01-10 dollar_volume>1 1.0 strategy=default
entry signals: ['AAA', 'BBB']
exit signals: ['CCC', 'DDD']
budget suggestions: {'AAA': 500.0, 'BBB': 500.0}
```

Simulation commands interpret dates as trade days. A signal generated on
`2024-01-10` leads to an order on `2024-01-11` when running `start_simulate`,
whereas `find_history_signal 2024-01-10 ...` lists the signals produced on
`2024-01-10` itself. This example contrasts the two modes:

```bash
(stock-indicator) start_simulate start=2024-01-10 dollar_volume>1 ema_sma_cross ema_sma_cross
# trades execute on 2024-01-11
(stock-indicator) find_history_signal 2024-01-10 dollar_volume>1 ema_sma_cross ema_sma_cross 1.0
entry signals: ['AAA']
exit signals: ['BBB']
```

To refresh data for all cached symbols and compute today's signals, run
`find_history_signal` without a date:

```bash
(stock-indicator) find_history_signal dollar_volume>1 ema_sma_cross ema_sma_cross 1.0
entry signals: ['AAA']
exit signals: ['BBB']
budget suggestions: {'AAA': 500.0}
```

Developers can also call `daily_job.find_history_signal("2024-01-10", "dollar_volume>1", "ema_sma_cross", "ema_sma_cross", 1.0)` to compute
the same data from Python code. This function recalculates signals rather than
reading them from log files. Passing ``None`` as the first argument evaluates
the most recent trading day. Signal calculation uses the same group dynamic
ratio and Top-N rule as `start_simulate`.

The shell can also simulate trading strategies. The `dollar_volume` filter
accepts a minimum threshold in millions, a percentage of total market volume,
and a ranking when combined with a comma. The command below evaluates
`ftd_ema_sma_cross` using only the six symbols whose 50-day average dollar
volume exceeds 10,000 million:

```bash
(stock-indicator) start_simulate starting_cash=5000 withdraw=1000 dollar_volume>10000,6th strategy=default 1.0 true
```

Here `dollar_volume>10000,6th` first drops symbols below the threshold and then
selects the six highest-volume symbols from the remainder. The tests
`tests/test_manage.py::test_start_simulate_dollar_volume_threshold_and_rank` and
`tests/test_strategy.py::test_evaluate_combined_strategy_dollar_volume_filter_and_rank`
demonstrate this combined syntax.

The filter optionally accepts a trailing `PickM` segment to allow multiple
symbols from the same sector when a ranking is used. For example
`dollar_volume>1%,Top3,Pick2` selects the top three symbols by dollar volume
with at most two symbols drawn from any single Fama–French group.

An optional stop loss value and a trailing `True` or `False` flag may be added
after the strategy names. The numeric stop loss sets the fractional decline
that triggers an exit on the next day's open, and the boolean flag controls
whether individual trade details are printed.

Strategies may also limit the simple moving average slope. These identifiers follow the `ema_sma_signal_with_slope_n_k` pattern where `n` and `k` are the lower and upper slope bounds. The bounds accept negative or positive floating-point numbers. For example:

```bash
(stock-indicator) start_simulate dollar_volume>1 ema_sma_cross_with_slope_-5.7_50.2 ema_sma_cross_with_slope_-5.7_50.2
```

You can combine slope bounds with a custom EMA/SMA window size by placing the integer before the bounds:

```bash
(stock-indicator) start_simulate dollar_volume>1 ema_sma_cross_with_slope_40_-5.7_50.2 ema_sma_cross_with_slope_40_-5.7_50.2
```

For experimentation, the `ema_sma_cross_testing` strategy offers the same
optional window size and slope range suffixes. It omits the long-term simple
moving average requirement and additionally filters signals using chip
concentration metrics. Strategy names follow the pattern
`ema_sma_cross_testing_<window>_<lower>_<upper>_<near_min>,<near_max>_<above_min>,<above_max>`, where:

* `<window>` — EMA and SMA window size (default `40`).
* `<lower>` and `<upper>` — inclusive bounds for the simple moving average angle in degrees (defaults `-16.7` and `65`).
* `<near_min>` and `<near_max>` — inclusive bounds for the fraction of volume near the current price (default `0.0`–`0.12`).
* `<above_min>` and `<above_max>` — inclusive bounds for the fraction of volume above the current price (default `0.0`–`0.10`).

Example with custom chip concentration ranges:

```bash
(stock-indicator) start_simulate dollar_volume>1 \
  ema_sma_cross_testing_40_-16.7_65_0.1,0.2_0.05,0.15 \
  ema_sma_cross_testing_40_-16.7_65_0.1,0.2_0.05,0.15
```

When omitted, the window size defaults to 40 days.

### Strategy Sets

Define named strategy pairs in `data/strategy_sets.csv` and reference them
with `strategy=ID`.

- CSV columns: `strategy_id,buy,sell`
- Example:

```
strategy_id,buy,sell
default,ema_sma_cross_with_slope_40,ema_sma_cross_with_slope_50
s1,ema_shift_cross_with_slope_35,ema_shift_cross_with_slope_35
```

Shorthand forms using a strategy id (omit explicit buy/sell tokens):

- Start simulate
  - `start_simulate [starting_cash=...] [withdraw=...] [start=YYYY-MM-DD] [margin=NUMBER] DOLLAR_VOLUME_FILTER [STOP_LOSS] [SHOW_DETAILS] strategy=ID [group=1,2,...]`
- Single symbol
  - `start_simulate_single_symbol symbol=QQQ [starting_cash=...] [withdraw=...] [start=YYYY-MM-DD] [STOP_LOSS] [SHOW_DETAILS] strategy=ID`
- N symbols
  - `start_simulate_n_symbol symbols=QQQ,SPY [starting_cash=...] [withdraw=...] [start=YYYY-MM-DD] [STOP_LOSS] [SHOW_DETAILS] strategy=ID`
- Find signal
- `find_history_signal DATE DOLLAR_VOLUME_FILTER STOP_LOSS strategy=ID [group=1,2,...]` — signal calculation uses the same group dynamic ratio and Top-N rule as `start_simulate`.

Notes:
- When `strategy=ID` is present, do not include explicit `BUY`/`SELL` tokens.
- Strategy ids are read from `data/strategy_sets.csv` at runtime.

The tests `tests/test_manage.py::test_start_simulate_accepts_slope_range_strategy_names` and `tests/test_strategy.py::test_evaluate_combined_strategy_passes_slope_range` demonstrate the slope-bound syntax. The former shows that `start_simulate` recognizes strategy identifiers with slope ranges, while the latter verifies that the evaluation function passes the provided bounds to the strategy implementation.

The summary printed after each simulation includes the maximum drawdown. This
value represents the largest peak-to-trough decline in portfolio value over the
test period and is expressed as a percentage.

To express the threshold as a percentage of total market dollar volume, use a
percent sign. For example `dollar_volume>1%` retains only symbols whose
50-day average dollar volume is greater than one percent of the combined
volume across all symbols.

## Sector Classification Pipeline

The module `stock_indicator.sector_pipeline.pipeline` creates a data set that
links ticker symbols to Standard Industrial Classification codes and
Fama-French industry groups. Each run of
`build_sector_classification_dataset` caches SEC submission data in
`cache/submissions` and records its configuration in `cache/last_run.json`.
Calling `update_latest_dataset` later reloads that configuration and rebuilds
the output while reusing the cached submissions, so only new symbols require
additional downloads. If no prior configuration exists, the pipeline falls back
to the bundled default mapping at `data/sic_to_ff.csv` so it runs out of the box.
The resulting table is written to
`data/symbols_with_sector.parquet` and can also be exported as a CSV file.
The ticker universe is always derived from the SEC `company_tickers.json`
dataset and cannot be overridden with a custom list.

## Contribution Guidelines
1. Fork the repository and create a new branch for each feature or bug fix.
2. Ensure your code passes all tests by running `pytest` before submitting.
3. Open a pull request with a clear description of your changes.

This project is released under the MIT License. By contributing, you agree to license your work under the same terms.
