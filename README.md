# Stock Indicator

## Project Description
Stock Indicator provides a collection of Python utilities for computing common technical indicators used in stock market analysis. The goal is to make it easy to calculate metrics such as moving averages, RSI, and MACD on historical price data.

## Quick Start

### Requirements
- Python 3.10+
- Packages: `pandas`, `numpy`, `yfinance`, `matplotlib`
- Internet connection for downloading market data from providers like [Yahoo Finance](https://finance.yahoo.com) or [Alpha Vantage](https://www.alphavantage.co/)

### Installation
```bash
git clone https://github.com/yourusername/stock_indicator.git
cd stock_indicator
pip install pandas numpy yfinance matplotlib
```

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
style.

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
downloading historical price data.

```bash
python -m stock_indicator.manage

(stock-indicator) update_symbols
(stock-indicator) update_yf_symbols
(stock-indicator) update_data_from_yf AAPL 2024-01-01 2024-02-01
(stock-indicator) update_all_data_from_yf 2024-01-01 2024-02-01
(stock-indicator) exit
```

* `update_symbols` downloads the latest list of available ticker symbols from the SEC `company_tickers.json` dataset (via the sector pipeline integration) and writes `data/symbols.txt`.
* `update_yf_symbols` probes Yahoo Finance for a small recent window and writes the subset of tickers that return data to `data/symbols_yf.txt`. Daily jobs require this list (no SEC fallback).
* `update_data_from_yf SYMBOL START END` saves historical data for the given symbol to
  `data/<SYMBOL>.csv`.
* `update_all_data_from_yf START END` performs the download for every cached symbol.
* `find_signal DATE DOLLAR_VOLUME_FILTER STOP_LOSS strategy=ID`
  recalculates the entry and exit signals for `DATE`. Accepted forms are
  `BUY SELL STOP_LOSS` or `STOP_LOSS strategy=ID`. The first form supplies
  explicit buy and sell strategy names, while the second references a strategy
  set identifier (see Strategy Sets below). Signal calculation uses the same
  group dynamic ratio and Top-N rule as `start_simulate`.

For example:

```bash
(stock-indicator) find_signal 2024-01-10 dollar_volume>1 1.0 strategy=default
['AAA', 'BBB']
['CCC', 'DDD']
```

Developers can also call `daily_job.find_signal("2024-01-10", "dollar_volume>1", "ema_sma_cross", "ema_sma_cross", 1.0)` to compute
the same data from Python code. This function recalculates signals rather than
reading them from log files. Signal calculation uses the same group dynamic ratio and Top-N rule as `start_simulate`.

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

An optional stop loss value and a trailing `True` or `False` flag may be added
after the strategy names. The numeric stop loss sets the fractional decline
that triggers an exit on the next day's open, and the boolean flag controls
whether individual trade details are printed.

Strategies may also limit the simple moving average slope. These identifiers follow the `ema_sma_signal_with_slope_n_k` pattern where `n` and `k` are the lower and upper slope bounds. The bounds accept negative or positive floating-point numbers. For example:

```bash
(stock-indicator) start_simulate dollar_volume>1 ema_sma_cross_with_slope_-0.1_1.2 ema_sma_cross_with_slope_-0.1_1.2
```

You can combine slope bounds with a custom EMA/SMA window size by placing the integer before the bounds:

```bash
(stock-indicator) start_simulate dollar_volume>1 ema_sma_cross_with_slope_40_-0.1_1.2 ema_sma_cross_with_slope_40_-0.1_1.2
```

For experimentation, the `ema_sma_cross_testing` strategy offers the same
optional window size and slope range suffixes. It omits the long-term simple
moving average requirement and additionally filters signals using chip
concentration metrics, enforcing default thresholds of 0.12 for the near-price
volume ratio and 0.10 for the above-price volume ratio.

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
  - `find_signal DATE DOLLAR_VOLUME_FILTER STOP_LOSS strategy=ID [group=1,2,...]` — signal calculation uses the same group dynamic ratio and Top-N rule as `start_simulate`.

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
