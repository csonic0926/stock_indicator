"""Tests for the download_history utility."""
# TODO: review

import logging
import os
import sys
import types
import pathlib

import pandas
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Create a minimal yfinance stub before importing the module under test
fake_yfinance_module = types.ModuleType("yfinance")
fake_yfinance_module.download = lambda *args, **kwargs: pandas.DataFrame()
sys.modules["yfinance"] = fake_yfinance_module

from stock_indicator.data_loader import download_history


def test_download_history_returns_dataframe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The function should adjust price and volume data provided by yfinance."""
    raw_dataframe = pandas.DataFrame(
        {
            "Open": [100.0, 200.0],
            "High": [110.0, 210.0],
            "Low": [90.0, 190.0],
            "Close": [100.0, 200.0],
            "Adj Close": [50.0, 100.0],
            "Volume": [1000.0, 2000.0],
        }
    )

    def stubbed_download(
        symbol: str,
        start: str,
        end: str,
        progress: bool = False,
        auto_adjust: bool = False,
    ) -> pandas.DataFrame:
        return raw_dataframe

    monkeypatch.setattr(
        "stock_indicator.data_loader.yfinance.download", stubbed_download
    )
    monkeypatch.setattr("stock_indicator.symbols.load_symbols", lambda: ["TEST"])
    result_dataframe = download_history("TEST", "2021-01-01", "2021-01-02")

    adjustment_ratio = raw_dataframe["Adj Close"] / raw_dataframe["Close"]
    expected_dataframe = pandas.DataFrame(
        {
            "open": raw_dataframe["Open"] * adjustment_ratio,
            "high": raw_dataframe["High"] * adjustment_ratio,
            "low": raw_dataframe["Low"] * adjustment_ratio,
            "close": raw_dataframe["Close"] * adjustment_ratio,
            "volume": raw_dataframe["Volume"] / adjustment_ratio,
        }
    )
    pandas.testing.assert_frame_equal(result_dataframe, expected_dataframe)


def test_download_history_flattens_multiindex_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The function should flatten MultiIndex columns and normalize their names."""
    raw_dataframe = pandas.DataFrame(
        {
            ("Open", "TEST"): [100.0],
            ("High", "TEST"): [110.0],
            ("Low", "TEST"): [90.0],
            ("Close", "TEST"): [100.0],
            ("Adj Close", "TEST"): [50.0],
            ("Volume", "TEST"): [1000.0],
        }
    )

    def stubbed_download(
        symbol: str,
        start: str,
        end: str,
        progress: bool = False,
        auto_adjust: bool = False,
    ) -> pandas.DataFrame:
        return raw_dataframe

    monkeypatch.setattr(
        "stock_indicator.data_loader.yfinance.download", stubbed_download
    )
    monkeypatch.setattr("stock_indicator.symbols.load_symbols", lambda: ["TEST"])
    result_dataframe = download_history("TEST", "2021-01-01", "2021-01-02")
    expected_frame = pandas.DataFrame(
        {
            "open": [50.0],
            "high": [55.0],
            "low": [45.0],
            "close": [50.0],
            "volume": [2000.0],
        }
    )
    pandas.testing.assert_frame_equal(result_dataframe, expected_frame)


def test_download_history_retries_on_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The function should retry and log warnings on temporary failures."""
    call_counter = {"count": 0}
    raw_dataframe = pandas.DataFrame(
        {
            "Open": [100.0],
            "High": [110.0],
            "Low": [90.0],
            "Close": [100.0],
            "Adj Close": [50.0],
            "Volume": [1000.0],
        }
    )

    def flaky_download(
        symbol: str,
        start: str,
        end: str,
        progress: bool = False,
        auto_adjust: bool = False,
    ) -> pandas.DataFrame:
        call_counter["count"] += 1
        if call_counter["count"] < 3:
            raise ValueError("temporary error")
        return raw_dataframe

    monkeypatch.setattr(
        "stock_indicator.data_loader.yfinance.download", flaky_download
    )
    monkeypatch.setattr("stock_indicator.symbols.load_symbols", lambda: ["TEST"])
    with caplog.at_level(logging.WARNING):
        result_dataframe = download_history("TEST", "2021-01-01", "2021-01-02")

    assert call_counter["count"] == 3
    expected_dataframe = pandas.DataFrame(
        {
            "open": [50.0],
            "high": [55.0],
            "low": [45.0],
            "close": [50.0],
            "volume": [2000.0],
        }
    )
    pandas.testing.assert_frame_equal(result_dataframe, expected_dataframe)
    assert "Attempt 1 to download data for TEST failed" in caplog.text


def test_download_history_raises_after_max_attempts(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The function should raise the last error after exhausting retries."""

    def failing_download(
        symbol: str,
        start: str,
        end: str,
        progress: bool = False,
        auto_adjust: bool = False,
    ) -> pandas.DataFrame:
        raise ValueError("permanent error")

    monkeypatch.setattr(
        "stock_indicator.data_loader.yfinance.download", failing_download
    )
    monkeypatch.setattr("stock_indicator.symbols.load_symbols", lambda: ["TEST"])
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            download_history("TEST", "2021-01-01", "2021-01-02")
    assert "Failed to download data for TEST after" in caplog.text


def test_download_history_forwards_optional_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The function should forward optional keyword arguments to yfinance."""
    captured_arguments: dict[str, str] = {}

    def stubbed_download(
        symbol: str,
        start: str,
        end: str,
        progress: bool = False,
        **options: str,
    ) -> pandas.DataFrame:
        captured_arguments.update(options)
        return pandas.DataFrame(
            {
                "Open": [100.0],
                "High": [110.0],
                "Low": [90.0],
                "Close": [100.0],
                "Adj Close": [100.0],
                "Volume": [1000.0],
            }
        )

    monkeypatch.setattr(
        "stock_indicator.data_loader.yfinance.download", stubbed_download
    )
    monkeypatch.setattr("stock_indicator.symbols.load_symbols", lambda: ["TEST"])
    download_history("TEST", "2021-01-01", "2021-01-02", interval="1h")
    assert captured_arguments["interval"] == "1h"


def test_download_history_uses_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """When cached data is present, only missing rows should be requested."""
    symbol_name = "TEST"
    cache_file_path = tmp_path / "TEST.csv"
    existing_frame = pandas.DataFrame(
        {
            "open": [50.0, 60.0],
            "high": [55.0, 65.0],
            "low": [45.0, 55.0],
            "close": [50.0, 60.0],
            "volume": [2000.0, 3000.0],
        },
        index=pandas.to_datetime(["2023-01-01", "2023-01-02"]),
    )
    existing_frame.to_csv(cache_file_path)

    captured_arguments: dict[str, str] = {}
    downloaded_raw_frame = pandas.DataFrame(
        {
            "Open": [100.0, 120.0],
            "High": [110.0, 130.0],
            "Low": [90.0, 110.0],
            "Close": [100.0, 120.0],
            "Adj Close": [50.0, 60.0],
            "Volume": [1000.0, 1200.0],
        },
        index=pandas.to_datetime(["2023-01-03", "2023-01-04"]),
    )

    def stubbed_download(
        symbol: str,
        start: str,
        end: str,
        progress: bool = False,
        auto_adjust: bool = False,
    ) -> pandas.DataFrame:
        captured_arguments["start"] = start
        captured_arguments["end"] = end
        return downloaded_raw_frame

    monkeypatch.setattr(
        "stock_indicator.data_loader.yfinance.download", stubbed_download
    )
    monkeypatch.setattr("stock_indicator.symbols.load_symbols", lambda: [symbol_name])

    combined_frame = download_history(
        symbol_name,
        "2023-01-01",
        "2023-01-05",
        cache_path=cache_file_path,
    )

    assert captured_arguments["start"] == "2023-01-03"
    assert captured_arguments["end"] == "2023-01-05"
    assert list(combined_frame.index) == list(
        pandas.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"])
    )
    saved_frame = pandas.read_csv(cache_file_path, index_col=0, parse_dates=True)
    pandas.testing.assert_frame_equal(combined_frame, saved_frame)
