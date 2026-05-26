"""Tests for the download_history utility."""
# TODO: review

import logging
import os
import sys
import types
import pathlib

import pandas
import pytest
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Create a minimal yfinance stub before importing the module under test
fake_yfinance_module = types.ModuleType("yfinance")
fake_yfinance_module.download = lambda *args, **kwargs: pandas.DataFrame()
sys.modules["yfinance"] = fake_yfinance_module

from stock_indicator.data_loader import download_history


def test_download_history_returns_dataframe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The function should return data provided by yfinance."""
    raw_dataframe = pandas.DataFrame({"Close": [1.0, 2.0]})

    def stubbed_download(
        symbol: str,
        start: str,
        end: str,
        progress: bool = False,
        auto_adjust: bool = True,
    ) -> pandas.DataFrame:
        return raw_dataframe

    monkeypatch.setattr(
        "stock_indicator.data_loader.yfinance.download", stubbed_download
    )
    monkeypatch.setattr("stock_indicator.symbols.load_symbols", lambda: ["TEST"])
    result_dataframe = download_history("TEST", "2021-01-01", "2021-01-02")
    expected_dataframe = raw_dataframe.rename(
        columns=lambda name: name.lower().replace(" ", "_")
    )
    pandas.testing.assert_frame_equal(result_dataframe, expected_dataframe)


def test_download_history_flattens_multiindex_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The function should flatten MultiIndex columns and normalize their names."""
    raw_dataframe = pandas.DataFrame(
        {
            ("Close", "TEST"): [1.0],
            ("Open", "TEST"): [1.5],
        }
    )

    def stubbed_download(
        symbol: str,
        start: str,
        end: str,
        progress: bool = False,
        auto_adjust: bool = True,
    ) -> pandas.DataFrame:
        return raw_dataframe

    monkeypatch.setattr(
        "stock_indicator.data_loader.yfinance.download", stubbed_download
    )
    monkeypatch.setattr("stock_indicator.symbols.load_symbols", lambda: ["TEST"])
    result_dataframe = download_history("TEST", "2021-01-01", "2021-01-02")
    assert list(result_dataframe.columns) == ["close", "open"]


def test_download_history_retries_on_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The function should retry and log warnings on temporary failures."""
    call_counter = {"count": 0}
    raw_dataframe = pandas.DataFrame({"Close": [1.0]})

    def flaky_download(
        symbol: str,
        start: str,
        end: str,
        progress: bool = False,
        auto_adjust: bool = True,
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
    expected_dataframe = raw_dataframe.rename(
        columns=lambda name: name.lower().replace(" ", "_")
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
        auto_adjust: bool = True,
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
    captured_arguments: dict[str, Any] = {}

    def stubbed_download(
        symbol: str,
        start: str,
        end: str,
        progress: bool = False,
        **options: Any,
    ) -> pandas.DataFrame:
        captured_arguments.update(options)
        return pandas.DataFrame()

    monkeypatch.setattr(
        "stock_indicator.data_loader.yfinance.download", stubbed_download
    )
    monkeypatch.setattr("stock_indicator.symbols.load_symbols", lambda: ["TEST"])
    download_history("TEST", "2021-01-01", "2021-01-02", interval="1h")
    assert captured_arguments["interval"] == "1h"
    assert captured_arguments["auto_adjust"] is True


def test_download_history_uses_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """When cached data is present, only missing rows should be requested."""
    symbol_name = "TEST"
    cache_file_path = tmp_path / "TEST.csv"
    existing_frame = pandas.DataFrame(
        {"close": [1.0, 2.0]},
        index=pandas.to_datetime(["2023-01-01", "2023-01-02"]),
    )
    existing_frame.to_csv(cache_file_path)

    captured_arguments: dict[str, str] = {}
    downloaded_raw_frame = pandas.DataFrame(
        {"Close": [3.0, 4.0]},
        index=pandas.to_datetime(["2023-01-03", "2023-01-04"]),
    )

    def stubbed_download(
        symbol: str,
        start: str,
        end: str,
        progress: bool = False,
        auto_adjust: bool = True,
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


def test_download_history_does_not_request_empty_exclusive_end_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Do not call Yahoo when the cache already reaches the exclusive end."""
    symbol_name = "TEST"
    cache_file_path = tmp_path / "TEST.csv"
    cached_frame = pandas.DataFrame(
        {"close": [1.0, 2.0]},
        index=pandas.to_datetime(["2023-01-01", "2023-01-02"]),
    )
    cached_frame.to_csv(cache_file_path)

    download_call_count = 0

    def stubbed_download(
        symbol: str,
        start: str,
        end: str,
        progress: bool = False,
        auto_adjust: bool = True,
    ) -> pandas.DataFrame:
        nonlocal download_call_count
        download_call_count += 1
        return pandas.DataFrame()

    monkeypatch.setattr(
        "stock_indicator.data_loader.yfinance.download", stubbed_download
    )
    monkeypatch.setattr("stock_indicator.symbols.load_symbols", lambda: [symbol_name])

    combined_frame = download_history(
        symbol_name,
        "2023-01-01",
        "2023-01-03",
        cache_path=cache_file_path,
    )

    assert download_call_count == 0
    saved_frame = pandas.read_csv(cache_file_path, index_col=0, parse_dates=True)
    pandas.testing.assert_frame_equal(combined_frame, saved_frame)


def test_download_history_extends_cache_backwards_and_forwards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The function should prepend and append data around an existing cache."""
    symbol_name = "TEST"
    cache_file_path = tmp_path / "TEST.csv"
    cached_frame = pandas.DataFrame(
        {"close": [2.0, 3.0]},
        index=pandas.to_datetime(["2023-01-02", "2023-01-03"]),
    )
    cached_frame.to_csv(cache_file_path)

    call_arguments: list[dict[str, str]] = []
    early_raw_frame = pandas.DataFrame(
        {"Close": [1.0]},
        index=pandas.to_datetime(["2023-01-01"]),
    )
    forward_raw_frame = pandas.DataFrame(
        {"Close": [4.0]},
        index=pandas.to_datetime(["2023-01-04"]),
    )

    def stubbed_download(
        symbol: str,
        start: str,
        end: str,
        progress: bool = False,
        auto_adjust: bool = True,
    ) -> pandas.DataFrame:
        call_arguments.append({"start": start, "end": end})
        if start == "2023-01-01":
            return early_raw_frame
        return forward_raw_frame

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

    assert call_arguments[0]["start"] == "2023-01-01"
    assert call_arguments[0]["end"] == "2023-01-02"
    assert call_arguments[1]["start"] == "2023-01-04"
    assert call_arguments[1]["end"] == "2023-01-05"
    assert list(combined_frame.index) == list(
        pandas.to_datetime(
            ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
        )
    )
    saved_frame = pandas.read_csv(cache_file_path, index_col=0, parse_dates=True)
    pandas.testing.assert_frame_equal(combined_frame, saved_frame)
