"""Tests for dollar volume analysis utilities."""
# TODO: review

from __future__ import annotations

from pathlib import Path

import pandas
import pytest

from stock_indicator.volume import (
    count_symbols_with_average_dollar_volume_above,
)


def test_count_symbols_with_average_dollar_volume_above_returns_correct_count(
    tmp_path: Path,
) -> None:
    """The function should count symbols above the specified dollar volume threshold."""

    date_index = pandas.date_range("2020-01-01", periods=60, freq="D")

    high_volume_price_values = [10.0] * 60
    high_volume_volume_values = [1_000_000] * 60
    high_volume_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": high_volume_price_values,
            "close": high_volume_price_values,
            "volume": high_volume_volume_values,
        }
    )
    high_volume_data_frame.to_csv(tmp_path / "high.csv", index=False)

    low_volume_price_values = [10.0] * 60
    low_volume_volume_values = [200_000] * 60
    low_volume_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": low_volume_price_values,
            "close": low_volume_price_values,
            "volume": low_volume_volume_values,
        }
    )
    low_volume_data_frame.to_csv(tmp_path / "low.csv", index=False)

    result = count_symbols_with_average_dollar_volume_above(tmp_path, 5)
    assert result == 1

    result = count_symbols_with_average_dollar_volume_above(tmp_path, 1)
    assert result == 2


def test_count_symbols_with_average_dollar_volume_above_requires_volume_column(
    tmp_path: Path,
) -> None:
    """The function should raise when the volume column is missing."""

    date_index = pandas.date_range("2020-01-01", periods=60, freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": [10.0] * 60,
            "close": [10.0] * 60,
        }
    )
    price_data_frame.to_csv(tmp_path / "missing.csv", index=False)

    with pytest.raises(ValueError, match="Volume column is required"):
        count_symbols_with_average_dollar_volume_above(tmp_path, 1)


def test_count_symbols_with_average_dollar_volume_above_computes_value_correctly(
    tmp_path: Path,
) -> None:
    """The function should compute the correct 50-day average dollar volume."""

    date_index = pandas.date_range("2020-01-01", periods=60, freq="D")
    price_values = [10.0] * 60
    volume_values = [1_000_000.0] * 60
    data_frame = pandas.DataFrame(
        {"Date": date_index, "open": price_values, "close": price_values, "volume": volume_values}
    )
    data_frame.to_csv(tmp_path / "single.csv", index=False)

    below_threshold_result = count_symbols_with_average_dollar_volume_above(
        tmp_path, 9.9
    )
    above_threshold_result = count_symbols_with_average_dollar_volume_above(
        tmp_path, 10.1
    )
    assert below_threshold_result == 1
    assert above_threshold_result == 0
