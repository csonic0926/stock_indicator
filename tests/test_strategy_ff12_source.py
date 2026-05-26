"""Tests for simulation-specific FF12 source loading."""

# TODO: review

from pathlib import Path

import pandas

from stock_indicator import strategy


def test_load_ff12_groups_supports_custom_symbol_csv(tmp_path: Path) -> None:
    """A frozen universe map can use symbol/ff12_numeric column names."""

    sector_path = tmp_path / "legacy_sector.csv"
    pandas.DataFrame(
        [
            {"symbol": "BRK.B", "ff12_numeric": 11},
            {"symbol": "AAA", "ff12_numeric": 6},
        ]
    ).to_csv(sector_path, index=False)

    symbol_to_group = strategy.load_ff12_groups_by_symbol(sector_path)

    assert symbol_to_group["AAA"] == 6
    assert symbol_to_group["BRK.B"] == 11
    assert symbol_to_group["BRK-B"] == 11


def test_ff12_group_source_override_restores_previous_source(
    tmp_path: Path,
) -> None:
    """The temporary override should not leak across simulations."""

    first_sector_path = tmp_path / "first_sector.csv"
    second_sector_path = tmp_path / "second_sector.csv"
    pandas.DataFrame([{"ticker": "AAA", "ff12": 6}]).to_csv(
        first_sector_path,
        index=False,
    )
    pandas.DataFrame([{"ticker": "BBB", "ff12": 9}]).to_csv(
        second_sector_path,
        index=False,
    )

    with strategy.override_ff12_group_source_path(first_sector_path):
        assert strategy.load_ff12_groups_by_symbol() == {"AAA": 6}
        with strategy.override_ff12_group_source_path(second_sector_path):
            assert strategy.load_ff12_groups_by_symbol() == {"BBB": 9}
        assert strategy.load_ff12_groups_by_symbol() == {"AAA": 6}
