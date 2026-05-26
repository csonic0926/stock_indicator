"""Tests for symbol cache utilities."""
# TODO: review

import os
import sys
from pathlib import Path

import pandas
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)


def test_load_symbols_fetches_filters_and_caches_sec_tickers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The loader should cache only common-stock-like SEC tickers."""

    import stock_indicator.symbols as symbols_module

    cache_path = tmp_path / "symbols.txt"
    monkeypatch.setattr(symbols_module, "SYMBOL_CACHE_PATH", cache_path)

    call_counter = {"count": 0}

    def fake_fetch_company_ticker_table() -> pandas.DataFrame:
        call_counter["count"] += 1
        return pandas.DataFrame(
            [
                {"ticker": "AAA", "title": "AAA Technologies Inc."},
                {"ticker": "bbb", "title": "BBB Software Inc."},
                {"ticker": "AAA", "title": "AAA Technologies Inc."},
                {"ticker": None, "title": "Missing Symbol Inc."},
                {"ticker": "SPY", "title": "SPDR S&P 500 ETF TRUST"},
                {"ticker": "GLD", "title": "SPDR GOLD TRUST"},
                {"ticker": "ACHR-WT", "title": "Archer Aviation Inc."},
                {"ticker": "AACBU", "title": "Ares Acquisition Corp II"},
                {"ticker": "ABR-PD", "title": "Arbor Realty Trust Inc."},
            ]
        )

    monkeypatch.setattr(
        "stock_indicator.sector_pipeline.sec_api.fetch_company_ticker_table",
        fake_fetch_company_ticker_table,
    )

    symbol_list = symbols_module.load_symbols()
    assert symbol_list == ["AAA", "BBB"]
    assert cache_path.exists()

    symbol_list_second = symbols_module.load_symbols()
    assert symbol_list_second == ["AAA", "BBB"]
    assert call_counter["count"] == 1


def test_hard_filter_keeps_ambiguous_trusts_for_later_semantic_review() -> None:
    """Hard filter should not reject plain trust or fund wording by itself."""

    import stock_indicator.symbols as symbols_module

    company_ticker_table = pandas.DataFrame(
        [
            {"ticker": "BXMT", "title": "Blackstone Mortgage Trust Inc."},
            {"ticker": "BRK-B", "title": "Berkshire Hathaway Inc."},
            {"ticker": "FUND", "title": "Sprott Focus Trust Fund"},
            {"ticker": "QQQ", "title": "Invesco QQQ Trust, Series 1"},
        ]
    )

    assert symbols_module.build_hard_filtered_symbols_from_company_tickers(
        company_ticker_table
    ) == ["BRK.B", "BXMT", "FUND"]
