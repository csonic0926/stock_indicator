"""Integration tests for running daily job commands via the manager CLI."""

# TODO: review

import io
import datetime
from pathlib import Path
import os
import sys

import pandas
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)


def test_manager_cli_generates_logs_and_signals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running daily job steps through the CLI should create log files."""

    import stock_indicator.manage as manage_module

    data_directory = tmp_path / "data"
    stock_data_directory = data_directory / "stock_data"
    cron_log_directory = tmp_path / "cron_logs"
    date_log_directory = tmp_path / "logs"
    cron_log_directory.mkdir()
    date_log_directory.mkdir()

    monkeypatch.setattr(manage_module, "DATA_DIRECTORY", data_directory)
    monkeypatch.setattr(manage_module, "STOCK_DATA_DIRECTORY", stock_data_directory)

    import stock_indicator.daily_job as daily_job_module

    monkeypatch.setattr(daily_job_module, "DATA_DIRECTORY", data_directory)
    monkeypatch.setattr(daily_job_module, "STOCK_DATA_DIRECTORY", stock_data_directory)

    monkeypatch.setattr(daily_job_module, "load_symbols", lambda: ["AAA"])
    monkeypatch.setattr(
        daily_job_module.strategy, "load_ff12_groups_by_symbol", lambda: {"AAA": 1}
    )
    monkeypatch.setattr(
        daily_job_module.strategy, "load_symbols_excluded_by_industry", lambda: set()
    )
    monkeypatch.setattr(
        daily_job_module, "load_symbols_rejected_by_asset_metadata", lambda: set()
    )
    monkeypatch.setattr(
        daily_job_module, "load_symbols_rejected_by_listing_name", lambda: set()
    )

    recorded_end_dates: list[str] = []

    def fake_download_history(
        symbol_name: str,
        start: str,
        end: str,
        cache_path: Path | None = None,
    ) -> pandas.DataFrame:
        recorded_end_dates.append(end)
        data_frame = pandas.DataFrame(
            {"open": [1.0], "close": [1.0]}, index=pandas.to_datetime(["2024-01-10"])
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            data_frame.to_csv(cache_path)
        return data_frame

    monkeypatch.setattr(manage_module.data_loader, "download_history", fake_download_history)
    monkeypatch.setattr(daily_job_module, "download_history", fake_download_history)
    monkeypatch.setattr(manage_module, "_cleanup_yfinance_session", lambda: None)

    recorded_arguments: dict[str, str] = {}

    def fake_run_daily_tasks(
        buy_strategy_name: str,
        sell_strategy_name: str,
        start_date: str,
        end_date: str,
        **_: object,
    ) -> dict[str, list[str]]:
        recorded_arguments["end"] = end_date
        return {"entry_signals": ["AAA"], "exit_signals": ["BBB"]}

    monkeypatch.setattr(daily_job_module, "run_daily_tasks", fake_run_daily_tasks)
    monkeypatch.setattr(
        daily_job_module,
        "determine_latest_trading_date",
        lambda: datetime.date(2024, 1, 10),
    )

    output_buffer = io.StringIO()
    shell = manage_module.StockShell(stdout=output_buffer)
    shell.onecmd("update_all_data_from_yf 2024-01-09 2024-01-10")
    shell.onecmd("find_history_signal dollar_volume>1 ema_sma_cross ema_sma_cross 1.0")

    log_file_path = cron_log_directory / "cron_stdout.log"
    log_file_path.write_text(output_buffer.getvalue(), encoding="utf-8")

    date_marker_path = date_log_directory / "2024-01-10.log"
    date_marker_path.touch()

    assert log_file_path.exists()
    assert date_marker_path.exists()
    assert recorded_arguments["end"] == "2024-01-10"
    assert set(recorded_end_dates) == {"2024-01-11"}

