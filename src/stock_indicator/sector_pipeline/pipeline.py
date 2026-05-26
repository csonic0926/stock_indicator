"""Pipeline for tagging symbols with SIC and Fama-French groups.

The pipeline stores intermediate data in directories under ``cache/`` within
the project root. Each call to :func:`build_sector_classification_dataset`
records its configuration in ``cache/last_run.json`` and caches SEC submission
files in ``cache/submissions``. Subsequent executions can call
:func:`update_latest_dataset` to rebuild the output using that saved
configuration while reusing any cached submissions. This incremental approach
avoids downloading data for symbols that have already been processed.
"""

# TODO: review

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from stock_indicator.symbols import load_symbols

from .config import (
    SUBMISSIONS_DIRECTORY,
    LAST_RUN_CONFIG_PATH,
    DEFAULT_OUTPUT_PARQUET_PATH,
    DEFAULT_OUTPUT_CSV_PATH,
    SIC_TO_FAMA_FRENCH_MAPPING_PATH,
)
from .utils import (
    ensure_directory_exists,
    save_json_file,
    load_json_file,
    normalize_ticker_symbol,
)
from .sec_api import (
    map_tickers_to_central_index_and_classification,
)
from .ff_mapping import (
    load_fama_french_mapping,
    build_classification_lookup,
    attach_fama_french_groups,
)
from .secondary_classification import apply_secondary_classifications

logger = logging.getLogger(__name__)

DEFAULT_SYMBOL_UNIVERSE_SOURCE = "symbols.txt"

# Ensure required cache directories exist for incremental updates
ensure_directory_exists(LAST_RUN_CONFIG_PATH.parent)
ensure_directory_exists(SUBMISSIONS_DIRECTORY)


def load_universe(source: str | Path) -> pd.DataFrame:
    """Load a universe of ticker symbols from ``source``.

    ``source`` may be a CSV file with a ``ticker`` column or a newline-separated list
    of symbols provided as a file or URL.
    """
    source_str = str(source)
    if source_str.endswith(".csv"):
        data_frame = pd.read_csv(source_str)
        if "ticker" not in data_frame.columns:
            raise ValueError("CSV input must contain a 'ticker' column")
        return data_frame[["ticker"]].dropna().drop_duplicates()
    if source_str.startswith("http://") or source_str.startswith("https://"):
        try:
            logger.info("Downloading ticker universe from %s", source_str)
            response = requests.get(source_str, timeout=30)
            response.raise_for_status()
            text = response.text
        except requests.RequestException as error:
            logger.error("Failed to download ticker universe from %s: %s", source_str, error)
            raise
    else:
        with Path(source_str).open("r", encoding="utf-8") as file_pointer:
            text = file_pointer.read()
    tickers = [
        normalize_ticker_symbol(symbol)
        for symbol in text.splitlines()
        if symbol.strip()
    ]
    return pd.DataFrame({"ticker": tickers})


def load_current_symbol_universe() -> pd.DataFrame:
    """Return the current tradable symbol universe from ``data/symbols.txt``."""

    normalized_symbols = [
        normalize_ticker_symbol(symbol_name)
        for symbol_name in load_symbols()
        if symbol_name and str(symbol_name).strip()
    ]
    unique_symbols = sorted(dict.fromkeys(normalized_symbols))
    return pd.DataFrame({"ticker": unique_symbols})


def resolve_sector_universe(
    universe_source: str | Path | pd.DataFrame | None,
) -> pd.DataFrame:
    """Return the symbol universe used for sector classification.

    ``None`` means the already-curated project symbol cache. Passing a
    DataFrame or explicit file/URL remains available for tests and one-off
    audits, but production should classify only ``symbols.txt``.
    """

    if universe_source is None:
        return load_current_symbol_universe()
    if isinstance(universe_source, pd.DataFrame):
        if "ticker" not in universe_source.columns:
            raise ValueError("Universe DataFrame must contain a 'ticker' column")
        universe_data_frame = universe_source[["ticker"]].copy()
    else:
        universe_data_frame = load_universe(universe_source)
    universe_data_frame["ticker"] = universe_data_frame["ticker"].map(
        normalize_ticker_symbol
    )
    universe_data_frame = universe_data_frame.dropna()
    universe_data_frame = universe_data_frame.loc[universe_data_frame["ticker"] != ""]
    return universe_data_frame.drop_duplicates(subset=["ticker"])


def build_sector_classification_dataset(
    mapping_source: str | Path = SIC_TO_FAMA_FRENCH_MAPPING_PATH,
    output_parquet_path: Path = DEFAULT_OUTPUT_PARQUET_PATH,
    output_csv_path: Optional[Path] = DEFAULT_OUTPUT_CSV_PATH,
    universe_source: str | Path | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate a dataset of symbols with CIK, SIC, and Fama-French codes.

    The production ticker universe comes from the curated ``data/symbols.txt``
    cache. ``mapping_source`` may be a path to a CSV file or a URL pointing to
    one. By default the local ``sic_to_ff.csv`` file under the repository's
    ``data`` directory is used.
    """
    ensure_directory_exists(LAST_RUN_CONFIG_PATH.parent)
    ensure_directory_exists(SUBMISSIONS_DIRECTORY)
    universe_data_frame = resolve_sector_universe(universe_source)
    ticker_mapping_data_frame = map_tickers_to_central_index_and_classification(
        universe_data_frame
    )
    mapping_data_frame = load_fama_french_mapping(mapping_source)
    lookup_data_frame = build_classification_lookup(mapping_data_frame)
    classified_data_frame = attach_fama_french_groups(
        ticker_mapping_data_frame, lookup_data_frame
    )
    classified_data_frame = apply_secondary_classifications(classified_data_frame)
    classified_data_frame["sic_desc"] = ""
    ensure_directory_exists(output_parquet_path.parent)
    classified_data_frame.to_parquet(output_parquet_path, index=False)
    if output_csv_path is not None:
        ensure_directory_exists(output_csv_path.parent)
        classified_data_frame.to_csv(output_csv_path, index=False)
    save_json_file(
        {
            "mapping_source": str(mapping_source),
            "output": str(output_parquet_path),
            "universe_source": (
                str(universe_source)
                if universe_source is not None
                else DEFAULT_SYMBOL_UNIVERSE_SOURCE
            ),
        },
        LAST_RUN_CONFIG_PATH,
    )
    return classified_data_frame


def update_latest_dataset() -> pd.DataFrame:
    """Rebuild the classification data using saved or default configuration.

    If no prior configuration exists, use the default mapping path
    (``data/sic_to_ff.csv``) and default output path.
    """
    if LAST_RUN_CONFIG_PATH.exists():
        configuration = load_json_file(LAST_RUN_CONFIG_PATH)
    else:
        configuration = {}
    mapping_source = str(
        configuration.get("mapping_source", SIC_TO_FAMA_FRENCH_MAPPING_PATH)
    )
    output_path = Path(configuration.get("output", DEFAULT_OUTPUT_PARQUET_PATH))
    # If the source looks like a local path, ensure it exists or fall back to default
    if not (mapping_source.startswith("http://") or mapping_source.startswith("https://")):
        mapping_path = Path(mapping_source)
        if not mapping_path.exists():
            default_path = Path(SIC_TO_FAMA_FRENCH_MAPPING_PATH)
            if default_path.exists():
                mapping_source = str(default_path)
            else:
                raise FileNotFoundError(
                    f"Fama-French mapping file not found: {mapping_source}. "
                    "Place a mapping at data/sic_to_ff.csv or run 'update_sector_data --ff-map-url=URL OUTPUT_PATH'."
                )
    configured_universe_source = configuration.get("universe_source")
    universe_source = (
        None
        if configured_universe_source in (None, DEFAULT_SYMBOL_UNIVERSE_SOURCE)
        else configured_universe_source
    )
    return build_sector_classification_dataset(
        mapping_source,
        output_path,
        DEFAULT_OUTPUT_CSV_PATH,
        universe_source=universe_source,
    )


def generate_coverage_report(data_frame: pd.DataFrame) -> str:
    """Return coverage information for ``data_frame`` as a formatted string."""
    total_count = len(data_frame)
    with_cik = data_frame["cik"].notna().sum()
    with_sic = data_frame["sic"].notna().sum()
    with_ff12 = data_frame["ff12"].notna().sum()
    if "classification_confidence" in data_frame.columns:
        low_confidence_ff12 = (
            data_frame["classification_confidence"].astype(str).str.lower()
            == "low"
        ).sum()
    else:
        low_confidence_ff12 = 0
    trusted_ff12 = with_ff12 - low_confidence_ff12
    return (
        f"Total: {total_count}\n"
        f"CIK: {with_cik} ({with_cik / total_count:.1%})\n"
        f"SIC: {with_sic} ({with_sic / total_count:.1%})\n"
        f"FF12 trusted: {trusted_ff12} ({trusted_ff12 / total_count:.1%})\n"
        f"FF12 low-confidence fallback: "
        f"{low_confidence_ff12} ({low_confidence_ff12 / total_count:.1%})"
    )
