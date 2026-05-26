"""Configuration settings for the sector classification pipeline.

The pipeline requires a table that maps Standard Industrial Classification
codes (SIC) to Fama–French industry groups. Obtain the official SIC→Fama–French
definitions from the Kenneth R. French Data Library and store them locally as
``data/sic_to_ff.csv``. For source material and periodic updates, refer to:

- Kenneth R. French Data Library: Industry definitions and SIC ranges
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
  (see the 49-Industry or 12-Industry SIC definitions)

You can also supply a URL or file path at runtime via the management shell
command ``update_sector_data --ff-map-url=URL OUTPUT_PATH``. When omitted, the
pipeline falls back to the local ``data/sic_to_ff.csv`` file.

SEC API requests must include a descriptive User-Agent with contact details per
SEC guidance. Update ``SEC_USER_AGENT`` below with your organization/app name and
a valid email/URL where you can be reached.
"""

# TODO: review

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_TICKERS_EXCHANGE_URL = (
    "https://www.sec.gov/files/company_tickers_exchange.json"
)
SEC_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik_padded}.json"

# Per SEC guidelines, identify your app and provide a real contact.
# Example format: "app-name/version (contact: email@domain.com)"
# TODO: review
SEC_USER_AGENT = "stock-indicator/1.0 (contact: maintainer@example.com)"

CACHE_DIRECTORY = REPOSITORY_ROOT / "cache"
SUBMISSIONS_DIRECTORY = CACHE_DIRECTORY / "submissions"
LAST_RUN_CONFIG_PATH = CACHE_DIRECTORY / "last_run.json"

DATA_DIRECTORY = REPOSITORY_ROOT / "data"
SIC_TO_FAMA_FRENCH_MAPPING_PATH = DATA_DIRECTORY / "sic_to_ff.csv"
DEFAULT_OUTPUT_PARQUET_PATH = DATA_DIRECTORY / "symbols_with_sector.parquet"
DEFAULT_OUTPUT_CSV_PATH = DATA_DIRECTORY / "symbols_with_sector.csv"

NORMALIZE_DOT_TO_DASH = True
