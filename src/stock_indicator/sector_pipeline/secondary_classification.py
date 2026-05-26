"""Secondary sector classification for symbols without SEC SIC data."""

# TODO: review

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas

from .config import DATA_DIRECTORY

LOGGER = logging.getLogger(__name__)

SECONDARY_CLASSIFICATIONS_CSV_PATH = (
    DATA_DIRECTORY / "sector_secondary_classifications.csv"
)
LLM_CLASSIFICATIONS_CSV_PATH = (
    DATA_DIRECTORY / "sector_missing_sic_llm_classification.csv"
)

FAMA_FRENCH_LABEL_BY_GROUP_IDENTIFIER: dict[int, str] = {
    1: "NoDur",
    2: "Durbl",
    3: "Manuf",
    4: "Enrgy",
    5: "Chems",
    6: "BusEq",
    7: "Telcm",
    8: "Utils",
    9: "Shops",
    10: "Hlth",
    11: "Money",
    12: "Other",
}

TRUSTED_CLASSIFICATION_CONFIDENCE_VALUES = {"high", "medium"}


def _clean_metadata_text(value: Any) -> str:
    """Return normalized metadata text for deterministic rule matching."""

    if value is None:
        return ""
    try:
        if pandas.isna(value):
            return ""
    except TypeError:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _text_contains_any(text_value: str, patterns: tuple[str, ...]) -> bool:
    """Return whether ``text_value`` contains any whole-pattern substring."""

    return any(pattern in text_value for pattern in patterns)


def _build_yfinance_classification(
    group_identifier: int,
    confidence_value: str,
    reason: str,
) -> dict[str, Any]:
    """Return a normalized secondary-classification record."""

    return {
        "ff12": group_identifier,
        "ff_label": FAMA_FRENCH_LABEL_BY_GROUP_IDENTIFIER[group_identifier],
        "classification_confidence": confidence_value,
        "reason": reason,
    }


def classify_yfinance_sector_industry(
    yfinance_sector: Any,
    yfinance_industry: Any,
) -> dict[str, Any] | None:
    """Classify Yahoo Finance sector/industry metadata into FF12.

    The mapping is intentionally conservative: industry-specific rules win over
    broad sector fallbacks, and ambiguous industries return ``None`` so they
    remain on the missing-SIC audit list instead of becoming fake "Other".
    """

    sector_text = _clean_metadata_text(yfinance_sector)
    industry_text = _clean_metadata_text(yfinance_industry)
    combined_text = f"{sector_text} {industry_text}".strip()
    if not combined_text:
        return None
    if _text_contains_any(industry_text, ("shell companies", "shell company")):
        return None

    # FF12=12 can be a valid bucket when the industry is outside the first
    # eleven Fama-French groups. Keep those cases medium-confidence because the
    # source is Yahoo taxonomy rather than SEC SIC.
    other_industry_patterns = (
        "airlines",
        "airports",
        "lodging",
        "resorts",
        "casinos",
        "gambling",
        "travel services",
        "leisure",
        "entertainment",
        "broadcasting",
        "publishing",
        "advertising agencies",
        "marine shipping",
        "trucking",
        "railroads",
        "integrated freight",
        "logistics",
        "engineering",
        "construction",
        "consulting",
        "staffing",
        "rental",
        "waste management",
        "education",
        "conglomerates",
        "security & protection services",
        "specialty business services",
        "gold",
        "silver",
        "copper",
        "other industrial metals",
        "other precious metals",
        "uranium",
        "thermal coal",
        "coking coal",
    )
    if _text_contains_any(industry_text, other_industry_patterns):
        return _build_yfinance_classification(
            12,
            "medium",
            "Yahoo industry maps to FF12 Other rather than a missing-SIC fallback.",
        )

    money_patterns = (
        "asset management",
        "banks",
        "bank",
        "capital markets",
        "credit services",
        "insurance",
        "mortgage finance",
        "financial conglomerates",
        "financial data",
        "stock exchanges",
        "reit",
        "real estate",
    )
    if sector_text in {"financial services", "real estate"} or _text_contains_any(
        industry_text,
        money_patterns,
    ):
        return _build_yfinance_classification(
            11,
            "high",
            "Yahoo sector/industry indicates finance or real-estate equity.",
        )

    if sector_text == "healthcare":
        return _build_yfinance_classification(
            10,
            "high",
            "Yahoo sector indicates healthcare.",
        )

    shop_patterns = (
        "retail",
        "restaurants",
        "grocery",
        "discount stores",
        "department stores",
        "home improvement",
        "internet retail",
        "specialty retail",
        "auto & truck dealerships",
    )
    if _text_contains_any(industry_text, shop_patterns):
        return _build_yfinance_classification(
            9,
            "high",
            "Yahoo industry indicates retail, restaurants, or wholesale shops.",
        )

    durable_patterns = (
        "auto manufacturers",
        "auto parts",
        "recreational vehicles",
        "furnishings",
        "fixtures",
        "appliances",
        "electronics",
    )
    if _text_contains_any(industry_text, durable_patterns):
        return _build_yfinance_classification(
            2,
            "high",
            "Yahoo industry indicates durable consumer goods.",
        )

    non_durable_patterns = (
        "apparel",
        "footwear",
        "textile",
        "packaged foods",
        "beverages",
        "tobacco",
        "household",
        "personal products",
        "luxury goods",
        "confectioners",
        "farm products",
    )
    if sector_text == "consumer defensive" or _text_contains_any(
        industry_text,
        non_durable_patterns,
    ):
        return _build_yfinance_classification(
            1,
            "medium",
            "Yahoo sector/industry indicates non-durable consumer goods.",
        )

    manufacturing_patterns = (
        "aerospace",
        "defense",
        "machinery",
        "industrial distribution",
        "electrical equipment",
        "metal fabrication",
        "tools",
        "building products",
        "building materials",
        "packaging",
        "paper",
        "lumber",
        "steel",
        "aluminum",
        "specialty materials",
    )
    if _text_contains_any(industry_text, manufacturing_patterns):
        return _build_yfinance_classification(
            3,
            "medium",
            "Yahoo industry indicates manufacturing or industrial materials.",
        )

    if sector_text == "energy":
        return _build_yfinance_classification(
            4,
            "high",
            "Yahoo sector indicates energy.",
        )

    chemical_patterns = (
        "chemicals",
        "specialty chemicals",
        "agricultural inputs",
    )
    if _text_contains_any(industry_text, chemical_patterns):
        return _build_yfinance_classification(
            5,
            "high",
            "Yahoo industry indicates chemicals.",
        )

    if sector_text == "technology":
        return _build_yfinance_classification(
            6,
            "high",
            "Yahoo sector indicates business equipment or technology.",
        )

    if _text_contains_any(
        industry_text,
        (
            "software",
            "semiconductors",
            "computer hardware",
            "communication equipment",
            "electronic components",
            "information technology",
            "internet content",
            "electronic gaming",
        ),
    ):
        return _build_yfinance_classification(
            6,
            "medium",
            "Yahoo industry indicates software, hardware, or online services.",
        )

    if _text_contains_any(
        industry_text,
        ("telecom", "wireless", "integrated telecommunication"),
    ):
        return _build_yfinance_classification(
            7,
            "high",
            "Yahoo industry indicates telecommunications.",
        )

    if sector_text == "utilities":
        return _build_yfinance_classification(
            8,
            "high",
            "Yahoo sector indicates utilities.",
        )

    return None


def build_secondary_classifications_from_yfinance_metadata(
    metadata_frame: pandas.DataFrame,
) -> pandas.DataFrame:
    """Return secondary FF12 classifications from Yahoo metadata rows."""

    if "symbol" not in metadata_frame.columns:
        raise ValueError("Yahoo metadata must contain a 'symbol' column")

    classification_records: list[dict[str, Any]] = []
    for metadata_record in metadata_frame.to_dict("records"):
        ticker_symbol = str(metadata_record.get("symbol", "")).strip().upper()
        if not ticker_symbol:
            continue
        quote_type = _clean_metadata_text(metadata_record.get("yf_quote_type"))
        if quote_type and quote_type != "equity":
            continue

        yfinance_sector = metadata_record.get("yf_sector")
        yfinance_industry = metadata_record.get("yf_industry")
        classification = classify_yfinance_sector_industry(
            yfinance_sector,
            yfinance_industry,
        )
        if classification is None:
            continue

        classification_records.append(
            {
                "ticker": ticker_symbol,
                "ff12": classification["ff12"],
                "ff_label": classification["ff_label"],
                "ff12_source": "secondary_yfinance",
                "classification_confidence": classification[
                    "classification_confidence"
                ],
                "classification_issue": "resolved_missing_sic",
                "secondary_sector": yfinance_sector,
                "secondary_industry": yfinance_industry,
                "secondary_source": "yfinance",
                "secondary_reason": classification["reason"],
            }
        )

    return pandas.DataFrame(classification_records)


def _load_secondary_classification_file(
    classification_path: Path = SECONDARY_CLASSIFICATIONS_CSV_PATH,
) -> pandas.DataFrame:
    """Load a secondary sector classification file if available."""

    if not classification_path.exists():
        return pandas.DataFrame()
    try:
        classification_frame = pandas.read_csv(classification_path)
    except (OSError, pandas.errors.ParserError) as read_error:
        LOGGER.warning(
            "Could not read secondary classifications %s: %s",
            classification_path,
            read_error,
        )
        return pandas.DataFrame()
    classification_frame.columns = [
        str(column_name).strip().lower()
        for column_name in classification_frame.columns
    ]
    required_columns = {"ticker", "ff12"}
    if not required_columns.issubset(classification_frame.columns):
        LOGGER.warning(
            "Secondary classifications %s must contain columns: %s",
            classification_path,
            sorted(required_columns),
        )
        return pandas.DataFrame()
    classification_frame["ticker"] = (
        classification_frame["ticker"].astype(str).str.strip().str.upper()
    )
    classification_frame["ff12"] = pandas.to_numeric(
        classification_frame["ff12"],
        errors="coerce",
    ).astype("Int64")
    classification_frame = classification_frame.dropna(subset=["ticker", "ff12"])
    classification_frame = classification_frame.loc[
        classification_frame["ticker"] != ""
    ]
    classification_frame = classification_frame.loc[
        classification_frame["ff12"] >= 1
    ]
    return classification_frame.drop_duplicates(subset=["ticker"], keep="last")


def _load_llm_classification_file(
    classification_path: Path = LLM_CLASSIFICATIONS_CSV_PATH,
) -> pandas.DataFrame:
    """Load LLM-reviewed classifications and keep only included symbols."""

    if not classification_path.exists():
        return pandas.DataFrame()
    try:
        classification_frame = pandas.read_csv(classification_path)
    except (OSError, pandas.errors.ParserError) as read_error:
        LOGGER.warning(
            "Could not read LLM classifications %s: %s",
            classification_path,
            read_error,
        )
        return pandas.DataFrame()
    classification_frame.columns = [
        str(column_name).strip().lower()
        for column_name in classification_frame.columns
    ]
    required_columns = {"ticker", "llm_decision", "ff12"}
    if not required_columns.issubset(classification_frame.columns):
        LOGGER.warning(
            "LLM classifications %s must contain columns: %s",
            classification_path,
            sorted(required_columns),
        )
        return pandas.DataFrame()

    included_classification_frame = classification_frame.loc[
        classification_frame["llm_decision"].astype(str).str.strip().str.lower()
        == "include"
    ].copy()
    if included_classification_frame.empty:
        return pandas.DataFrame()

    included_classification_frame["ticker"] = (
        included_classification_frame["ticker"].astype(str).str.strip().str.upper()
    )
    included_classification_frame["ff12"] = pandas.to_numeric(
        included_classification_frame["ff12"],
        errors="coerce",
    ).astype("Int64")
    included_classification_frame = included_classification_frame.dropna(
        subset=["ticker", "ff12"]
    )
    included_classification_frame = included_classification_frame.loc[
        included_classification_frame["ticker"] != ""
    ]
    included_classification_frame = included_classification_frame.loc[
        included_classification_frame["ff12"] >= 1
    ].copy()
    if included_classification_frame.empty:
        return pandas.DataFrame()

    included_classification_frame["ff12"] = included_classification_frame[
        "ff12"
    ].astype(int)
    if "ff_label" not in included_classification_frame.columns:
        included_classification_frame["ff_label"] = ""
    label_values: list[str] = []
    for classification_record in included_classification_frame.to_dict("records"):
        raw_label_value = classification_record.get("ff_label")
        if raw_label_value is not None and not pandas.isna(raw_label_value):
            label_value = str(raw_label_value).strip()
        else:
            label_value = ""
        if not label_value:
            label_value = FAMA_FRENCH_LABEL_BY_GROUP_IDENTIFIER[
                int(classification_record["ff12"])
            ]
        label_values.append(label_value)
    included_classification_frame["ff_label"] = label_values
    if "ff12_source" not in included_classification_frame.columns:
        included_classification_frame["ff12_source"] = "secondary_llm"
    included_classification_frame["ff12_source"] = included_classification_frame[
        "ff12_source"
    ].fillna("secondary_llm")
    if "classification_confidence" not in included_classification_frame.columns:
        included_classification_frame["classification_confidence"] = "medium"
    included_classification_frame["classification_confidence"] = (
        included_classification_frame["classification_confidence"]
        .fillna("medium")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    if "classification_issue" not in included_classification_frame.columns:
        included_classification_frame[
            "classification_issue"
        ] = "resolved_missing_sic_llm"
    included_classification_frame["classification_issue"] = (
        included_classification_frame["classification_issue"]
        .fillna("resolved_missing_sic_llm")
        .astype(str)
        .str.strip()
    )
    if "secondary_source" not in included_classification_frame.columns:
        included_classification_frame["secondary_source"] = "llm_review"
    if "secondary_reason" not in included_classification_frame.columns:
        included_classification_frame["secondary_reason"] = included_classification_frame.get(
            "reason",
            "",
        )
    if "reason" in included_classification_frame.columns:
        included_classification_frame["secondary_reason"] = (
            included_classification_frame["secondary_reason"]
            .fillna(included_classification_frame["reason"])
        )

    return included_classification_frame.drop_duplicates(
        subset=["ticker"],
        keep="last",
    )


def load_secondary_classifications(
    classification_path: Path = SECONDARY_CLASSIFICATIONS_CSV_PATH,
    llm_classification_path: Path = LLM_CLASSIFICATIONS_CSV_PATH,
) -> pandas.DataFrame:
    """Load reviewed secondary sector classifications if available.

    Yahoo-derived secondary classifications are deterministic source data.
    LLM-reviewed rows are kept in a separate audit file so regenerating the
    Yahoo file does not overwrite the manual review layer.
    """

    secondary_classification_frame = _load_secondary_classification_file(
        classification_path
    )
    llm_classification_frame = _load_llm_classification_file(llm_classification_path)
    classification_frames = [
        current_classification_frame
        for current_classification_frame in (
            secondary_classification_frame,
            llm_classification_frame,
        )
        if not current_classification_frame.empty
    ]
    if not classification_frames:
        return pandas.DataFrame()
    combined_classification_frame = pandas.concat(
        classification_frames,
        ignore_index=True,
        sort=False,
    )
    combined_classification_frame["ticker"] = (
        combined_classification_frame["ticker"].astype(str).str.strip().str.upper()
    )
    combined_classification_frame = combined_classification_frame.loc[
        combined_classification_frame["ticker"] != ""
    ]
    return combined_classification_frame.drop_duplicates(
        subset=["ticker"],
        keep="last",
    )


def apply_secondary_classifications(
    classified_data_frame: pandas.DataFrame,
    secondary_classification_frame: pandas.DataFrame | None = None,
) -> pandas.DataFrame:
    """Upgrade missing-SIC fallback rows with reviewed secondary FF12 data."""

    if secondary_classification_frame is None:
        secondary_classification_frame = load_secondary_classifications()
    if secondary_classification_frame.empty:
        return classified_data_frame

    normalized_secondary_frame = secondary_classification_frame.copy()
    normalized_secondary_frame.columns = [
        str(column_name).strip().lower()
        for column_name in normalized_secondary_frame.columns
    ]
    if "ticker" not in normalized_secondary_frame.columns:
        return classified_data_frame

    normalized_secondary_frame["ticker"] = (
        normalized_secondary_frame["ticker"].astype(str).str.strip().str.upper()
    )
    secondary_record_by_ticker = {
        str(record["ticker"]): record
        for record in normalized_secondary_frame.to_dict("records")
    }

    upgraded_data_frame = classified_data_frame.copy()
    for row_index, sector_record in upgraded_data_frame.iterrows():
        ticker_symbol = str(sector_record.get("ticker", "")).strip().upper()
        secondary_record = secondary_record_by_ticker.get(ticker_symbol)
        if secondary_record is None:
            continue
        current_issue = str(sector_record.get("classification_issue", ""))
        current_confidence = str(sector_record.get("classification_confidence", ""))
        if (
            current_issue != "missing_sic"
            and current_confidence.lower() != "low"
        ):
            continue

        confidence_value = str(
            secondary_record.get("classification_confidence", "medium")
        ).strip().lower()
        if confidence_value not in TRUSTED_CLASSIFICATION_CONFIDENCE_VALUES:
            continue

        try:
            group_identifier = int(secondary_record["ff12"])
        except (TypeError, ValueError):
            continue
        if group_identifier < 1:
            continue

        upgraded_data_frame.loc[row_index, "ff12"] = group_identifier
        upgraded_data_frame.loc[row_index, "ff_label"] = secondary_record.get(
            "ff_label",
            FAMA_FRENCH_LABEL_BY_GROUP_IDENTIFIER.get(
                group_identifier,
                f"Custom_{group_identifier}",
            ),
        )
        raw_source_value = secondary_record.get("ff12_source", "secondary_review")
        if raw_source_value is None or pandas.isna(raw_source_value):
            source_value = "secondary_review"
        else:
            source_value = str(raw_source_value).strip() or "secondary_review"
        upgraded_data_frame.loc[row_index, "ff12_source"] = source_value
        upgraded_data_frame.loc[
            row_index,
            "classification_confidence",
        ] = confidence_value
        upgraded_data_frame.loc[
            row_index,
            "classification_issue",
        ] = secondary_record.get("classification_issue", "resolved_missing_sic")
        for optional_column_name in (
            "secondary_sector",
            "secondary_industry",
            "secondary_source",
            "secondary_reason",
        ):
            if optional_column_name in normalized_secondary_frame.columns:
                upgraded_data_frame.loc[row_index, optional_column_name] = (
                    secondary_record.get(optional_column_name, "")
                )

    return upgraded_data_frame
