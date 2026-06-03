"""Compact Futu order remark metadata for live trade management."""

from __future__ import annotations

from typing import Any

MAX_FUTU_REMARK_BYTES = 64

STRATEGY_ID_TO_REMARK_CODE = {
    "fish_head_vacuum_turn": "h",
    "fish_tail_blow_off_top": "t",
    "fish_head_b30_35": "b",
}
REMARK_CODE_TO_STRATEGY_ID = {
    remark_code: strategy_id
    for strategy_id, remark_code in STRATEGY_ID_TO_REMARK_CODE.items()
}
STRATEGY_ID_TO_DEFAULT_BUCKET = {
    "fish_head_vacuum_turn": "fish_head_production",
    "fish_tail_blow_off_top": "fish_tail_production",
    "fish_head_b30_35": "fish_head_b30_35",
}


def _pct_to_basis_points(percent_value: Any) -> int | None:
    """Convert a decimal percent value such as 0.0658 to basis points."""
    if percent_value is None:
        return None
    try:
        return int(round(float(percent_value) * 10_000))
    except (TypeError, ValueError):
        return None


def _basis_points_to_pct(basis_points_value: str) -> float | None:
    """Convert basis points from remark text to a decimal percent."""
    try:
        return int(basis_points_value) / 10_000
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    """Convert optional integer-like metadata without hiding invalid text."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    """Convert optional metadata to bool while preserving missing values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes"}


def format_futu_order_remark(order: dict[str, Any]) -> str:
    """Build a v2 Futu BUY order remark carrying live-management metadata.

    Futu restricts remarks to 64 UTF-8 bytes, so the wire schema is compact:
    si2|s=h|tp=658|sl=417|ms=1|ds=1|mh=14|rr=1
    """
    strategy_identifier = str(order.get("strategy_id") or "")
    strategy_code = STRATEGY_ID_TO_REMARK_CODE.get(strategy_identifier)
    if strategy_code is None:
        raise ValueError(f"unsupported strategy_id for Futu remark: {strategy_identifier}")

    take_profit_basis_points = _pct_to_basis_points(order.get("tp_pct"))
    stop_loss_basis_points = _pct_to_basis_points(order.get("sl_pct"))
    if take_profit_basis_points is None or stop_loss_basis_points is None:
        raise ValueError("BUY order requires tp_pct and sl_pct for Futu remark")

    min_hold_stop_loss = _optional_int(order.get("min_hold_sl"))
    if min_hold_stop_loss is None:
        min_hold_stop_loss = 1
    disable_stop_loss_trigger = 1 if _optional_bool(
        order.get("disable_sl_trigger")
    ) else 0
    reset_hold_value = 1 if _optional_bool(
        order.get("reset_hold_on_reentry_signal")
    ) else 0

    parts = [
        "si2",
        f"s={strategy_code}",
        f"tp={take_profit_basis_points}",
        f"sl={stop_loss_basis_points}",
        f"ms={min_hold_stop_loss}",
        f"ds={disable_stop_loss_trigger}",
    ]
    max_hold = _optional_int(order.get("max_hold"))
    if max_hold is not None:
        parts.append(f"mh={max_hold}")
    parts.append(f"rr={reset_hold_value}")

    remark_text = "|".join(parts)
    if len(remark_text.encode("utf-8")) > MAX_FUTU_REMARK_BYTES:
        raise ValueError(f"Futu remark exceeds {MAX_FUTU_REMARK_BYTES} bytes: {remark_text}")
    return remark_text


def parse_futu_order_remark(remark_text: str) -> dict[str, Any]:
    """Parse v2 remarks and legacy v1 max-hold-only remarks."""
    if not remark_text:
        return {}
    tokens = str(remark_text).split("|")
    if not tokens:
        return {}
    version_token = tokens[0]
    if version_token == "si2":
        return _parse_v2_tokens(tokens[1:])
    if version_token == "si":
        return _parse_legacy_tokens(tokens[1:])
    return {}


def _parse_v2_tokens(token_texts: list[str]) -> dict[str, Any]:
    """Parse the compact v2 live-management remark schema."""
    raw_values: dict[str, str] = {}
    for token_text in token_texts:
        key_text, separator, value_text = token_text.partition("=")
        if separator:
            raw_values[key_text] = value_text

    strategy_code = raw_values.get("s", "")
    strategy_identifier = REMARK_CODE_TO_STRATEGY_ID.get(strategy_code)
    metadata: dict[str, Any] = {"remark_version": "si2"}
    if strategy_identifier is not None:
        metadata["strategy_id"] = strategy_identifier
        metadata["bucket"] = STRATEGY_ID_TO_DEFAULT_BUCKET.get(strategy_identifier)

    take_profit_pct = _basis_points_to_pct(raw_values.get("tp", ""))
    stop_loss_pct = _basis_points_to_pct(raw_values.get("sl", ""))
    if take_profit_pct is not None:
        metadata["tp_pct"] = take_profit_pct
    if stop_loss_pct is not None:
        metadata["sl_pct"] = stop_loss_pct

    integer_fields = {
        "ms": "min_hold_sl",
        "mh": "max_hold",
    }
    for remark_key, metadata_key in integer_fields.items():
        parsed_value = _optional_int(raw_values.get(remark_key))
        if parsed_value is not None:
            metadata[metadata_key] = parsed_value

    if "ds" in raw_values:
        metadata["disable_sl_trigger"] = raw_values["ds"] == "1"
    if "rr" in raw_values:
        metadata["reset_hold_on_reentry_signal"] = raw_values["rr"] == "1"
    metadata["supports_tp_sl"] = "tp_pct" in metadata and "sl_pct" in metadata
    return metadata


def _parse_legacy_tokens(token_texts: list[str]) -> dict[str, Any]:
    """Parse old dashboard remarks for max-hold compatibility only."""
    metadata: dict[str, Any] = {"remark_version": "si", "supports_tp_sl": False}
    for token_text in token_texts:
        key_text, separator, value_text = token_text.partition("=")
        if not separator:
            continue
        if key_text == "sid":
            metadata["strategy_id"] = value_text
        elif key_text == "b":
            metadata["bucket"] = value_text
        elif key_text == "mh":
            parsed_value = _optional_int(value_text)
            if parsed_value is not None:
                metadata["max_hold"] = parsed_value
        elif key_text == "rr":
            metadata["reset_hold_on_reentry_signal"] = value_text == "1"
    return metadata
