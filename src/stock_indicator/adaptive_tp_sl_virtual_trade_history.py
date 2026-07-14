"""State boundary for the ADAPTIVE TP/SL virtual trade history.

This subsystem follows counterfactual raw strategy trades solely to supply the
rolling return samples used by ADAPTIVE TP/SL.  It is not a portfolio, a Futu
position mirror, an order-allocation ledger, or a performance account.

Keeping the complete purpose token in the module and state namespace prevents
generic names such as ``accepted_entries`` or ``closed_trades`` from being
mistaken for live holdings.
"""

from __future__ import annotations

from typing import Any


ADAPTIVE_TP_SL_VIRTUAL_TRADE_HISTORY_KEY = (
    "adaptive_tp_sl_virtual_trade_history"
)
ADAPTIVE_TP_SL_VIRTUAL_TRADE_HISTORY_SCHEMA_VERSION = 1

ADAPTIVE_TP_SL_VIRTUAL_OPEN_TRADES_KEY = "open_trades"
ADAPTIVE_TP_SL_VIRTUAL_CLOSED_TRADES_KEY = "closed_trades"
ADAPTIVE_TP_SL_VIRTUAL_PENDING_RETURNS_KEY = "pending_returns"
ADAPTIVE_TP_SL_VIRTUAL_WINNER_RETURNS_KEY = "winner_returns"
ADAPTIVE_TP_SL_VIRTUAL_LOSER_RETURNS_KEY = "loser_returns"
ADAPTIVE_TP_SL_VIRTUAL_RAW_RETURNS_KEY = "raw_returns"

LEGACY_TOP_LEVEL_KEY_BY_HISTORY_KEY = {
    ADAPTIVE_TP_SL_VIRTUAL_OPEN_TRADES_KEY: "accepted_entries",
    ADAPTIVE_TP_SL_VIRTUAL_CLOSED_TRADES_KEY: "closed_trades",
    ADAPTIVE_TP_SL_VIRTUAL_PENDING_RETURNS_KEY: "pending_rolling",
    ADAPTIVE_TP_SL_VIRTUAL_WINNER_RETURNS_KEY: "winners",
    ADAPTIVE_TP_SL_VIRTUAL_LOSER_RETURNS_KEY: "losers",
    ADAPTIVE_TP_SL_VIRTUAL_RAW_RETURNS_KEY: "raw_trade_profits",
}


# TODO: review
def empty_adaptive_tp_sl_virtual_trade_history() -> dict[str, Any]:
    """Return an empty ADAPTIVE TP/SL statistical reference history."""

    return {
        "schema_version": ADAPTIVE_TP_SL_VIRTUAL_TRADE_HISTORY_SCHEMA_VERSION,
        ADAPTIVE_TP_SL_VIRTUAL_OPEN_TRADES_KEY: [],
        ADAPTIVE_TP_SL_VIRTUAL_CLOSED_TRADES_KEY: [],
        ADAPTIVE_TP_SL_VIRTUAL_PENDING_RETURNS_KEY: [],
        ADAPTIVE_TP_SL_VIRTUAL_WINNER_RETURNS_KEY: [],
        ADAPTIVE_TP_SL_VIRTUAL_LOSER_RETURNS_KEY: [],
        ADAPTIVE_TP_SL_VIRTUAL_RAW_RETURNS_KEY: [],
    }


def get_adaptive_tp_sl_virtual_trade_history(
    state_document: dict[str, Any],
) -> dict[str, Any]:
    """Return the namespaced history and migrate legacy top-level fields.

    The migration is in-memory and deterministic.  The caller's normal atomic
    state save persists the new namespace; this helper never writes a file.
    Once a namespaced value exists it is authoritative, so stale legacy fields
    cannot silently override it.
    """

    history_value = state_document.get(
        ADAPTIVE_TP_SL_VIRTUAL_TRADE_HISTORY_KEY
    )
    if not isinstance(history_value, dict):
        history_value = {
            "schema_version": (
                ADAPTIVE_TP_SL_VIRTUAL_TRADE_HISTORY_SCHEMA_VERSION
            )
        }
        state_document[ADAPTIVE_TP_SL_VIRTUAL_TRADE_HISTORY_KEY] = history_value

    history_value.setdefault(
        "schema_version",
        ADAPTIVE_TP_SL_VIRTUAL_TRADE_HISTORY_SCHEMA_VERSION,
    )
    for history_key, legacy_top_level_key in (
        LEGACY_TOP_LEVEL_KEY_BY_HISTORY_KEY.items()
    ):
        if history_key not in history_value:
            legacy_value = state_document.get(legacy_top_level_key)
            history_value[history_key] = (
                legacy_value if isinstance(legacy_value, list) else []
            )
        state_document.pop(legacy_top_level_key, None)

    return history_value
