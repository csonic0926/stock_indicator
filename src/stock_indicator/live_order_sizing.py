"""Shared live-order quantity calculations."""

from __future__ import annotations

# TODO: review

import math

DEFAULT_MARGIN_MULTIPLIER = 1.5
HKD_PER_USD = 7.8


def compute_target_share_quantity(
    *,
    total_assets_hkd: float,
    price_usd: float,
    maximum_position_count: int,
    margin_multiplier: float = DEFAULT_MARGIN_MULTIPLIER,
) -> int:
    """Return the strategy target quantity for one equal-weight position."""
    if price_usd <= 0 or maximum_position_count <= 0:
        return 0
    hkd_per_position = (
        total_assets_hkd * margin_multiplier / maximum_position_count
    )
    usd_per_position = hkd_per_position / HKD_PER_USD
    return math.floor(usd_per_position / price_usd)
