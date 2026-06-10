"""Probe: does pre-surge BSV footprint density separate ft B (deep-MAE winner)
from C (deep-MAE loser)?

Hypothesis (freedom-degree frame, 2026-06-10): B and C share identical surge
shape because the surge is the voluntary phase; the constraint that makes B
recover lives outside the surge window. BSV footprint = validated
institutional-presence detector (orthogonal to price slope), so pre-surge
footprint density should be higher for B than C if "機構在" is the real divider.

Windows (relative to entry bar at index 0, all strictly before entry):
  recent20 : bars [-20, -1]   — includes the surge
  pre20    : bars [-30, -11]  — surge excluded (assumes surge ~ last 10 bars)
  pre40    : bars [-50, -11]  — longer accumulation lookback, surge excluded

Bonus probe (squeeze fuel): return over bars [-60, -11] and max drawdown within
that window — was the surge preceded by a decline that trapped shorts?
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / "data" / "stock_data_2010_yf_clean"
TRADES_CSV = (
    REPO / "logs" / "multi_bucket_simulation_result"
    / "multi_bucket_simulation_20260605_153449.csv"
)

MAE_DEEP_THRESHOLD = -0.04
BALANCE_THRESHOLDS = (0.15, 0.10)

WINDOWS = {
    "recent20": (-20, -1),
    "pre20": (-30, -11),
    "pre40": (-50, -11),
}


def load_symbol(symbol: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"{symbol}.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["Date"])
    frame = frame.set_index("Date").sort_index()
    return frame


def balance_series(frame: pd.DataFrame) -> pd.Series:
    span = frame["high"] - frame["low"]
    with np.errstate(divide="ignore", invalid="ignore"):
        imbalance = (2.0 * frame["close"] - frame["high"] - frame["low"]) / span
    return imbalance.abs()


def window_features(frame: pd.DataFrame, entry_pos: int) -> dict | None:
    features: dict[str, float] = {}
    imbalance = balance_series(frame)
    for name, (start, end) in WINDOWS.items():
        lo, hi = entry_pos + start, entry_pos + end
        if lo < 0:
            return None
        window_imbalance = imbalance.iloc[lo : hi + 1]
        window_volume = frame["volume"].iloc[lo : hi + 1]
        for threshold in BALANCE_THRESHOLDS:
            flagged = (window_imbalance < threshold).astype(float)
            features[f"{name}_density_t{threshold}"] = flagged.mean()
            total_volume = window_volume.sum()
            if total_volume > 0:
                features[f"{name}_vw_density_t{threshold}"] = (
                    (flagged * window_volume).sum() / total_volume
                )
            else:
                features[f"{name}_vw_density_t{threshold}"] = np.nan
    fuel_lo, fuel_hi = entry_pos - 60, entry_pos - 11
    if fuel_lo >= 0:
        closes = frame["close"].iloc[fuel_lo : fuel_hi + 1]
        features["fuel_return"] = closes.iloc[-1] / closes.iloc[0] - 1.0
        running_max = closes.cummax()
        features["fuel_drawdown"] = (closes / running_max - 1.0).min()
    return features


def auc_from_u(u_statistic: float, n_positive: int, n_negative: int) -> float:
    return u_statistic / (n_positive * n_negative)


def main() -> None:
    trades = pd.read_csv(TRADES_CSV, parse_dates=["entry_date"])
    fish_tail = trades[trades["bucket"] == "fish_tail_production"].copy()
    deep = fish_tail[
        fish_tail["max_adverse_excursion_pct"] <= MAE_DEEP_THRESHOLD
    ].copy()
    deep["group"] = np.where(deep["result"] == "win", "B", "C")
    print(
        f"deep-MAE ft trades (MAE<={MAE_DEEP_THRESHOLD}): "
        f"{len(deep)}  B={len(deep[deep.group=='B'])} C={len(deep[deep.group=='C'])}"
    )

    rows = []
    missing_symbols: set[str] = set()
    skipped = 0
    for trade in deep.itertuples():
        frame = load_symbol(trade.symbol)
        if frame is None:
            missing_symbols.add(trade.symbol)
            continue
        positions = frame.index.get_indexer([trade.entry_date])
        if positions[0] == -1:
            skipped += 1
            continue
        features = window_features(frame, positions[0])
        if features is None:
            skipped += 1
            continue
        features["group"] = trade.group
        features["above_price_volume_ratio"] = trade.above_price_volume_ratio
        rows.append(features)

    table = pd.DataFrame(rows)
    print(
        f"scored={len(table)}  missing_symbols={len(missing_symbols)}  "
        f"skipped_no_bar_or_history={skipped}"
    )
    print()

    b_mask = table["group"] == "B"
    feature_columns = [c for c in table.columns if c != "group"]
    results = []
    for column in feature_columns:
        b_values = table.loc[b_mask, column].dropna()
        c_values = table.loc[~b_mask, column].dropna()
        if len(b_values) < 10 or len(c_values) < 10:
            continue
        u_statistic, p_value = mannwhitneyu(
            b_values, c_values, alternative="two-sided"
        )
        results.append(
            {
                "feature": column,
                "B_mean": b_values.mean(),
                "C_mean": c_values.mean(),
                "B_median": b_values.median(),
                "C_median": c_values.median(),
                "AUC_B_vs_C": auc_from_u(u_statistic, len(b_values), len(c_values)),
                "p": p_value,
            }
        )
    report = pd.DataFrame(results).sort_values("p")
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(report.to_string(index=False))

    table.to_csv(REPO / "data" / "ft_bsv_precursor_features.csv", index=False)
    print("\nfeature table saved to data/ft_bsv_precursor_features.csv")


if __name__ == "__main__":
    main()
