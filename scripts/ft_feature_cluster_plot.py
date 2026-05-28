"""Visualize ft true-winners vs false-positives in 2D feature space.

Loads σ=0 sweep CSV (production-current). Classifies each ft trade
into one of four classes and scatters them across the strongest
discriminator pairs from the earlier quartile comparison:

  price_tightness × ema_angle
  price_tightness × d_ema_angle
  ema_angle       × d_ema_angle
  slope_60        × d_ema_angle

Each plot annotates the FP q75 line (upper bound of "false positive
typical range") so we can see how many true winners actually live
above that line in the relevant direction.

Output: logs/ft_feature_clusters/*.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

CSV = (
    Path(__file__).resolve().parent.parent
    / "logs"
    / "multi_bucket_simulation_result"
    / "multi_bucket_simulation_20260519_040339.csv"
)
OUT_DIR = Path(__file__).resolve().parent.parent / "logs" / "ft_feature_clusters"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Classification thresholds
TW_MFE_THRESHOLD = 0.08          # true winner = MFE >= 8% (sigma-independent)
FP_PCT_RANGE = (-0.05, -0.01)    # false positive = small loss
DEEP_LOSER_PCT = -0.05

FEATURES_TO_LOAD = [
    "price_tightness", "ema_angle", "d_ema_angle", "slope_60",
    "near_delta", "sma_angle",
]

PAIR_PLOTS = [
    ("price_tightness", "ema_angle"),
    ("price_tightness", "d_ema_angle"),
    ("ema_angle", "d_ema_angle"),
    ("slope_60", "d_ema_angle"),
]


def load_ft_trades() -> list[dict]:
    rows = []
    with CSV.open() as f:
        for row in csv.DictReader(f):
            if row["bucket"] != "fish_tail_production":
                continue
            rows.append(row)
    return rows


def classify(row: dict) -> str | None:
    try:
        pct = float(row["percentage_change"])
        mfe = float(row["max_favorable_excursion_pct"])
    except (TypeError, ValueError, KeyError):
        return None
    exit_reason = row.get("exit_reason", "")
    if mfe >= TW_MFE_THRESHOLD:
        return "TW"
    if FP_PCT_RANGE[0] <= pct <= FP_PCT_RANGE[1] and exit_reason == "signal":
        return "FP"
    if pct < DEEP_LOSER_PCT:
        return "DEEP_LOSS"
    if 0 < pct < TW_MFE_THRESHOLD:
        return "SMALL_WIN"
    return None


def feature_value(row: dict, feature_name: str) -> float | None:
    raw = row.get(feature_name)
    if raw in (None, "", "nan"):
        return None
    try:
        value = float(raw)
        if value != value:  # NaN
            return None
        return value
    except (TypeError, ValueError):
        return None


def main() -> int:
    rows = load_ft_trades()
    classes: dict[str, list[dict]] = {
        "TW": [], "FP": [], "DEEP_LOSS": [], "SMALL_WIN": []
    }
    for row in rows:
        klass = classify(row)
        if klass is None:
            continue
        classes[klass].append(row)
    print(f"ft 1994-2026 (σ=0.0) classification:")
    for k, v in classes.items():
        print(f"  {k:10}: {len(v):>5}")

    style = {
        "TW":        {"color": "#1f9d55", "alpha": 0.8, "size": 28, "marker": "o", "label": "True Winner (MFE>=+8%)"},
        "FP":        {"color": "#cc1f1a", "alpha": 0.5, "size": 16, "marker": "x", "label": "False Positive (-5%~-1% signal)"},
        "DEEP_LOSS": {"color": "#7a1219", "alpha": 0.4, "size": 14, "marker": "v", "label": "Deep Loser (<-5%)"},
        "SMALL_WIN": {"color": "#a6a6a6", "alpha": 0.18, "size": 9, "marker": ".", "label": "Small Winner (0~+8%)"},
    }

    for fx, fy in PAIR_PLOTS:
        fig, ax = plt.subplots(figsize=(8, 6))
        # Draw lighter classes first so they sit behind
        draw_order = ["SMALL_WIN", "DEEP_LOSS", "FP", "TW"]
        for klass in draw_order:
            rows_k = classes[klass]
            xs = []
            ys = []
            for r in rows_k:
                vx = feature_value(r, fx)
                vy = feature_value(r, fy)
                if vx is None or vy is None:
                    continue
                xs.append(vx)
                ys.append(vy)
            ax.scatter(xs, ys, c=style[klass]["color"],
                       alpha=style[klass]["alpha"],
                       s=style[klass]["size"],
                       marker=style[klass]["marker"],
                       label=f"{style[klass]['label']} (n={len(xs)})")

        # Compute FP q75 line for each axis to mark "above FP typical range"
        fp_xs = [feature_value(r, fx) for r in classes["FP"]]
        fp_xs = sorted(v for v in fp_xs if v is not None)
        fp_ys = [feature_value(r, fy) for r in classes["FP"]]
        fp_ys = sorted(v for v in fp_ys if v is not None)
        if fp_xs and fp_ys:
            qx = fp_xs[(3 * len(fp_xs)) // 4]
            qy = fp_ys[(3 * len(fp_ys)) // 4]
            ax.axvline(qx, color="gray", linestyle="--", alpha=0.4,
                       label=f"FP q75 {fx} = {qx:.3f}")
            ax.axhline(qy, color="gray", linestyle=":", alpha=0.4,
                       label=f"FP q75 {fy} = {qy:.3f}")

        ax.set_xlabel(fx)
        ax.set_ylabel(fy)
        ax.set_title(
            f"ft entry features: {fx} × {fy}\n"
            f"1994-2026 σ=0 baseline"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        out_path = OUT_DIR / f"ft_{fx}__x__{fy}.png"
        plt.savefig(out_path, dpi=110)
        plt.close()
        print(f"  wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
