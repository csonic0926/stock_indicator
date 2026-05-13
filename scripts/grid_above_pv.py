"""Grid scan over fish_head's above_pv band.

Hypothesis (2026-05-11): fish_head_vacuum_turn at above_pv >= 0.973 is a
saturation point for iceberg detection; there may be a separate
above_pv band in [0.10, 0.95] that yields a distinct fish_head sub-type
(mid-cycle iceberg + cross-up confirmation) when combined with
pre_cross_signal_lookback + free_fall slope/ND filters.

For each 0.05-wide band, we run a standalone fish_head sim (single
bucket occupying all max_position slots, 17-year 2010_safe universe).
A band has 'meat' if Calmar / P/L are competitive with the 0.973
baseline.

Outputs:
- logs/grid_above_pv/sim_fish_head_b{lo}_{hi}.log  (per-band sim stdout)
- logs/grid_above_pv/cfg_fish_head_b{lo}_{hi}.json (per-band config)
- logs/grid_above_pv/REPORT.md   (aggregate Markdown table)
- logs/grid_above_pv/results.json (parsed metrics)

Idempotent: if a sim log already contains a yearly breakdown, that
band is skipped on re-run.
"""
from __future__ import annotations

import copy
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE_CONFIG = REPO / "data" / "multi_bucket_production_test.json"
STRATEGY_SETS = REPO / "data" / "strategy_sets.csv"
GRID_DIR = REPO / "logs" / "grid_above_pv"
GRID_DIR.mkdir(parents=True, exist_ok=True)
VENV_PY = REPO / "venv" / "bin" / "python"

BANDS = [
    (0.10, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 0.30),
    (0.30, 0.35), (0.35, 0.40), (0.40, 0.45), (0.45, 0.50),
    (0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
    (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90),
    (0.90, 0.95),
    (0.973, 1.00),  # baseline (= existing fish_head_vacuum_turn)
]

SUMMARY_RE = re.compile(
    r"\[(Total|fish_head_[^\]]+)\] "
    r"Trades: (\d+), Win rate: ([\d.]+)%, Mean profit %: ([\d.-]+)%, "
    r"Profit % Std Dev: ([\d.]+)%, Mean loss %: ([\d.-]+)%, "
    r"Loss % Std Dev: ([\d.]+)%, P/L: ([\d.]+), "
    r"Mean holding period: ([\d.]+) bars[^,]*, "
    r"Holding period Std Dev: ([\d.]+) bars, "
    r"Max concurrent positions: (\d+), Final balance: ([\d.]+), "
    r"CAGR: ([\d.-]+)%, Max drawdown: ([\d.]+)%"
)


def band_id(lo: float, hi: float) -> str:
    if (lo, hi) == (0.973, 1.00):
        return "fish_head_vacuum_turn"  # existing entry
    return f"fish_head_b{int(round(lo * 100)):02d}_{int(round(hi * 100)):02d}"


def ensure_strategy_set_rows() -> None:
    """Append missing band rows to strategy_sets.csv (idempotent)."""
    existing = set()
    with STRATEGY_SETS.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            sid = (row.get("strategy_id") or "").strip()
            if sid:
                existing.add(sid)
    rows_to_add = []
    for lo, hi in BANDS:
        sid = band_id(lo, hi)
        if sid in existing:
            continue
        # Buy/sell pattern mirrors fish_head_vacuum_turn except above_range.
        buy = (
            f"ema_sma_cross_testing_3_-99_99_-99.0,99.0_"
            f"{lo:.2f},{hi:.2f}"
        )
        sell = "ema_sma_cross_testing_3_-0.01_65_-10.0,10.0_0.78,1.00"
        rows_to_add.append((sid, buy, sell))
    if not rows_to_add:
        return
    # Append rows preserving CSV quoting.
    with STRATEGY_SETS.open("a", encoding="utf-8", newline="") as fp:
        # If file doesn't end with newline, prepend one.
        # Read back the tail to decide.
        pass
    raw = STRATEGY_SETS.read_bytes()
    needs_newline = not raw.endswith(b"\n")
    with STRATEGY_SETS.open("a", encoding="utf-8") as fp:
        if needs_newline:
            fp.write("\n")
        writer = csv.writer(fp, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        for sid, buy, sell in rows_to_add:
            # 15 trailing empty cols to match header (18 cols total).
            writer.writerow([sid, buy, sell] + [""] * 15)
    print(
        f"[grid_above_pv] Added {len(rows_to_add)} rows to strategy_sets.csv",
        flush=True,
    )


def make_cfg(strategy_id: str, cfg_path: Path) -> None:
    """Write a standalone single-bucket config using the given strategy_id."""
    cfg = json.loads(BASE_CONFIG.read_text())
    fh = copy.deepcopy(cfg["buckets"][0])
    fh["strategy_id"] = strategy_id
    fh["label"] = strategy_id
    fh["max_positions"] = cfg["max_position_count"]
    cfg["buckets"] = [fh]
    cfg_path.write_text(json.dumps(cfg, indent=2))


def run_sim(cfg_path: Path, log_path: Path) -> int:
    cmd = (
        f'cd "{REPO}" && "{VENV_PY}" -m stock_indicator.manage <<< '
        f'"multi_bucket_simulation {cfg_path.relative_to(REPO)}\nexit" '
        f'> "{log_path}" 2>&1'
    )
    return subprocess.call(["/bin/bash", "-c", cmd])


def parse_summary(log_path: Path) -> dict:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    out: dict = {}
    for m in SUMMARY_RE.finditer(text):
        label = m.group(1)
        out[label] = {
            "trades": int(m.group(2)),
            "wr": float(m.group(3)) / 100,
            "mp": float(m.group(4)) / 100,
            "ml": float(m.group(6)) / 100,
            "pl": float(m.group(8)),
            "hold": float(m.group(9)),
            "final_balance": float(m.group(12)),
            "cagr": float(m.group(13)) / 100,
            "mdd": float(m.group(14)) / 100,
        }
    yearly = {}
    for m in re.finditer(r"\[Total\] Year (\d{4}): ([\d.-]+)%", text):
        yearly[int(m.group(1))] = float(m.group(2)) / 100
    out["__yearly_total"] = yearly
    out["__neg_years"] = sorted([y for y, r in yearly.items() if r < 0])
    return out


def is_done(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return "[Total] Trades:" in text and "[Total] Year" in text


def build_report(results: dict) -> str:
    lines = [
        "# above_pv band grid scan",
        "",
        "Standalone single-bucket fish_head with pre_cross_signal_lookback +",
        "free_fall slope/ND filters, varying above_pv band only. Universe:",
        "2010_safe, 17 years. max_positions=6, sigma=0.75.",
        "",
        "| Band | strategy_id | Trades | WR | MP | ML | P/L | Hold | CAGR | MDD | Calmar | NegYrs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    rows_sorted = sorted(BANDS)
    for lo, hi in rows_sorted:
        sid = band_id(lo, hi)
        r = results.get(sid, {})
        total = r.get("Total")
        if not total:
            lines.append(
                f"| {lo:.3f}-{hi:.3f} | {sid} | — | | | | | | | | | |"
            )
            continue
        cagr = total["cagr"]
        mdd = total["mdd"]
        calmar = cagr / mdd if mdd > 0 else float("nan")
        neg = r.get("__neg_years", [])
        neg_str = ",".join(str(y) for y in neg) if neg else "-"
        lines.append(
            f"| {lo:.3f}-{hi:.3f} | {sid} | {total['trades']} | "
            f"{total['wr']:.1%} | {total['mp']:.2%} | {total['ml']:.2%} | "
            f"{total['pl']:.2f} | {total['hold']:.1f} | {cagr:.2%} | "
            f"{mdd:.2%} | {calmar:.3f} | {neg_str} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    print(f"[grid_above_pv] Start at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    ensure_strategy_set_rows()
    results: dict = {}
    for index, (lo, hi) in enumerate(BANDS, start=1):
        sid = band_id(lo, hi)
        cfg_path = GRID_DIR / f"cfg_{sid}.json"
        log_path = GRID_DIR / f"sim_{sid}.log"
        if is_done(log_path):
            print(
                f"[grid_above_pv] [{index}/{len(BANDS)}] {sid} skip (done)",
                flush=True,
            )
        else:
            make_cfg(sid, cfg_path)
            t0 = time.time()
            print(
                f"[grid_above_pv] [{index}/{len(BANDS)}] {sid} start",
                flush=True,
            )
            rc = run_sim(cfg_path, log_path)
            elapsed = time.time() - t0
            print(
                f"[grid_above_pv] [{index}/{len(BANDS)}] {sid} rc={rc} "
                f"({elapsed:.0f}s)",
                flush=True,
            )
        results[sid] = parse_summary(log_path)
        # Incremental report write so partial results visible.
        (GRID_DIR / "results.json").write_text(
            json.dumps(results, indent=2, default=str)
        )
        (GRID_DIR / "REPORT.md").write_text(build_report(results))
    print(
        f"[grid_above_pv] Done at {time.strftime('%Y-%m-%d %H:%M:%S')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
