"""Re-run grid peak standalone sims after disable_sl_trigger fix.

Background: 2026-05-12 found production_test.json had
`disable_sl_trigger: false` left over from SL exploration period.
Combined with min_sl=0.01, SL was firing aggressively in production_test
sims. Fix: set disable_sl_trigger=true (SL fully off, aligned with
project_sl_structural_impossibility_2026_05_11).

The grid scan (scripts/grid_above_pv.py) ran 18 bands standalone, all
with the regression. Cal: don't redo full grid, just redo the peaks.

Peaks identified pre-fix:
  - fish_head_b30_35  (Calmar 0.137 pre-fix)
  - fish_head_b50_55  (Calmar 0.149 pre-fix)
  - fish_head_b75_80  (Calmar 0.118 pre-fix)
  - fish_head_b85_90  (Calmar 0.130 pre-fix)
  - fish_head_vacuum_turn (Calmar 0.284 pre-fix, the 0.973-1.0 band)

Re-runs use logs/grid_above_pv_postfix/ to keep old data intact.
"""
from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE_CONFIG = REPO / "data" / "multi_bucket_production_test.json"
RERUN_DIR = REPO / "logs" / "grid_above_pv_postfix"
RERUN_DIR.mkdir(parents=True, exist_ok=True)
VENV_PY = REPO / "venv" / "bin" / "python"

PEAKS = [
    "fish_head_b30_35",
    "fish_head_b50_55",
    "fish_head_b75_80",
    "fish_head_b85_90",
    "fish_head_vacuum_turn",
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


def make_cfg(strategy_id: str, cfg_path: Path) -> None:
    """Standalone single-bucket config: replace fh bucket with target strategy."""
    cfg = json.loads(BASE_CONFIG.read_text())
    fh = copy.deepcopy(cfg["buckets"][0])  # fish_head_production template
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
        out[m.group(1)] = {
            "trades": int(m.group(2)),
            "wr": float(m.group(3)) / 100,
            "mp": float(m.group(4)) / 100,
            "ml": float(m.group(6)) / 100,
            "pl": float(m.group(8)),
            "hold": float(m.group(9)),
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
        "# Peak re-run post disable_sl_trigger=true fix",
        "",
        "Standalone single-bucket sims, post-SL-fix. Compare to pre-fix",
        "values from logs/grid_above_pv/REPORT.md.",
        "",
        "| Strategy | Trades | WR | MP | ML | P/L | CAGR | MDD | Calmar | NegYrs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    pre_fix = {
        "fish_head_b30_35": (1491, 0.451, 0.0360, 0.2631, 0.137),
        "fish_head_b50_55": (1328, 0.456, 0.0334, 0.2236, 0.149),
        "fish_head_b75_80": (1348, 0.430, 0.0312, 0.2636, 0.118),
        "fish_head_b85_90": (1360, 0.456, 0.0426, 0.3284, 0.130),
        "fish_head_vacuum_turn": (1414, 0.470, 0.0801, 0.2821, 0.284),
    }
    for sid in PEAKS:
        r = results.get(sid, {})
        total = r.get("Total")
        if not total:
            lines.append(f"| {sid} | — | | | | | | | | |")
            continue
        cagr = total["cagr"]
        mdd = total["mdd"]
        calmar = cagr / mdd if mdd > 0 else float("nan")
        neg = r.get("__neg_years", [])
        neg_str = ",".join(str(y) for y in neg) if neg else "-"
        lines.append(
            f"| {sid} | {total['trades']} | {total['wr']:.1%} | "
            f"{total['mp']:.2%} | {total['ml']:.2%} | {total['pl']:.2f} | "
            f"{cagr:.2%} | {mdd:.2%} | **{calmar:.3f}** | {neg_str} |"
        )

    lines.append("")
    lines.append("## Pre-fix comparison (reference)")
    lines.append("")
    lines.append("| Strategy | Pre-fix Trades | Pre-fix CAGR | Pre-fix MDD | Pre-fix Calmar |")
    lines.append("|---|---:|---:|---:|---:|")
    for sid in PEAKS:
        t, w, c, m, cal = pre_fix[sid]
        lines.append(f"| {sid} | {t} | {c:.2%} | {m:.2%} | {cal:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    print(f"[peaks_rerun] Start {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"[peaks_rerun] Base: {BASE_CONFIG}", flush=True)
    results: dict = {}
    for idx, sid in enumerate(PEAKS, 1):
        cfg_path = RERUN_DIR / f"cfg_{sid}.json"
        log_path = RERUN_DIR / f"sim_{sid}.log"
        if is_done(log_path):
            print(f"[peaks_rerun] [{idx}/{len(PEAKS)}] {sid} skip (done)", flush=True)
        else:
            make_cfg(sid, cfg_path)
            t0 = time.time()
            print(f"[peaks_rerun] [{idx}/{len(PEAKS)}] {sid} start", flush=True)
            rc = run_sim(cfg_path, log_path)
            elapsed = time.time() - t0
            print(
                f"[peaks_rerun] [{idx}/{len(PEAKS)}] {sid} rc={rc} "
                f"({elapsed:.0f}s)",
                flush=True,
            )
        results[sid] = parse_summary(log_path)
        (RERUN_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))
        (RERUN_DIR / "REPORT.md").write_text(build_report(results))
    print(f"[peaks_rerun] Done {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
