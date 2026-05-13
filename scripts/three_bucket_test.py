"""3-bucket test: fh + ft + one mid-band peak at a time.

Tests whether any individual mid-band peak (0.30-0.35, 0.50-0.55,
0.75-0.80, 0.85-0.90) added as a priority-2 third bucket to the
fh+ft dual config produces net positive change vs dual baseline.

Hypothesis: 6-bucket test (2026-05-12) showed cannibalization — all 4
mid-bands together hurt Calmar (-64%). But individually, one of them
might be sufficiently regime-orthogonal to fh/ft to add diversification.

Each sim is sequential. ETA per sim ~60-90 min (3 buckets = slower than
dual but faster than 6). Total ~4-6 hr for 4 configs.

Outputs:
- logs/three_bucket/cfg_with_b<lo>_<hi>.json (per-test config)
- logs/three_bucket/sim_with_b<lo>_<hi>.log (per-test stdout)
- logs/three_bucket/REPORT.md (aggregate table)
- logs/three_bucket/results.json (parsed metrics)

Idempotent: skips sims with [Total] line already present.
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
BASE_CONFIG = REPO / "data" / "multi_bucket_six_bucket_test.json"
TEST_DIR = REPO / "logs" / "three_bucket"
TEST_DIR.mkdir(parents=True, exist_ok=True)
VENV_PY = REPO / "venv" / "bin" / "python"

MID_BANDS = [
    "fish_head_b30_35",
    "fish_head_b50_55",
    "fish_head_b75_80",
    "fish_head_b85_90",
]

SUMMARY_RE = re.compile(
    r"\[(Total|fish_[^\]]+)\] "
    r"Trades: (\d+), Win rate: ([\d.]+)%, Mean profit %: ([\d.-]+)%, "
    r"Profit % Std Dev: ([\d.]+)%, Mean loss %: ([\d.-]+)%, "
    r"Loss % Std Dev: ([\d.]+)%, P/L: ([\d.]+), "
    r"Mean holding period: ([\d.]+) bars[^,]*, "
    r"Holding period Std Dev: ([\d.]+) bars, "
    r"Max concurrent positions: (\d+), Final balance: ([\d.]+), "
    r"CAGR: ([\d.-]+)%, Max drawdown: ([\d.]+)%"
)


def make_cfg(mid_band_id: str, cfg_path: Path) -> None:
    """3-bucket config: fh + ft (priority 1) + given mid-band (priority 2)."""
    cfg = json.loads(BASE_CONFIG.read_text())
    # Drop all priority-2 buckets, keep only fh+ft from 6-bucket base
    cfg["buckets"] = [b for b in cfg["buckets"] if b.get("priority", 0) == 1]
    # Find the mid-band template from the 6-bucket config
    six_bucket = json.loads(BASE_CONFIG.read_text())
    mid_band_template = next(
        b for b in six_bucket["buckets"] if b["strategy_id"] == mid_band_id
    )
    cfg["buckets"].append(copy.deepcopy(mid_band_template))
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
            "max_concurrent": int(m.group(11)),
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
        "# 3-bucket test — fh + ft + one mid-band peak at a time",
        "",
        "Tests whether any individual mid-band peak added as priority-2 third",
        "bucket to fh+ft dual config produces net positive change vs dual",
        "baseline (Calmar 0.838, 2282 trades).",
        "",
        "## Total comparison",
        "",
        "| Config | Total Trades | WR | P/L | CAGR | MDD | Calmar | NegYrs |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for mb in MID_BANDS:
        r = results.get(mb, {})
        total = r.get("Total")
        if not total:
            lines.append(f"| fh+ft+{mb} | — | | | | | | |")
            continue
        cagr = total["cagr"]
        mdd = total["mdd"]
        calmar = cagr / mdd if mdd > 0 else float("nan")
        neg = r.get("__neg_years", [])
        neg_str = ",".join(str(y) for y in neg) if neg else "-"
        lines.append(
            f"| fh+ft+{mb} | {total['trades']} | "
            f"{total['wr']:.1%} | {total['pl']:.2f} | "
            f"{cagr:.2%} | {mdd:.2%} | **{calmar:.3f}** | {neg_str} |"
        )

    lines.append("")
    lines.append("## Per-bucket detail")
    lines.append("")
    for mb in MID_BANDS:
        r = results.get(mb, {})
        if not r.get("Total"):
            continue
        lines.append(f"### fh+ft+{mb}")
        lines.append("")
        lines.append("| Bucket | Trades | WR | P/L | CAGR | MDD | Calmar | Max conc |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for label in ("Total", "fish_head_production", "fish_tail_explore", mb):
            b = r.get(label)
            if not b:
                continue
            cagr = b["cagr"]
            mdd = b["mdd"]
            cal = cagr / mdd if mdd > 0 else float("nan")
            lines.append(
                f"| {label} | {b['trades']} | {b['wr']:.1%} | {b['pl']:.2f} | "
                f"{cagr:.2%} | {mdd:.2%} | {cal:.3f} | {b['max_concurrent']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    print(
        f"[three_bucket] Start at {time.strftime('%Y-%m-%d %H:%M:%S')}",
        flush=True,
    )
    results: dict = {}
    for index, mid_band_id in enumerate(MID_BANDS, start=1):
        cfg_path = TEST_DIR / f"cfg_with_{mid_band_id}.json"
        log_path = TEST_DIR / f"sim_with_{mid_band_id}.log"
        if is_done(log_path):
            print(
                f"[three_bucket] [{index}/{len(MID_BANDS)}] {mid_band_id} "
                f"skip (done)",
                flush=True,
            )
        else:
            make_cfg(mid_band_id, cfg_path)
            t0 = time.time()
            print(
                f"[three_bucket] [{index}/{len(MID_BANDS)}] {mid_band_id} start",
                flush=True,
            )
            rc = run_sim(cfg_path, log_path)
            elapsed = time.time() - t0
            print(
                f"[three_bucket] [{index}/{len(MID_BANDS)}] {mid_band_id} "
                f"rc={rc} ({elapsed:.0f}s)",
                flush=True,
            )
        results[mid_band_id] = parse_summary(log_path)
        (TEST_DIR / "results.json").write_text(
            json.dumps(results, indent=2, default=str)
        )
        (TEST_DIR / "REPORT.md").write_text(build_report(results))
    print(
        f"[three_bucket] Done at {time.strftime('%Y-%m-%d %H:%M:%S')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
