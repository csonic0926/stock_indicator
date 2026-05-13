"""TP sigma sweep on b85_90 paired with fh.

Methodology: anchor fh at sigma=0.75 (known production setting),
sweep b85_90 sigma across {0.0, 0.25, 0.5, 0.75, 1.0, 1.25}.
Run fh + b85_90 (no ft, no other buckets) — pair structure mirrors
how ft's optimal sigma=0.0 was originally discovered.

Find b85_90's best sigma in fh-pair context. That sigma value then
goes into multi-bucket (fh + ft + b85_90 tuned) config later.

Per-year totals are critical for crash-stability evaluation.
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
SWEEP_DIR = REPO / "logs" / "sigma_sweep_b85_90"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)
VENV_PY = REPO / "venv" / "bin" / "python"

SIGMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]

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


def make_cfg(sigma: float, cfg_path: Path) -> None:
    """fh + b85_90 pair, fh fixed sigma=0.75, b85_90 sigma=sweep value."""
    cfg = json.loads(BASE_CONFIG.read_text())
    fh = copy.deepcopy(cfg["buckets"][0])  # fish_head_production with sigma=0.75
    b85_90 = copy.deepcopy(fh)
    b85_90["strategy_id"] = "fish_head_b85_90"
    b85_90["label"] = "fish_head_b85_90"
    b85_90["sigma"] = sigma
    cfg["buckets"] = [fh, b85_90]
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
        "# fh + b85_90 sigma sweep (fh fixed 0.75, b85_90 swept)",
        "",
        "## Total comparison",
        "",
        "| b85_90 sigma | Trades | WR | P/L | CAGR | MDD | Calmar | NegYrs |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for sigma in SIGMAS:
        r = results.get(sigma, {})
        total = r.get("Total")
        if not total:
            lines.append(f"| {sigma:.2f} | — | | | | | | |")
            continue
        cagr = total["cagr"]
        mdd = total["mdd"]
        cal = cagr / mdd if mdd > 0 else float("nan")
        neg = r.get("__neg_years", [])
        neg_str = ",".join(str(y) for y in neg) if neg else "-"
        lines.append(
            f"| {sigma:.2f} | {total['trades']} | {total['wr']:.1%} | "
            f"{total['pl']:.2f} | {cagr:.2%} | {mdd:.2%} | "
            f"**{cal:.3f}** | {neg_str} |"
        )

    # Per-year matrix
    lines.append("")
    lines.append("## Per-year Total %")
    lines.append("")
    years = sorted(set().union(
        *[set(r.get("__yearly_total", {}).keys()) for r in results.values()]
    ))
    header = "| Year | " + " | ".join(f"σ={s:.2f}" for s in SIGMAS) + " |"
    sep = "|---|" + "|".join("---:" for _ in SIGMAS) + "|"
    lines.append(header)
    lines.append(sep)
    for year in years:
        row = [f"| {year} "]
        for sigma in SIGMAS:
            r = results.get(sigma, {})
            yearly = r.get("__yearly_total", {})
            val = yearly.get(year)
            if val is None:
                row.append("| — ")
            else:
                marker = "**" if val < 0 else ""
                row.append(f"| {marker}{val*100:+.2f}%{marker} ")
        row.append("|")
        lines.append("".join(row))
    return "\n".join(lines) + "\n"


def main() -> int:
    print(f"[sigma_sweep_b85_90] Start {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    results: dict = {}
    for idx, sigma in enumerate(SIGMAS, 1):
        cfg_path = SWEEP_DIR / f"cfg_sigma{sigma:.2f}.json"
        log_path = SWEEP_DIR / f"sim_sigma{sigma:.2f}.log"
        if is_done(log_path):
            print(f"[sigma_sweep_b85_90] [{idx}/{len(SIGMAS)}] σ={sigma:.2f} skip (done)", flush=True)
        else:
            make_cfg(sigma, cfg_path)
            t0 = time.time()
            print(f"[sigma_sweep_b85_90] [{idx}/{len(SIGMAS)}] σ={sigma:.2f} start", flush=True)
            rc = run_sim(cfg_path, log_path)
            elapsed = time.time() - t0
            print(
                f"[sigma_sweep_b85_90] [{idx}/{len(SIGMAS)}] σ={sigma:.2f} "
                f"rc={rc} ({elapsed:.0f}s)",
                flush=True,
            )
        results[sigma] = parse_summary(log_path)
        (SWEEP_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))
        (SWEEP_DIR / "REPORT.md").write_text(build_report(results))
    print(f"[sigma_sweep_b85_90] Done {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
