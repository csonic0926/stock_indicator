"""Grid search over adaptive_tp_sl.sigma using multi_bucket_production_test.json as base.

Runs each sigma value through full multi_bucket_simulation, captures Total
and per-bucket summary lines, writes aggregate report to
logs/grid_sigma/REPORT.md.
"""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE_CONFIG = REPO / "data" / "multi_bucket_production_test.json"
GRID_DIR = REPO / "logs" / "grid_sigma"
GRID_DIR.mkdir(parents=True, exist_ok=True)

SIGMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

VENV_PY = REPO / "venv" / "bin" / "python"


def run_sim(cfg_path: Path, out_path: Path) -> int:
    cmd = f'cd "{REPO}" && "{VENV_PY}" -m stock_indicator.manage <<< "multi_bucket_simulation {cfg_path.relative_to(REPO)}\nexit" > "{out_path}" 2>&1'
    rc = subprocess.call(["/bin/bash", "-c", cmd])
    return rc


SUMMARY_RE = re.compile(
    r"\[(Total|fish_head_production|fish_tail_explore)\] "
    r"Trades: (\d+), Win rate: ([\d.]+)%, Mean profit %: ([\d.-]+)%, "
    r"Profit % Std Dev: ([\d.]+)%, Mean loss %: ([\d.-]+)%, "
    r"Loss % Std Dev: ([\d.]+)%, P/L: ([\d.]+), "
    r"Mean holding period: ([\d.]+) bars[^,]*, "
    r"Holding period Std Dev: ([\d.]+) bars, "
    r"Max concurrent positions: (\d+), Final balance: ([\d.]+), "
    r"CAGR: ([\d.-]+)%, Max drawdown: ([\d.]+)%"
)


def parse_summary(out_path: Path) -> dict:
    text = out_path.read_text(encoding="utf-8", errors="replace")
    out: dict = {}
    for m in SUMMARY_RE.finditer(text):
        label = m.group(1)
        out[label] = {
            "trades": int(m.group(2)),
            "wr": float(m.group(3)) / 100,
            "mp": float(m.group(4)) / 100,
            "mp_std": float(m.group(5)) / 100,
            "ml": float(m.group(6)) / 100,
            "ml_std": float(m.group(7)) / 100,
            "pl": float(m.group(8)),
            "hold": float(m.group(9)),
            "final_balance": float(m.group(12)),
            "cagr": float(m.group(13)) / 100,
            "mdd": float(m.group(14)) / 100,
        }
    # Yearly returns from Total bucket
    yearly = {}
    for m in re.finditer(r"\[Total\] Year (\d{4}): ([\d.-]+)%", text):
        yearly[int(m.group(1))] = float(m.group(2)) / 100
    out["__yearly_total"] = yearly
    # Yearly negatives count
    neg_years = sorted([y for y, r in yearly.items() if r < 0])
    out["__neg_years"] = neg_years
    return out


def main() -> int:
    print(f"[grid_sigma] Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"[grid_sigma] Base config: {BASE_CONFIG.name}", flush=True)
    print(f"[grid_sigma] Sigmas: {SIGMAS}", flush=True)

    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    results: dict[float, dict] = {}

    for idx, sigma in enumerate(SIGMAS, 1):
        sigma_str = f"{sigma:.2f}"
        cfg = copy.deepcopy(base)
        cfg["adaptive_tp_sl"]["sigma"] = sigma
        cfg_path = REPO / "data" / f"_grid_sigma_{sigma_str}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        out_path = GRID_DIR / f"sigma_{sigma_str}.log"

        t0 = time.time()
        print(f"\n[grid_sigma] [{idx}/{len(SIGMAS)}] sigma={sigma_str} starting at {time.strftime('%H:%M:%S')}", flush=True)
        rc = run_sim(cfg_path, out_path)
        elapsed = time.time() - t0
        print(f"[grid_sigma] [{idx}/{len(SIGMAS)}] sigma={sigma_str} done rc={rc} elapsed={elapsed/60:.1f}min", flush=True)

        parsed = parse_summary(out_path)
        if "Total" in parsed:
            t = parsed["Total"]
            print(f"  → Total: n={t['trades']}, WR={t['wr']:.4f}, P/L={t['pl']:.4f}, "
                  f"CAGR={t['cagr']:.4f}, MDD={t['mdd']:.4f}, "
                  f"Calmar={t['cagr']/t['mdd'] if t['mdd'] > 0 else 0:.3f}, "
                  f"neg_years={parsed['__neg_years']}", flush=True)
        else:
            print(f"  → no summary parsed (rc={rc}), see {out_path}", flush=True)

        results[sigma] = parsed

        # Cleanup temp config
        try:
            cfg_path.unlink()
        except OSError:
            pass

    # Report
    report_path = GRID_DIR / "REPORT.md"
    lines = ["# Sigma Grid Search Report\n"]
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Base config: `{BASE_CONFIG.name}`")
    lines.append(f"Sigmas tested: {SIGMAS}")
    lines.append("")
    lines.append("## Total summary\n")
    lines.append("| sigma | n | WR | MP | ML | **P/L** | CAGR | MDD | Calmar | hold | neg_years |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for sigma in SIGMAS:
        r = results.get(sigma, {})
        if "Total" not in r:
            lines.append(f"| {sigma:.2f} | — | — | — | — | — | — | — | — | — | failed |")
            continue
        t = r["Total"]
        calmar = t["cagr"] / t["mdd"] if t["mdd"] > 0 else 0
        marker = "**" if sigma == 0.5 else ""
        neg_years = r["__neg_years"]
        lines.append(
            f"| {marker}{sigma:.2f}{marker} | {t['trades']} | {t['wr']:.4f} | "
            f"{t['mp']:.4f} | {t['ml']:.4f} | **{t['pl']:.4f}** | "
            f"{t['cagr']:.4f} | {t['mdd']:.4f} | {calmar:.3f} | "
            f"{t['hold']:.2f} | {len(neg_years)} ({','.join(str(y) for y in neg_years)}) |"
        )
    lines.append("")
    lines.append("## Per-bucket\n")
    for bucket in ["fish_head_production", "fish_tail_explore"]:
        lines.append(f"### {bucket}\n")
        lines.append("| sigma | n | WR | MP | ML | P/L | CAGR | MDD | Calmar |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for sigma in SIGMAS:
            r = results.get(sigma, {})
            if bucket not in r:
                continue
            b = r[bucket]
            calmar = b["cagr"] / b["mdd"] if b["mdd"] > 0 else 0
            lines.append(
                f"| {sigma:.2f} | {b['trades']} | {b['wr']:.4f} | "
                f"{b['mp']:.4f} | {b['ml']:.4f} | **{b['pl']:.4f}** | "
                f"{b['cagr']:.4f} | {b['mdd']:.4f} | {calmar:.3f} |"
            )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[grid_sigma] Report: {report_path}", flush=True)

    # Save raw JSON too
    json_path = GRID_DIR / "results.json"
    json_path.write_text(json.dumps(
        {f"{s:.2f}": v for s, v in results.items()}, indent=2, default=str
    ), encoding="utf-8")
    print(f"[grid_sigma] Raw results: {json_path}", flush=True)
    print(f"[grid_sigma] Done at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
