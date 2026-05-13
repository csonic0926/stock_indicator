"""Per-bucket 2D sigma grid search.

Sweeps fish_head sigma × fish_tail sigma independently. Waits for the
1D grid (scripts/grid_sigma.py via logs/grid_sigma.run.log) to finish
before starting so the two runs don't compete for CPU.

Output:
- logs/grid_sigma_2d/sigma_fhX_ftY.log  (per-combo simulator output)
- logs/grid_sigma_2d/REPORT.md           (aggregate Markdown table)
- logs/grid_sigma_2d/results.json        (raw parsed metrics)
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
GRID_DIR = REPO / "logs" / "grid_sigma_2d"
GRID_DIR.mkdir(parents=True, exist_ok=True)

WAIT_LOG = REPO / "logs" / "grid_sigma.run.log"

FISH_HEAD_SIGMAS = [0.5, 0.75, 1.0]
FISH_TAIL_SIGMAS = [0.0, 0.25, 0.5, 0.75]

VENV_PY = REPO / "venv" / "bin" / "python"

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


def wait_for_1d_grid_to_finish() -> None:
    """Block until the 1D grid script writes its done sentinel.

    The 1D script ends with `[grid_sigma] Done at ...`. If the log file
    doesn't exist or the sentinel never appears (e.g., grid was cancelled),
    we still proceed after a 30-min watchdog so the 2D run isn't stuck.
    """
    print(f"[grid_sigma_2d] Waiting for 1D grid to finish via {WAIT_LOG.name}", flush=True)
    deadline = time.time() + 30 * 3600  # 30h watchdog
    while time.time() < deadline:
        if WAIT_LOG.exists():
            text = WAIT_LOG.read_text(encoding="utf-8", errors="replace")
            if "[grid_sigma] Done at" in text:
                print(f"[grid_sigma_2d] 1D grid completed; proceeding.", flush=True)
                return
        time.sleep(60)
    print(f"[grid_sigma_2d] WATCHDOG: 30h elapsed without 1D done sentinel. Starting anyway.", flush=True)


def run_sim(cfg_path: Path, out_path: Path) -> int:
    cmd = (
        f'cd "{REPO}" && "{VENV_PY}" -m stock_indicator.manage <<< '
        f'"multi_bucket_simulation {cfg_path.relative_to(REPO)}\nexit" '
        f'> "{out_path}" 2>&1'
    )
    return subprocess.call(["/bin/bash", "-c", cmd])


def parse_summary(out_path: Path) -> dict:
    text = out_path.read_text(encoding="utf-8", errors="replace")
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


def main() -> int:
    wait_for_1d_grid_to_finish()

    print(f"[grid_sigma_2d] Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"[grid_sigma_2d] fish_head sigmas: {FISH_HEAD_SIGMAS}", flush=True)
    print(f"[grid_sigma_2d] fish_tail sigmas: {FISH_TAIL_SIGMAS}", flush=True)
    print(f"[grid_sigma_2d] Total combos: {len(FISH_HEAD_SIGMAS) * len(FISH_TAIL_SIGMAS)}", flush=True)

    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    results: dict = {}

    combos = [(fh, ft) for fh in FISH_HEAD_SIGMAS for ft in FISH_TAIL_SIGMAS]

    for idx, (fh_sigma, ft_sigma) in enumerate(combos, 1):
        cfg = copy.deepcopy(base)
        # Set per-bucket sigma override
        for b in cfg["buckets"]:
            if b["label"] == "fish_head_production":
                b["sigma"] = fh_sigma
            elif b["label"] == "fish_tail_explore":
                b["sigma"] = ft_sigma
        # Parent sigma stays as fallback (irrelevant since both buckets override)

        tag = f"fh{fh_sigma:.2f}_ft{ft_sigma:.2f}"
        cfg_path = REPO / "data" / f"_grid_2d_{tag}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        out_path = GRID_DIR / f"sigma_{tag}.log"

        t0 = time.time()
        print(
            f"\n[grid_sigma_2d] [{idx}/{len(combos)}] "
            f"fh={fh_sigma:.2f} ft={ft_sigma:.2f} starting at {time.strftime('%H:%M:%S')}",
            flush=True,
        )
        rc = run_sim(cfg_path, out_path)
        elapsed = time.time() - t0
        print(
            f"[grid_sigma_2d] [{idx}/{len(combos)}] "
            f"fh={fh_sigma:.2f} ft={ft_sigma:.2f} done rc={rc} elapsed={elapsed/60:.1f}min",
            flush=True,
        )

        parsed = parse_summary(out_path)
        if "Total" in parsed:
            t = parsed["Total"]
            calmar = t["cagr"] / t["mdd"] if t["mdd"] > 0 else 0
            print(
                f"  → Total: n={t['trades']}, WR={t['wr']:.4f}, P/L={t['pl']:.4f}, "
                f"CAGR={t['cagr']:.4f}, MDD={t['mdd']:.4f}, "
                f"Calmar={calmar:.3f}, neg_years={parsed['__neg_years']}",
                flush=True,
            )
        else:
            print(f"  → no summary parsed (rc={rc}), see {out_path}", flush=True)

        results[tag] = {
            "fh_sigma": fh_sigma,
            "ft_sigma": ft_sigma,
            **parsed,
        }
        try:
            cfg_path.unlink()
        except OSError:
            pass

    # Build report
    report_path = GRID_DIR / "REPORT.md"
    lines = ["# Per-bucket Sigma 2D Grid Search\n"]
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Base config: `{BASE_CONFIG.name}`")
    lines.append(f"fish_head sigmas: {FISH_HEAD_SIGMAS}")
    lines.append(f"fish_tail sigmas: {FISH_TAIL_SIGMAS}")
    lines.append("")

    # Total table by (fh_sigma, ft_sigma)
    def cell(metric: str) -> list[str]:
        ll = [f"## Total {metric}\n"]
        ll.append("| fh\\\\ft | " + " | ".join(f"{ft:.2f}" for ft in FISH_TAIL_SIGMAS) + " |")
        ll.append("|---" + "|---" * len(FISH_TAIL_SIGMAS) + "|")
        for fh in FISH_HEAD_SIGMAS:
            row = [f"{fh:.2f}"]
            for ft in FISH_TAIL_SIGMAS:
                tag = f"fh{fh:.2f}_ft{ft:.2f}"
                r = results.get(tag, {})
                if "Total" not in r:
                    row.append("—")
                    continue
                t = r["Total"]
                if metric == "P/L":
                    row.append(f"{t['pl']:.3f}")
                elif metric == "CAGR":
                    row.append(f"{t['cagr']*100:.2f}%")
                elif metric == "MDD":
                    row.append(f"{t['mdd']*100:.2f}%")
                elif metric == "Calmar":
                    c = t["cagr"] / t["mdd"] if t["mdd"] > 0 else 0
                    row.append(f"{c:.3f}")
                elif metric == "WR":
                    row.append(f"{t['wr']*100:.2f}%")
                elif metric == "Trades":
                    row.append(str(t["trades"]))
                elif metric == "neg_years":
                    row.append(str(len(r.get("__neg_years", []))))
                else:
                    row.append("?")
            ll.append("| " + " | ".join(row) + " |")
        ll.append("")
        return ll

    for metric in ["P/L", "CAGR", "MDD", "Calmar", "WR", "Trades", "neg_years"]:
        lines.extend(cell(metric))

    # Best combinations
    lines.append("## Top combos by Calmar\n")
    lines.append("| rank | fh | ft | P/L | CAGR | MDD | Calmar | WR | n | neg_y |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    ranked = sorted(
        results.values(),
        key=lambda r: (r["Total"]["cagr"] / r["Total"]["mdd"]) if "Total" in r and r["Total"]["mdd"] > 0 else -1,
        reverse=True,
    )
    for rk, r in enumerate(ranked[:10], 1):
        if "Total" not in r:
            continue
        t = r["Total"]
        calmar = t["cagr"] / t["mdd"] if t["mdd"] > 0 else 0
        lines.append(
            f"| {rk} | {r['fh_sigma']:.2f} | {r['ft_sigma']:.2f} | "
            f"{t['pl']:.3f} | {t['cagr']*100:.2f}% | {t['mdd']*100:.2f}% | "
            f"**{calmar:.3f}** | {t['wr']*100:.2f}% | {t['trades']} | "
            f"{len(r.get('__neg_years', []))} |"
        )
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[grid_sigma_2d] Report: {report_path}", flush=True)

    json_path = GRID_DIR / "results.json"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"[grid_sigma_2d] Raw results: {json_path}", flush=True)
    print(f"[grid_sigma_2d] Done at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
