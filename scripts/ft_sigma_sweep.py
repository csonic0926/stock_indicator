"""Sweep ft bucket's TP sigma on triple_explore config, 1994-2026.

Production has ft sigma=0.0. Cal recalls sigma=0.75 didn't lose Calmar
in recent years and may help 1996 (the WR-collapse year). This script
runs three ft sigma variants while keeping fh/b30_35 at production
sigma=0.75 and gate at stop-only 75.

Variants: ft sigma in {0.0 (current), 0.5, 0.75, 1.0}.

Outputs side-by-side per-year + summary to
logs/ft_sigma_sweep/REPORT.md so Cal can verify 1996 improves and
recent years don't regress.
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
BASE_CONFIG = REPO / "data" / "multi_bucket_triple_explore.json"
OUT_DIR = REPO / "logs" / "ft_sigma_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VENV_PY = REPO / "venv" / "bin" / "python"

FT_SIGMAS = [0.0, 0.5, 0.75, 1.0]

SUMMARY_RE = re.compile(
    r"\[(Total|fish_head_production|fish_tail_production|fish_head_b30_35)\] "
    r"Trades: (\d+), Win rate: ([\d.]+)%, Mean profit %: ([\d.-]+)%, "
    r"Profit % Std Dev: ([\d.]+)%, Mean loss %: ([\d.-]+)%, "
    r"Loss % Std Dev: ([\d.]+)%, P/L: ([\d.]+), "
    r"Mean holding period: ([\d.]+) bars[^,]*, "
    r"Holding period Std Dev: ([\d.]+) bars, "
    r"Max concurrent positions: (\d+), Final balance: ([\d.]+), "
    r"CAGR: ([\d.-]+)%, Max drawdown: ([\d.]+)%"
)
YEAR_RE = re.compile(
    r"\[(Total|fish_head_production|fish_tail_production|fish_head_b30_35)\] "
    r"Year (\d{4}): ([\d.-]+)%"
)


def make_cfg(ft_sigma: float, cfg_path: Path) -> None:
    cfg = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    for bucket in cfg.get("buckets", []):
        if bucket.get("strategy_id") == "fish_tail_blow_off_top":
            bucket["sigma"] = ft_sigma
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def run_sim(cfg_path: Path, out_path: Path) -> int:
    cmd = (
        f'cd "{REPO}" && "{VENV_PY}" -m stock_indicator.manage <<< '
        f'"multi_bucket_simulation {cfg_path.relative_to(REPO)}\nexit" '
        f'> "{out_path}" 2>&1'
    )
    return subprocess.call(["/bin/bash", "-c", cmd])


def parse(out_path: Path) -> dict:
    text = out_path.read_text(encoding="utf-8", errors="replace")
    out: dict = {"summary": {}, "yearly": {}}
    for m in SUMMARY_RE.finditer(text):
        label = m.group(1)
        out["summary"][label] = {
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
    for m in YEAR_RE.finditer(text):
        label, year, ret_pct = m.group(1), int(m.group(2)), float(m.group(3)) / 100
        out["yearly"].setdefault(label, {})[year] = ret_pct
    return out


def fmt_pct(x: float, sign: bool = False) -> str:
    return f"{x*100:+.2f}%" if sign else f"{x*100:.2f}%"


def main() -> int:
    print(f"[ft_sigma_sweep] Start {time.strftime('%H:%M:%S')}", flush=True)
    print(f"[ft_sigma_sweep] Sigmas: {FT_SIGMAS}", flush=True)
    results: dict[float, dict] = {}
    for sigma in FT_SIGMAS:
        sigma_text = f"{sigma:.2f}"
        cfg_path = OUT_DIR / f"cfg_ftsigma_{sigma_text}.json"
        out_path = OUT_DIR / f"sim_ftsigma_{sigma_text}.log"
        make_cfg(sigma, cfg_path)
        t0 = time.time()
        print(
            f"[ft_sigma_sweep] ft_sigma={sigma_text} starting at "
            f"{time.strftime('%H:%M:%S')}",
            flush=True,
        )
        rc = run_sim(cfg_path, out_path)
        elapsed = time.time() - t0
        parsed = parse(out_path)
        results[sigma] = {"parsed": parsed, "rc": rc, "elapsed": elapsed}
        if "Total" in parsed["summary"]:
            t = parsed["summary"]["Total"]
            cal = t["cagr"] / t["mdd"] if t["mdd"] > 0 else 0
            print(
                f"  → ft_sigma={sigma_text}: Total Calmar={cal:.3f} "
                f"CAGR={t['cagr']:.2%} MDD={t['mdd']:.2%}",
                flush=True,
            )
            ft = parsed["summary"].get("fish_tail_production")
            if ft:
                ft_cal = ft["cagr"] / ft["mdd"] if ft["mdd"] > 0 else 0
                print(
                    f"    ft bucket: Calmar={ft_cal:.3f} CAGR={ft['cagr']:.2%} "
                    f"MDD={ft['mdd']:.2%}",
                    flush=True,
                )
        print(f"  elapsed: {elapsed/60:.1f}min", flush=True)

    # Build report
    lines = [
        "# ft sigma sweep — 1994-2026 (triple_explore base)",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Base: `{BASE_CONFIG.name}` (gate stop-only=75, fh sigma=0.75, b30_35 sigma=0.75)",
        f"Only ft bucket's sigma is varied.",
        "",
        "## Total bucket summary",
        "",
        "| ft_sigma | Trades | WR | CAGR | MDD | Calmar | Final |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sigma in FT_SIGMAS:
        t = results[sigma]["parsed"]["summary"].get("Total")
        if not t:
            lines.append(f"| {sigma:.2f} | — | | | | | failed |")
            continue
        cal = t["cagr"] / t["mdd"] if t["mdd"] > 0 else 0
        lines.append(
            f"| {sigma:.2f} | {t['trades']} | {t['wr']:.2%} | "
            f"{t['cagr']:.2%} | {t['mdd']:.2%} | **{cal:.3f}** | "
            f"${t['final_balance']:,.0f} |"
        )

    lines.append("")
    lines.append("## ft bucket isolated")
    lines.append("")
    lines.append("| ft_sigma | Trades | WR | CAGR | MDD | Calmar |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for sigma in FT_SIGMAS:
        t = results[sigma]["parsed"]["summary"].get("fish_tail_production")
        if not t:
            lines.append(f"| {sigma:.2f} | — | | | | |")
            continue
        cal = t["cagr"] / t["mdd"] if t["mdd"] > 0 else 0
        lines.append(
            f"| {sigma:.2f} | {t['trades']} | {t['wr']:.2%} | "
            f"{t['cagr']:.2%} | {t['mdd']:.2%} | {cal:.3f} |"
        )

    # Per-year ft for the focus years
    lines.append("")
    lines.append("## ft bucket per-year (focus on 1996 + recent calibration)")
    lines.append("")
    header = "| Year | " + " | ".join(f"σ={s:.2f}" for s in FT_SIGMAS) + " |"
    sep = "|---|" + "|".join("---:" for _ in FT_SIGMAS) + "|"
    lines.append(header)
    lines.append(sep)
    years = sorted(
        set().union(
            *[
                set(results[s]["parsed"]["yearly"].get("fish_tail_production", {}).keys())
                for s in FT_SIGMAS
            ]
        )
    )
    for y in years:
        row = [str(y)]
        for sigma in FT_SIGMAS:
            yv = (
                results[sigma]["parsed"]["yearly"]
                .get("fish_tail_production", {})
                .get(y)
            )
            row.append(fmt_pct(yv, sign=True) if yv is not None else "—")
        lines.append("| " + " | ".join(row) + " |")

    # Total per-year too
    lines.append("")
    lines.append("## Total per-year")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for y in years:
        row = [str(y)]
        for sigma in FT_SIGMAS:
            yv = (
                results[sigma]["parsed"]["yearly"]
                .get("Total", {})
                .get(y)
            )
            row.append(fmt_pct(yv, sign=True) if yv is not None else "—")
        lines.append("| " + " | ".join(row) + " |")

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "results.json").write_text(
        json.dumps({f"{k:.2f}": v for k, v in results.items()},
                   indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[ft_sigma_sweep] Done {time.strftime('%H:%M:%S')}", flush=True)
    print(f"  Report: {OUT_DIR / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
