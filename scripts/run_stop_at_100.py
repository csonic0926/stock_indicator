"""Run a single sim variant with stop_threshold=100 and compare with
the existing baseline + stop_only results from logs/risk_score_gate_compare/.

Tests whether elevating the stop threshold so only score-100 months
(2008-04 to 2009-03, 12 months) trigger stop — instead of also stopping
the 75-score subprime warning window (2007-08 to 2008-03, 8 months) —
produces better/worse outcomes.
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
OUT_DIR = REPO / "logs" / "risk_score_gate_compare"
EXISTING_RESULTS = OUT_DIR / "results.json"
VENV_PY = REPO / "venv" / "bin" / "python"

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


def make_cfg(cfg_path: Path) -> None:
    cfg = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    gate = cfg.setdefault("risk_score_gate", {})
    gate["csv_path"] = "data/historical_risk_scores.csv"
    gate["stop_threshold"] = 100
    gate.pop("reduce_threshold", None)
    gate.pop("reduce_margin", None)
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


def main() -> int:
    print(f"[stop_at_100] Start {time.strftime('%H:%M:%S')}", flush=True)
    cfg_path = OUT_DIR / "cfg_stop_at_100.json"
    out_path = OUT_DIR / "sim_stop_at_100.log"
    make_cfg(cfg_path)
    t0 = time.time()
    print(
        f"[stop_at_100] starting sim at {time.strftime('%H:%M:%S')}",
        flush=True,
    )
    rc = run_sim(cfg_path, out_path)
    elapsed = time.time() - t0
    parsed = parse(out_path)
    if "Total" in parsed["summary"]:
        t = parsed["summary"]["Total"]
        cal = t["cagr"] / t["mdd"] if t["mdd"] > 0 else 0
        print(
            f"  → stop_at_100: n={t['trades']} WR={t['wr']:.2%} "
            f"CAGR={t['cagr']:.2%} MDD={t['mdd']:.2%} Calmar={cal:.3f} "
            f"final=${t['final_balance']:,.0f}",
            flush=True,
        )
    print(f"  elapsed: {elapsed/60:.1f}min", flush=True)

    # Compose comparison report.
    existing = json.loads(EXISTING_RESULTS.read_text(encoding="utf-8"))
    variants = ["baseline", "stop_only", "stop_at_100"]
    runs = {
        "baseline": existing["baseline"]["parsed"],
        "stop_only": existing["stop_only"]["parsed"],
        "stop_at_100": parsed,
    }

    def fmt_pct(x: float, sign: bool = False) -> str:
        return f"{x*100:+.2f}%" if sign else f"{x*100:.2f}%"

    lines = [
        "# Stop Threshold Comparison: 75 vs 100",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "stop_only uses stop_threshold=75 (catches the score=75 subprime",
        "warning window 2007-08 to 2008-03 + the score=100 GFC window",
        "2008-04 to 2009-03). stop_at_100 uses stop_threshold=100 (only",
        "the 12 GFC-peak months).",
        "",
        "## Full-run summary (Total)",
        "",
        "| Variant | Trades | WR | CAGR | MDD | Calmar | Final |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for v in variants:
        t = runs[v]["summary"].get("Total")
        if not t:
            lines.append(f"| {v} | — | | | | | failed |")
            continue
        cal = t["cagr"] / t["mdd"] if t["mdd"] > 0 else 0
        lines.append(
            f"| {v} | {t['trades']} | {t['wr']:.2%} | "
            f"{t['cagr']:.2%} | {t['mdd']:.2%} | **{cal:.3f}** | "
            f"${t['final_balance']:,.0f} |"
        )

    lines.append("")
    lines.append("## Year-by-year Total (2006-2010 window — the critical zone)")
    lines.append("")
    lines.append("| Year | baseline | stop_only | stop_at_100 |")
    lines.append("|---|---:|---:|---:|")
    for y in range(2006, 2011):
        cells = [str(y)]
        for v in variants:
            yv = runs[v]["yearly"].get("Total", {}).get(y)
            cells.append(fmt_pct(yv, sign=True) if yv is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Year-by-year Total (full series)")
    lines.append("")
    lines.append("| Year | baseline | stop_only | stop_at_100 |")
    lines.append("|---|---:|---:|---:|")
    years = sorted(
        set().union(
            *[set(runs[v]["yearly"].get("Total", {}).keys()) for v in variants]
        )
    )
    for y in years:
        cells = [str(y)]
        for v in variants:
            yv = runs[v]["yearly"].get("Total", {}).get(y)
            cells.append(fmt_pct(yv, sign=True) if yv is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")

    report_path = OUT_DIR / "STOP_THRESHOLD_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[stop_at_100] Report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
