"""Compare baseline (no risk-score gate) vs gated (stop 75 + reduce 50→margin 1.0).

Uses multi_bucket_triple_explore.json as the source config and runs two
variants:
  - baseline: risk_score_gate stripped → identical to pre-gate behavior
  - gated:    risk_score_gate kept (current default)

Outputs CSV / log artifacts to logs/risk_score_gate_compare/ and prints
side-by-side metrics (full-run + per-year totals).

The 2010_safe 2010-2026 historical CSV has max score 50 so:
  - stop branch fires 0 months (threshold 75)
  - reduce branch fires 11 months (threshold 50)
Expected effect: drag on 2010 (3 months) + 2020-03 + 2022-09 + 2023-03 +
2025-11 + 2026 Q2 — overall mild equity reduction in those windows.
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
OUT_DIR.mkdir(parents=True, exist_ok=True)
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


def make_cfg(variant: str, cfg_path: Path) -> None:
    cfg = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    if variant == "baseline":
        cfg.pop("risk_score_gate", None)
    elif variant == "gated":
        # leave risk_score_gate as configured
        pass
    else:
        raise ValueError(f"Unknown variant: {variant}")
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
    print(f"[gate_compare] Start {time.strftime('%H:%M:%S')}", flush=True)
    results: dict[str, dict] = {}
    for variant in ("baseline", "gated"):
        cfg_path = OUT_DIR / f"cfg_{variant}.json"
        out_path = OUT_DIR / f"sim_{variant}.log"
        make_cfg(variant, cfg_path)
        t0 = time.time()
        print(
            f"[gate_compare] {variant} starting at {time.strftime('%H:%M:%S')}",
            flush=True,
        )
        rc = run_sim(cfg_path, out_path)
        elapsed = time.time() - t0
        parsed = parse(out_path)
        results[variant] = {"parsed": parsed, "rc": rc, "elapsed": elapsed}
        if "Total" in parsed["summary"]:
            t = parsed["summary"]["Total"]
            cal = t["cagr"] / t["mdd"] if t["mdd"] > 0 else 0
            print(
                f"  → {variant}: n={t['trades']} WR={t['wr']:.2%} "
                f"CAGR={t['cagr']:.2%} MDD={t['mdd']:.2%} Calmar={cal:.3f} "
                f"final={t['final_balance']:.0f}",
                flush=True,
            )
        print(f"  elapsed: {elapsed/60:.1f}min", flush=True)

    # Report
    lines = [
        "# Risk-Score Gate Comparison",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Base: `{BASE_CONFIG.name}` (2010_safe, 2010-2026)",
        "",
        "## Full-run summary (Total bucket)",
        "",
        "| Variant | Trades | WR | CAGR | MDD | Calmar | Final Balance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in ("baseline", "gated"):
        t = results[variant]["parsed"]["summary"].get("Total")
        if not t:
            lines.append(f"| {variant} | — | | | | | failed |")
            continue
        cal = t["cagr"] / t["mdd"] if t["mdd"] > 0 else 0
        lines.append(
            f"| {variant} | {t['trades']} | {t['wr']:.2%} | "
            f"{t['cagr']:.2%} | {t['mdd']:.2%} | **{cal:.3f}** | "
            f"${t['final_balance']:,.0f} |"
        )
    # Delta row
    b = results["baseline"]["parsed"]["summary"].get("Total")
    g = results["gated"]["parsed"]["summary"].get("Total")
    if b and g:
        cal_b = b["cagr"] / b["mdd"] if b["mdd"] > 0 else 0
        cal_g = g["cagr"] / g["mdd"] if g["mdd"] > 0 else 0
        lines.append(
            f"| delta | {g['trades']-b['trades']:+d} | "
            f"{(g['wr']-b['wr'])*100:+.2f}pp | "
            f"{(g['cagr']-b['cagr'])*100:+.2f}pp | "
            f"{(g['mdd']-b['mdd'])*100:+.2f}pp | "
            f"{cal_g-cal_b:+.3f} | "
            f"${g['final_balance']-b['final_balance']:+,.0f} |"
        )

    lines.append("")
    lines.append("## Per-bucket detail")
    for bucket in (
        "fish_head_production", "fish_tail_production", "fish_head_b30_35"
    ):
        lines.append("")
        lines.append(f"### {bucket}")
        lines.append("")
        lines.append("| Variant | Trades | WR | CAGR | MDD | Calmar |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for variant in ("baseline", "gated"):
            t = results[variant]["parsed"]["summary"].get(bucket)
            if not t:
                lines.append(f"| {variant} | — | | | | |")
                continue
            cal = t["cagr"] / t["mdd"] if t["mdd"] > 0 else 0
            lines.append(
                f"| {variant} | {t['trades']} | {t['wr']:.2%} | "
                f"{t['cagr']:.2%} | {t['mdd']:.2%} | {cal:.3f} |"
            )

    # Year-by-year Total
    lines.append("")
    lines.append("## Year-by-year Total returns")
    lines.append("")
    years = sorted(
        set(results["baseline"]["parsed"]["yearly"].get("Total", {}).keys())
        | set(results["gated"]["parsed"]["yearly"].get("Total", {}).keys())
    )
    lines.append("| Year | Baseline | Gated | Δ |")
    lines.append("|---|---:|---:|---:|")
    for y in years:
        bp = results["baseline"]["parsed"]["yearly"].get("Total", {}).get(y)
        gp = results["gated"]["parsed"]["yearly"].get("Total", {}).get(y)
        if bp is None and gp is None:
            continue
        b_s = fmt_pct(bp, sign=True) if bp is not None else "—"
        g_s = fmt_pct(gp, sign=True) if gp is not None else "—"
        if bp is not None and gp is not None:
            d_s = f"{(gp-bp)*100:+.2f}pp"
        else:
            d_s = "—"
        lines.append(f"| {y} | {b_s} | {g_s} | {d_s} |")

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(f"[gate_compare] Done {time.strftime('%H:%M:%S')}", flush=True)
    print(f"  Report: {OUT_DIR / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
