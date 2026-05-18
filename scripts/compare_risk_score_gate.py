"""Compare risk-score gate variants for anti-overfit validation.

Runs four config variants over the same multi-bucket base:

  baseline      no `risk_score_gate` at all (control)
  stop_only     gate present, reduce_threshold high (>= 100) so only
                stop (score >= 75) fires; isolates the 2007-2009 GFC
                window's contribution to total Calmar lift
  fh_b30_only   full gate, but ft bucket's `gate_enabled` flipped to
                false; tests whether ft's near-zero gate sensitivity
                from the per-bucket detail is real or noise
  full          gate as configured (stop 75 + reduce 50 -> margin 1.0)
                applied to every bucket

Outputs side-by-side summary + per-year totals to
logs/risk_score_gate_compare/REPORT.md.
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

VARIANTS = ("baseline", "stop_only", "fh_b30_only", "full")

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
    elif variant == "stop_only":
        # Keep gate present (stop_threshold default) but push reduce
        # threshold above max possible score so reduce never fires.
        gate = cfg.setdefault("risk_score_gate", {})
        gate["reduce_threshold"] = 200
    elif variant == "fh_b30_only":
        # Flip gate_enabled=false on ft bucket only. fh + b30_35 keep
        # gating (default gate_enabled=true).
        for bucket in cfg.get("buckets", []):
            if bucket.get("strategy_id") == "fish_tail_blow_off_top":
                bucket["gate_enabled"] = False
    elif variant == "full":
        # Use config as-is.
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
    print(f"[gate_compare] Variants: {VARIANTS}", flush=True)
    results: dict[str, dict] = {}
    for variant in VARIANTS:
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
                f"final=${t['final_balance']:,.0f}",
                flush=True,
            )
        print(f"  elapsed: {elapsed/60:.1f}min", flush=True)

    # Report
    lines = [
        "# Risk-Score Gate Variant Comparison",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Base config: `{BASE_CONFIG.name}`",
        "",
        "Variants:",
        "  - **baseline**: no gate (control)",
        "  - **stop_only**: gate present, reduce_threshold raised so only",
        "    stop (score >= 75) fires",
        "  - **fh_b30_only**: full gate but ft bucket's gate_enabled=false",
        "  - **full**: gate applied to all buckets (default)",
        "",
        "## Full-run summary (Total bucket)",
        "",
        "| Variant | Trades | WR | CAGR | MDD | Calmar | Final |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
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

    # Delta vs baseline
    base_total = results["baseline"]["parsed"]["summary"].get("Total")
    if base_total:
        lines.append("")
        lines.append("## Delta vs baseline")
        lines.append("")
        lines.append("| Variant | ΔTrades | ΔCAGR | ΔMDD | ΔCalmar | ΔFinal |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        cal_b = base_total["cagr"] / base_total["mdd"] if base_total["mdd"] > 0 else 0
        for variant in VARIANTS:
            if variant == "baseline":
                continue
            g = results[variant]["parsed"]["summary"].get("Total")
            if not g:
                continue
            cal_g = g["cagr"] / g["mdd"] if g["mdd"] > 0 else 0
            lines.append(
                f"| {variant} | {g['trades']-base_total['trades']:+d} | "
                f"{(g['cagr']-base_total['cagr'])*100:+.2f}pp | "
                f"{(g['mdd']-base_total['mdd'])*100:+.2f}pp | "
                f"{cal_g-cal_b:+.3f} | "
                f"${g['final_balance']-base_total['final_balance']:+,.0f} |"
            )

    # Per-bucket detail
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
        for variant in VARIANTS:
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
        set().union(
            *[
                set(results[v]["parsed"]["yearly"].get("Total", {}).keys())
                for v in VARIANTS
            ]
        )
    )
    header_cells = ["Year"] + list(VARIANTS)
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|---|" + "|".join("---:" for _ in VARIANTS) + "|")
    for y in years:
        cells = [str(y)]
        for v in VARIANTS:
            yv = results[v]["parsed"]["yearly"].get("Total", {}).get(y)
            cells.append(fmt_pct(yv, sign=True) if yv is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(f"[gate_compare] Done {time.strftime('%H:%M:%S')}", flush=True)
    print(f"  Report: {OUT_DIR / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
