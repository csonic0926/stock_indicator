"""Compare production config across the old (2010_safe) and new
(current_stock_universe) symbol universes.

Runs both variants through ``multi_bucket_simulation`` and writes a
side-by-side markdown report covering summary metrics + yearly returns.
Both runs use ``data_source=2010`` so the long backtest is reproducible.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENV_PY = REPO / "venv" / "bin" / "python"
OUT_DIR = REPO / "logs" / "universe_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "old_2010_safe": REPO / "data" / "multi_bucket_universe_compare_old.json",
    "new_current_universe": REPO / "data" / "multi_bucket_universe_compare_new.json",
}

SUMMARY_RE = re.compile(
    r"\[(Total|fish_head_production|fish_tail_explore|fish_head_b30_35)\] "
    r"Trades: (\d+), Win rate: ([\d.]+)%, Mean profit %: ([\d.-]+)%, "
    r"Profit % Std Dev: ([\d.]+)%, Mean loss %: ([\d.-]+)%, "
    r"Loss % Std Dev: ([\d.]+)%, P/L: ([\d.]+), "
    r"Mean holding period: ([\d.]+) bars[^,]*, "
    r"Holding period Std Dev: ([\d.]+) bars, "
    r"Max concurrent positions: (\d+), Final balance: ([\d.]+), "
    r"CAGR: ([\d.-]+)%, Max drawdown: ([\d.]+)%"
)
YEAR_RE = re.compile(
    r"\[(Total|fish_head_production|fish_tail_explore|fish_head_b30_35)\] "
    r"Year (\d{4}): ([\d.-]+)%"
)


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


def fmt_pct(x: float | None, sign: bool = False) -> str:
    if x is None:
        return "—"
    return f"{x*100:+.2f}%" if sign else f"{x*100:.2f}%"


def main() -> int:
    runs: dict[str, dict] = {}
    for name, cfg in VARIANTS.items():
        out_path = OUT_DIR / f"sim_{name}.log"
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] running {name} ...", flush=True)
        rc = run_sim(cfg, out_path)
        elapsed = time.time() - t0
        parsed = parse(out_path)
        t = parsed["summary"].get("Total")
        if t:
            cal = t["cagr"] / t["mdd"] if t["mdd"] > 0 else 0.0
            print(
                f"  → {name}: n={t['trades']} WR={t['wr']:.2%} "
                f"CAGR={t['cagr']:.2%} MDD={t['mdd']:.2%} "
                f"Calmar={cal:.3f} final=${t['final_balance']:,.0f} "
                f"(rc={rc}, {elapsed/60:.1f}min)",
                flush=True,
            )
        else:
            print(f"  ! {name}: no Total summary parsed (rc={rc})", flush=True)
        runs[name] = parsed

    labels = ["Total", "fish_head_production", "fish_tail_explore", "fish_head_b30_35"]
    lines: list[str] = [
        "# Universe Comparison: 2010_safe vs current_stock_universe",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Both runs use `data_source=2010`, `start_date=2010-01-01`, and the",
        "production bucket config (3 buckets + risk_score_gate stop=75,",
        "`disable_sl_trigger=true`, `fixed_sl=null`). Only `symbol_list`",
        "differs:",
        "",
        "- **old_2010_safe**: 5,004 symbols (`2010_safe`)",
        "- **new_current_universe**: 7,464 symbols (`current_stock_universe`)",
        "",
        "## Per-bucket + Total summary (full backtest)",
        "",
    ]
    for lbl in labels:
        lines.append(f"### {lbl}")
        lines.append("")
        lines.append("| Variant | Trades | WR | P/L | Hold | CAGR | MDD | Calmar | Final |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for v in VARIANTS:
            s = runs[v]["summary"].get(lbl)
            if not s:
                lines.append(f"| {v} | — | | | | | | | failed |")
                continue
            cal = s["cagr"] / s["mdd"] if s["mdd"] > 0 else 0.0
            lines.append(
                f"| {v} | {s['trades']} | {s['wr']:.2%} | {s['pl']:.3f} | "
                f"{s['hold']:.1f} | {s['cagr']:.2%} | {s['mdd']:.2%} | "
                f"**{cal:.3f}** | ${s['final_balance']:,.0f} |"
            )
        lines.append("")

    lines.append("## Year-by-year Total return")
    lines.append("")
    years = sorted(
        set().union(*[set(runs[v]["yearly"].get("Total", {}).keys()) for v in VARIANTS])
    )
    lines.append("| Year | old_2010_safe | new_current_universe | Δ (new - old) |")
    lines.append("|---|---:|---:|---:|")
    for y in years:
        old_v = runs["old_2010_safe"]["yearly"].get("Total", {}).get(y)
        new_v = runs["new_current_universe"]["yearly"].get("Total", {}).get(y)
        delta = (new_v - old_v) if (old_v is not None and new_v is not None) else None
        lines.append(
            f"| {y} | {fmt_pct(old_v, sign=True)} | "
            f"{fmt_pct(new_v, sign=True)} | {fmt_pct(delta, sign=True)} |"
        )

    report_path = OUT_DIR / "UNIVERSE_COMPARE_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    parsed_dump = OUT_DIR / "results.json"
    parsed_dump.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"[done] Report: {report_path}", flush=True)
    print(f"[done] Raw parsed: {parsed_dump}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
