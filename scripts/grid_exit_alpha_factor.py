"""Grid search over fh+b30_35 exit_alpha_factor on multi_bucket_triple_explore.

Focus window: 2005-2011 (Cal: 2008 stress study with lead-in + recovery).
Base config is triple_explore.json (already has 2005-01-01 start_date).
ft bucket's exit_alpha_factor stays at its configured value — only fish_head
family (fh and b30_35) is varied since they share the cross-down exit
mechanism we're investigating.

Per-year totals + a derived "2005-2011 sub-window" rollup (cumulative
return, count of negative years, approx max cumulative drawdown computed
from yearly returns) feed the report. Full-run Calmar (2005-2026) is also
shown as a sanity reference.
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
GRID_DIR = REPO / "logs" / "grid_exit_alpha_factor"
GRID_DIR.mkdir(parents=True, exist_ok=True)
VENV_PY = REPO / "venv" / "bin" / "python"

# Round 2: 2010_safe clean data, focus 1 vs 3 only (the two viable alphas
# from the 1994_clean run: alpha=0.25 was best, alpha=0.75 production was
# worst — verify on cleaner data).
FACTORS = [1, 3]

# Round 2 focus = full 2010-2026 range (2010_safe data starts at 2010).
FOCUS_YEARS = list(range(2010, 2027))

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


def make_cfg(factor: int, cfg_path: Path) -> None:
    cfg = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    for bucket in cfg["buckets"]:
        if bucket.get("strategy_id") in {
            "fish_head_vacuum_turn", "fish_head_b30_35"
        }:
            bucket["exit_alpha_factor"] = factor
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
            "cagr": float(m.group(13)) / 100,
            "mdd": float(m.group(14)) / 100,
        }
    for m in YEAR_RE.finditer(text):
        label, year, ret_pct = m.group(1), int(m.group(2)), float(m.group(3)) / 100
        out["yearly"].setdefault(label, {})[year] = ret_pct
    return out


def focus_metrics(yearly_total: dict[int, float]) -> dict:
    """2005-2011 sub-window rollup from yearly returns.

    cumulative_return = product of (1 + r_i) over focus years.
    max_dd_yoy_approx = largest peak-to-trough drop walking the
    cumulative curve year-end to year-end (this is coarser than a true
    daily MDD, but captures the regime shape).
    """
    rets = [yearly_total.get(y, 0.0) for y in FOCUS_YEARS]
    cum_curve = []
    val = 1.0
    for r in rets:
        val *= (1.0 + r)
        cum_curve.append(val)
    if not cum_curve:
        return {"cum_ret": 0.0, "max_dd_yoy": 0.0, "neg_years": []}
    peak = cum_curve[0]
    max_dd = 0.0
    for v in cum_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    cum_ret = cum_curve[-1] - 1.0
    neg = [y for y, r in zip(FOCUS_YEARS, rets) if r < 0]
    n_yrs = len(FOCUS_YEARS)
    cagr = ((1 + cum_ret) ** (1.0 / n_yrs) - 1) if cum_ret > -1 else -1.0
    calmar = cagr / max_dd if max_dd > 0 else float("nan")
    return {
        "cum_ret": cum_ret,
        "cagr": cagr,
        "max_dd_yoy": max_dd,
        "calmar": calmar,
        "neg_years": neg,
    }


def main() -> int:
    print(f"[grid_exit_alpha_factor] Start {time.strftime('%H:%M:%S')}", flush=True)
    print(f"[grid_exit_alpha_factor] Factors: {FACTORS}", flush=True)
    results: dict[int, dict] = {}

    for idx, factor in enumerate(FACTORS, 1):
        cfg_path = GRID_DIR / f"cfg_factor{factor}.json"
        out_path = GRID_DIR / f"sim_factor{factor}.log"
        make_cfg(factor, cfg_path)
        t0 = time.time()
        print(
            f"[{idx}/{len(FACTORS)}] factor={factor} starting at "
            f"{time.strftime('%H:%M:%S')}",
            flush=True,
        )
        rc = run_sim(cfg_path, out_path)
        elapsed = time.time() - t0
        parsed = parse(out_path)
        focus = focus_metrics(parsed["yearly"].get("Total", {}))
        results[factor] = {"parsed": parsed, "focus": focus, "rc": rc, "elapsed": elapsed}
        if "Total" in parsed["summary"]:
            t = parsed["summary"]["Total"]
            print(
                f"  → full(2005-2026): n={t['trades']} CAGR={t['cagr']:.2%} "
                f"MDD={t['mdd']:.2%} Calmar={t['cagr']/t['mdd'] if t['mdd']>0 else 0:.3f}",
                flush=True,
            )
            print(
                f"  → focus(2005-2011): cum_ret={focus['cum_ret']:+.2%} "
                f"CAGR={focus['cagr']:+.2%} maxDD_yoy={focus['max_dd_yoy']:.2%} "
                f"Calmar={focus['calmar']:.3f} negYrs={focus['neg_years']}",
                flush=True,
            )
        else:
            print(f"  → no summary parsed (rc={rc})", flush=True)
        print(f"  elapsed: {elapsed/60:.1f}min", flush=True)

    # Build report
    lines = [
        f"# exit_alpha_factor grid (fh + b30_35) — focus 2005-2011",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Base: `{BASE_CONFIG.name}` (start 2005-01-01, 1994_clean data)",
        f"Varied buckets: fish_head_production, fish_head_b30_35",
        f"ft bucket unchanged (different exit mechanism)",
        "",
        "## Focus window (2005-2011) — rollup from yearly Total returns",
        "",
        "| factor | cum_ret | CAGR | maxDD(yoy) | Calmar | NegYrs |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for f in FACTORS:
        r = results.get(f, {})
        focus = r.get("focus", {})
        if not focus:
            lines.append(f"| {f} | — | — | — | — | failed |")
            continue
        neg = focus["neg_years"]
        neg_str = ",".join(str(y) for y in neg) if neg else "-"
        lines.append(
            f"| {f} | {focus['cum_ret']:+.2%} | {focus['cagr']:+.2%} | "
            f"{focus['max_dd_yoy']:.2%} | **{focus['calmar']:.3f}** | {neg_str} |"
        )

    lines.append("")
    lines.append("## Full run (2005-2026) — sanity reference")
    lines.append("")
    lines.append("| factor | Trades | WR | CAGR | MDD | Calmar | mean_hold |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for f in FACTORS:
        r = results.get(f, {})
        t = r.get("parsed", {}).get("summary", {}).get("Total")
        if not t:
            lines.append(f"| {f} | — | | | | | |")
            continue
        cal = t["cagr"] / t["mdd"] if t["mdd"] > 0 else float("nan")
        lines.append(
            f"| {f} | {t['trades']} | {t['wr']:.2%} | {t['cagr']:.2%} | "
            f"{t['mdd']:.2%} | {cal:.3f} | {t['hold']:.2f} |"
        )

    lines.append("")
    lines.append("## Per-year Total (focus window detail)")
    lines.append("")
    header = "| Year | " + " | ".join(f"f={f}" for f in FACTORS) + " |"
    sep = "|---|" + "|".join("---:" for _ in FACTORS) + "|"
    lines.append(header)
    lines.append(sep)
    for year in FOCUS_YEARS:
        row = [f"| {year} "]
        for f in FACTORS:
            yt = results.get(f, {}).get("parsed", {}).get("yearly", {}).get("Total", {})
            ret = yt.get(year)
            if ret is None:
                row.append("| — ")
            else:
                marker = "**" if ret < 0 else ""
                row.append(f"| {marker}{ret*100:+.2f}%{marker} ")
        row.append("|")
        lines.append("".join(row))

    # Per-bucket fh detail for 2008
    lines.append("")
    lines.append("## fh bucket — yearly return in focus window")
    lines.append("")
    lines.append(header.replace("| Year", "| Year"))
    lines.append(sep)
    for year in FOCUS_YEARS:
        row = [f"| {year} "]
        for f in FACTORS:
            fh_y = (
                results.get(f, {})
                .get("parsed", {})
                .get("yearly", {})
                .get("fish_head_production", {})
            )
            ret = fh_y.get(year)
            if ret is None:
                row.append("| — ")
            else:
                marker = "**" if ret < 0 else ""
                row.append(f"| {marker}{ret*100:+.2f}%{marker} ")
        row.append("|")
        lines.append("".join(row))

    (GRID_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (GRID_DIR / "results.json").write_text(
        json.dumps(
            {str(k): v for k, v in results.items()}, indent=2, default=str
        ),
        encoding="utf-8",
    )
    print(f"[grid_exit_alpha_factor] Done {time.strftime('%H:%M:%S')}", flush=True)
    print(f"  Report: {GRID_DIR / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
