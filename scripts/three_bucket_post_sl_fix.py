"""Re-run 3-bucket (fh+ft+one_peak) tests post disable_sl_trigger fix.

Tests complementarity of each mid-band peak with fh+ft. The key question
is not the peak's standalone magnitude (covered in peaks_rerun_post_sl_fix),
but whether the peak fires in moments fh+ft don't fully fill the
portfolio — i.e. whether it ADDS to dual baseline or just CANNIBALIZES.

Skipping b75_80 (post-fix Calmar 0.090 with MDD 54.74% — trap band).

Each sim is fh+ft (priority 1, max=6) + one mid-band (priority 2, max=6,
share-slot, NO fill_remaining yet). Compare Total Calmar to fresh dual
baseline (= production_test post-fix = Calmar 0.945).
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
SIX_BUCKET = REPO / "data" / "multi_bucket_six_bucket_test.json"
TEST_DIR = REPO / "logs" / "three_bucket_postfix"
TEST_DIR.mkdir(parents=True, exist_ok=True)
VENV_PY = REPO / "venv" / "bin" / "python"

MID_BANDS = [
    "fish_head_b30_35",
    "fish_head_b50_55",
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
    """3-bucket config: fh + ft (from production_test, SL-fixed) + mid-band template (from six_bucket)."""
    cfg = json.loads(BASE_CONFIG.read_text())
    # fh + ft already in BASE_CONFIG (production_test), with disable_sl_trigger=true
    # Pull mid-band template from six_bucket file
    six = json.loads(SIX_BUCKET.read_text())
    mid_band_template = next(
        b for b in six["buckets"] if b["strategy_id"] == mid_band_id
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
        "# 3-bucket test post disable_sl_trigger fix",
        "",
        "fh+ft (priority 1) + one mid-band (priority 2, share-slot, NO fill_remaining).",
        "Tests whether each mid-band ADDS to fresh dual baseline 0.945 or cannibalizes.",
        "",
        "## Total comparison vs baseline",
        "",
        "| Config | Total Trades | WR | P/L | CAGR | MDD | Calmar | NegYrs | Δ vs 0.945 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
        "| fresh dual baseline | 2282 | 60.04% | 1.08 | 26.08% | 27.61% | **0.945** | 2018 | — |",
    ]
    for mb in MID_BANDS:
        r = results.get(mb, {})
        total = r.get("Total")
        if not total:
            lines.append(f"| fh+ft+{mb} | — | | | | | | | |")
            continue
        cagr = total["cagr"]
        mdd = total["mdd"]
        calmar = cagr / mdd if mdd > 0 else float("nan")
        neg = r.get("__neg_years", [])
        neg_str = ",".join(str(y) for y in neg) if neg else "-"
        delta_pct = (calmar / 0.945 - 1) * 100
        delta_str = f"+{delta_pct:.1f}%" if delta_pct >= 0 else f"{delta_pct:.1f}%"
        lines.append(
            f"| fh+ft+{mb} | {total['trades']} | "
            f"{total['wr']:.1%} | {total['pl']:.2f} | "
            f"{cagr:.2%} | {mdd:.2%} | **{calmar:.3f}** | {neg_str} | {delta_str} |"
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
    print(f"[three_bucket_postfix] Start {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    results: dict = {}
    for idx, mb in enumerate(MID_BANDS, 1):
        cfg_path = TEST_DIR / f"cfg_with_{mb}.json"
        log_path = TEST_DIR / f"sim_with_{mb}.log"
        if is_done(log_path):
            print(f"[three_bucket_postfix] [{idx}/{len(MID_BANDS)}] {mb} skip (done)", flush=True)
        else:
            make_cfg(mb, cfg_path)
            t0 = time.time()
            print(f"[three_bucket_postfix] [{idx}/{len(MID_BANDS)}] {mb} start", flush=True)
            rc = run_sim(cfg_path, log_path)
            elapsed = time.time() - t0
            print(
                f"[three_bucket_postfix] [{idx}/{len(MID_BANDS)}] {mb} "
                f"rc={rc} ({elapsed:.0f}s)",
                flush=True,
            )
        results[mb] = parse_summary(log_path)
        (TEST_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))
        (TEST_DIR / "REPORT.md").write_text(build_report(results))
    print(f"[three_bucket_postfix] Done {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
