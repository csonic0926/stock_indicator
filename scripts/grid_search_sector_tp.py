"""Grid search sector rolling TP/SL: sigma and min_rr."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = str(REPO / "venv/bin/python")
SRC = str(REPO / "src")

BASE_CONFIG = {
    "max_position_count": 6,
    "starting_cash": 60000,
    "start_date": "2014-01-01",
    "data_source": "2014",
    "withdraw": 0,
    "min_hold": 5,
    "margin": 1.0,
    "confirmation_mode": None,
    "confirmation_sma_angle_min": -99,
    "confirmation_sma_angle_max": 99,
    "show_trade_details": False,
    "buckets": [
        {
            "label": "buy3_sector",
            "strategy_id": "buy3",
            "dollar_volume_filter": "dollar_volume>0.05%,Top200,Pick5",
            "stop_loss": 1.0,
            "take_profit": 0.0,
            "priority": 1,
            "max_positions": 6,
            "exit_alpha_factor": 3,
        }
    ],
}

SIGMAS = [0.3, 0.5, 0.7, 1.0]
MIN_RRS = [1.5, 2.0, 2.5, 3.0]


def run_one(sigma, min_rr):
    config = dict(BASE_CONFIG)
    config["adaptive_tp_sl"] = {
        "window": 20,
        "sigma": 0.5,
        "fixed_sl": 0.03,
        "override_min_hold_tp_only": True,
        "min_hold_tp": 1,
        "delayed_rolling_update": True,
        "sector_rolling": True,
        "sector_sl_sigma": sigma,
        "sector_min_rr": min_rr,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=str(REPO / "data")
    ) as f:
        json.dump(config, f)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            [PYTHON, "-m", "stock_indicator.manage", "multi_bucket_simulation", tmp_path],
            capture_output=True, text=True, cwd=SRC, timeout=1800,
        )
    except subprocess.TimeoutExpired:
        Path(tmp_path).unlink(missing_ok=True)
        return {"error": "TIMEOUT"}
    Path(tmp_path).unlink(missing_ok=True)

    out = result.stdout
    # Parse [Total] line
    for line in out.split("\n"):
        if line.startswith("[Total] Trades:"):
            parts = line.split(",")
            metrics = {}
            for p in parts:
                p = p.strip()
                if "CAGR" in p:
                    metrics["cagr"] = p.split(":")[1].strip().rstrip("%")
                elif "Max drawdown" in p:
                    metrics["mdd"] = p.split(":")[1].strip().rstrip("%")
                elif "P/L" in p and "Profit" not in p:
                    metrics["pl"] = p.split(":")[1].strip()
                elif "Win rate" in p:
                    metrics["wr"] = p.split(":")[1].strip().rstrip("%")
                elif "Mean loss" in p:
                    metrics["ml"] = p.split(":")[1].strip().rstrip("%")
            return metrics
    return {"error": out[:200]}


print(f"{'sigma':>6} {'min_rr':>7} {'CAGR%':>7} {'MDD%':>7} {'Calmar':>7} {'P/L':>6} {'WR%':>6} {'ML%':>7}")
print("-" * 60)

for sigma in SIGMAS:
    for min_rr in MIN_RRS:
        m = run_one(sigma, min_rr)
        if "error" in m:
            print(f"{sigma:>6} {min_rr:>7} ERROR: {m['error'][:40]}")
            continue
        cagr = float(m["cagr"])
        mdd = float(m["mdd"])
        calmar = cagr / mdd if mdd > 0 else 0
        print(
            f"{sigma:>6.1f} {min_rr:>7.1f} {cagr:>7.2f} {mdd:>7.2f} {calmar:>7.2f} "
            f"{m['pl']:>6} {m['wr']:>6} {m['ml']:>7}"
        )

print()
print("Baseline (production uniform): CAGR=20.57% MDD=11.01% Calmar=1.87 P/L=2.03")
