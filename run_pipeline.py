from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("data/processed")
STATUS_FILE = PROCESSED_DIR / "pipeline_status.json"
PRICE_FILE = PROCESSED_DIR / "stock_price.csv"

PIPELINE_STEPS = [
    ("collect", "collect_and_preprocess_stocks.py"),
    ("rebuild_features", "rebuild_feature_data.py"),
    ("detect_anomalies", "detect_anomalies.py"),
    ("generate_labels", "generate_risk_labels.py"),
    ("train_model", "train_xgboost_model.py"),
    ("predict_risk", "predict_risk.py"),
]


def latest_price_date() -> date | None:
    if not PRICE_FILE.exists():
        return None

    prices = pd.read_csv(PRICE_FILE, usecols=["date"], encoding="utf-8-sig")
    if prices.empty:
        return None

    return pd.to_datetime(prices["date"]).dt.date.max()


def data_freshness(max_stale_days: int) -> dict[str, object]:
    latest_date = latest_price_date()
    if latest_date is None:
        return {
            "latest_price_date": None,
            "age_days": None,
            "max_stale_days": max_stale_days,
            "is_fresh": False,
        }

    age_days = (date.today() - latest_date).days
    return {
        "latest_price_date": str(latest_date),
        "age_days": int(age_days),
        "max_stale_days": int(max_stale_days),
        "is_fresh": bool(age_days <= max_stale_days),
    }


def selected_steps(args: argparse.Namespace) -> list[tuple[str, str]]:
    steps = PIPELINE_STEPS
    if args.skip_collect:
        steps = [step for step in steps if step[0] != "collect"]
    if args.skip_train:
        steps = [step for step in steps if step[0] != "train_model"]
    return steps


def step_args(name: str, args: argparse.Namespace) -> list[str]:
    if name == "collect":
        collect_args = ["--universe", args.universe, "--max-stocks", str(args.max_stocks)]
        if args.prefer_cache:
            collect_args.append("--prefer-cache")
        collect_args.extend(["--download-timeout", str(args.download_timeout)])
        return collect_args
    return []


def run_step(name: str, script: str, args: argparse.Namespace) -> dict[str, object]:
    command = [sys.executable, script, *step_args(name, args)]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed_seconds = time.perf_counter() - started

    status = {
        "name": name,
        "script": script,
        "command": command,
        "returncode": int(completed.returncode),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(status, ensure_ascii=False, indent=2))
    return status


def write_status(status: dict[str, object]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stock risk data/model/prediction pipeline.")
    parser.add_argument("--skip-collect", action="store_true", help="Use existing stock_price.csv and feature_data.csv.")
    parser.add_argument("--skip-train", action="store_true", help="Reuse the latest trained model.")
    parser.add_argument("--max-stale-days", type=int, default=7, help="Maximum allowed calendar age for price data.")
    parser.add_argument("--allow-stale", action="store_true", help="Do not fail when collected price data is stale.")
    parser.add_argument("--universe", choices=["krx", "static", "cached"], default="krx")
    parser.add_argument("--max-stocks", type=int, default=300)
    parser.add_argument("--prefer-cache", action="store_true", help="Use existing raw CSV cache before calling yfinance.")
    parser.add_argument("--download-timeout", type=int, default=8, help="Per-symbol yfinance timeout in seconds.")
    args = parser.parse_args()

    status: dict[str, object] = {
        "started_at": pd.Timestamp.now().isoformat(),
        "steps": [],
        "initial_freshness": data_freshness(args.max_stale_days),
    }

    try:
        for name, script in selected_steps(args):
            print(f"running {name}: {script}", flush=True)
            step_status = run_step(name, script, args)
            status["steps"].append(step_status)
            print(step_status["stdout"], flush=True)

        final_freshness = data_freshness(args.max_stale_days)
        status["final_freshness"] = final_freshness
        status["completed_at"] = pd.Timestamp.now().isoformat()
        status["success"] = True

        if not final_freshness["is_fresh"] and not args.allow_stale:
            status["success"] = False
            write_status(status)
            raise RuntimeError(
                f"price data is stale: latest={final_freshness['latest_price_date']}, "
                f"age_days={final_freshness['age_days']}, max_stale_days={args.max_stale_days}"
            )

        write_status(status)
        print(f"pipeline_status_file={STATUS_FILE.resolve()}")
    except Exception as exc:
        status["completed_at"] = pd.Timestamp.now().isoformat()
        status["success"] = False
        status["error"] = str(exc)
        write_status(status)
        raise


if __name__ == "__main__":
    main()
