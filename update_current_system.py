from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path


STATUS_FILE = Path("data/processed/current_update_status.json")


def run_command(name: str, command: list[str], env: dict[str, str]) -> dict[str, object]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    elapsed = time.perf_counter() - started
    result = {
        "name": name,
        "command": command,
        "returncode": int(completed.returncode),
        "elapsed_seconds": round(elapsed, 3),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def write_status(status: dict[str, object]) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def run_once(args: argparse.Namespace) -> dict[str, object]:
    today = date.today().isoformat()
    env = os.environ.copy()
    if args.mysql_password:
        env["MYSQL_ENABLED"] = "1"
        env["MYSQL_PASSWORD"] = args.mysql_password
    if args.mysql_user:
        env["MYSQL_USER"] = args.mysql_user
    if args.mysql_database:
        env["MYSQL_DATABASE"] = args.mysql_database

    status: dict[str, object] = {
        "started_at": date.today().isoformat(),
        "target_date": today,
        "steps": [],
        "success": False,
    }
    write_status(status)

    status["steps"].append(run_command("market_index", [sys.executable, "collect_market_index.py"], env))
    write_status(status)

    pipeline_command = [
        sys.executable,
        "run_pipeline.py",
        "--universe",
        "krx" if args.refresh_universe else "cached",
        "--max-stocks",
        str(args.max_stocks),
        "--download-timeout",
        str(args.download_timeout),
        "--allow-stale",
    ]
    if not args.refresh_prices:
        pipeline_command.append("--prefer-cache")
    if args.skip_train:
        pipeline_command.append("--skip-train")
    status["steps"].append(run_command("pipeline", pipeline_command, env))
    write_status(status)

    status["steps"].append(run_command("sector_mapping", [sys.executable, "apply_sector_mapping.py"], env))
    write_status(status)

    status["steps"].append(
        run_command(
            "daily_backtest",
            [
                sys.executable,
                "daily_incremental_backtest.py",
                "--train-cutoff",
                args.train_cutoff,
                "--end-date",
                today,
            ],
            env,
        )
    )
    write_status(status)

    status["steps"].append(run_command("threshold_sweep", [sys.executable, "threshold_sweep_backtest.py"], env))
    write_status(status)

    if args.load_mysql:
        if not args.mysql_password:
            raise RuntimeError("--load-mysql requires --mysql-password or MYSQL_PASSWORD.")
        status["steps"].append(
            run_command(
                "mysql_load",
                [
                    sys.executable,
                    "database/load_csv_to_mysql.py",
                    "--password",
                    args.mysql_password,
                    "--replace",
                    "--batch-size",
                    str(args.mysql_batch_size),
                ],
                env,
            )
        )
        write_status(status)

    status["completed_at"] = date.today().isoformat()
    status["success"] = True
    write_status(status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Update current stock data, backtest files, and optional MySQL tables.")
    parser.add_argument("--max-stocks", type=int, default=300)
    parser.add_argument("--download-timeout", type=int, default=8)
    parser.add_argument("--train-cutoff", default="2026-06-30")
    parser.add_argument("--skip-train", action="store_true", help="Reuse the latest model while updating data and predictions.")
    parser.add_argument("--refresh-universe", action="store_true", help="Fetch a fresh KRX listing instead of using cached stock info.")
    parser.add_argument("--refresh-prices", action="store_true", help="Call yfinance even when raw cache files already exist.")
    parser.add_argument("--load-mysql", action="store_true", help="Load generated CSV files into MySQL after updating.")
    parser.add_argument("--mysql-password", default=os.getenv("MYSQL_PASSWORD", ""))
    parser.add_argument("--mysql-user", default=os.getenv("MYSQL_USER", "root"))
    parser.add_argument("--mysql-database", default=os.getenv("MYSQL_DATABASE", "stock_risk_project"))
    parser.add_argument("--mysql-batch-size", type=int, default=5000)
    parser.add_argument("--repeat-minutes", type=float, default=0, help="Run continuously every N minutes. 0 runs once.")
    args = parser.parse_args()

    while True:
        status = run_once(args)
        print(f"update_success target_date={status['target_date']}")
        print(f"status_file={STATUS_FILE.resolve()}")
        if args.repeat_minutes <= 0:
            break
        time.sleep(args.repeat_minutes * 60)


if __name__ == "__main__":
    main()
