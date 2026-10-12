from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


BACKTEST_DIR = Path("data/processed/daily_backtest")
PREDICTION_FILE = BACKTEST_DIR / "daily_predictions_latest.csv"
OUTPUT_DIR = BACKTEST_DIR / "threshold_sweep"
DEFAULT_THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.82, 0.85, 0.90, 0.95]
TOP_N_VALUES = [5, 10, 20]


def metric_row(frame: pd.DataFrame, threshold: float) -> dict[str, object]:
    comparable = frame[frame["is_label_available"] == 1].copy()
    y_true = comparable["risk_label"].astype(int)
    y_pred = (comparable["risk_probability"] >= threshold).astype(int)
    alert_mask = comparable["risk_probability"] >= threshold
    alert_frame = comparable[alert_mask]
    normal_frame = comparable[~alert_mask]

    row = {
        "threshold": float(threshold),
        "rows": int(len(comparable)),
        "actual_risks": int(y_true.sum()),
        "alert_count": int(alert_mask.sum()),
        "alert_rate": float(alert_mask.mean()),
        "avg_daily_alerts": float(
            frame.assign(predicted_alert=frame["risk_probability"] >= threshold)
            .groupby("trade_date")["predicted_alert"]
            .sum()
            .mean()
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "alert_mean_future_5d_return": (
            float(alert_frame["future_5d_return"].mean()) if not alert_frame.empty else None
        ),
        "alert_mean_future_min_return_1d": (
            float(alert_frame["future_min_return_1d"].mean()) if not alert_frame.empty else None
        ),
        "normal_risk_rate": float(normal_frame["risk_label"].mean()) if not normal_frame.empty else None,
        "alert_risk_rate": float(alert_frame["risk_label"].mean()) if not alert_frame.empty else None,
    }

    for top_n in TOP_N_VALUES:
        top_rows = (
            comparable.sort_values(["trade_date", "risk_probability"], ascending=[True, False])
            .groupby("trade_date", group_keys=False)
            .head(top_n)
        )
        row[f"top{top_n}_hit_rate"] = float(top_rows["risk_label"].mean()) if not top_rows.empty else None

    return row


def daily_rows(frame: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    comparable = frame[frame["is_label_available"] == 1].copy()
    rows = []
    for threshold in thresholds:
        for day, day_frame in comparable.groupby("trade_date", sort=True):
            y_true = day_frame["risk_label"].astype(int)
            y_pred = (day_frame["risk_probability"] >= threshold).astype(int)
            alerts = int(y_pred.sum())
            rows.append(
                {
                    "trade_date": day,
                    "threshold": float(threshold),
                    "rows": int(len(day_frame)),
                    "actual_risks": int(y_true.sum()),
                    "alert_count": alerts,
                    "alert_rate": float(alerts / len(day_frame)),
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                }
            )
    return pd.DataFrame(rows)


def parse_thresholds(raw: str | None) -> list[float]:
    if not raw:
        return DEFAULT_THRESHOLDS
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    return sorted(set(round(value, 4) for value in values))


def run(thresholds: list[float]) -> dict[str, object]:
    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(f"Missing prediction file: {PREDICTION_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(PREDICTION_FILE, dtype={"stock_code": str}, encoding="utf-8-sig")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["risk_probability", "risk_label"])

    summary = pd.DataFrame([metric_row(frame, threshold) for threshold in thresholds])
    daily = daily_rows(frame, thresholds)

    best_f1 = summary.sort_values(["f1", "precision"], ascending=False).iloc[0].to_dict()
    best_precision_20 = summary[summary["alert_rate"] <= 0.20]
    recommended = (
        best_precision_20.sort_values(["f1", "precision"], ascending=False).iloc[0].to_dict()
        if not best_precision_20.empty
        else best_f1
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = OUTPUT_DIR / f"threshold_summary_{run_id}.csv"
    daily_file = OUTPUT_DIR / f"threshold_daily_{run_id}.csv"
    report_file = OUTPUT_DIR / f"threshold_report_{run_id}.json"
    latest_summary_file = OUTPUT_DIR / "threshold_summary_latest.csv"
    latest_daily_file = OUTPUT_DIR / "threshold_daily_latest.csv"
    latest_report_file = OUTPUT_DIR / "threshold_report_latest.json"

    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")
    summary.to_csv(latest_summary_file, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_file, index=False, encoding="utf-8-sig")
    daily.to_csv(latest_daily_file, index=False, encoding="utf-8-sig")

    report = {
        "thresholds": thresholds,
        "rows": int(len(frame)),
        "scored_rows": int((frame["is_label_available"] == 1).sum()),
        "prediction_start": str(frame["trade_date"].min().date()),
        "prediction_end": str(frame["trade_date"].max().date()),
        "best_f1": best_f1,
        "recommended_for_operation": recommended,
        "files": {
            "summary": str(summary_file.resolve()),
            "daily": str(daily_file.resolve()),
            "report": str(report_file.resolve()),
        },
    }
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare daily backtest metrics across thresholds.")
    parser.add_argument("--thresholds", help="Comma-separated thresholds, for example: 0.7,0.8,0.9")
    args = parser.parse_args()

    report = run(parse_thresholds(args.thresholds))
    recommended = report["recommended_for_operation"]
    best = report["best_f1"]
    print(f"rows={report['rows']}")
    print(f"scored_rows={report['scored_rows']}")
    print(f"prediction_period={report['prediction_start']}~{report['prediction_end']}")
    print(f"best_f1_threshold={best['threshold']:.2f} f1={best['f1']:.4f} precision={best['precision']:.4f} recall={best['recall']:.4f}")
    print(
        "recommended_threshold="
        f"{recommended['threshold']:.2f} "
        f"alert_rate={recommended['alert_rate']:.4f} "
        f"precision={recommended['precision']:.4f} "
        f"recall={recommended['recall']:.4f} "
        f"f1={recommended['f1']:.4f}"
    )
    print(f"summary_file={report['files']['summary']}")
    print(f"daily_file={report['files']['daily']}")
    print(f"report_file={report['files']['report']}")


if __name__ == "__main__":
    main()
