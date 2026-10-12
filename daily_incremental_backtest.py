from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from model_features import ANOMALY_FEATURE_COLUMNS, FEATURE_COLUMNS, PRICE_FEATURE_COLUMNS
from risk_thresholds import risk_level, select_danger_threshold
from train_xgboost_model import (
    TARGET_WINDOW_DAYS,
    class_weight,
    find_best_threshold,
    train_model,
)


PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = PROCESSED_DIR / "daily_backtest"
FEATURE_FILE = PROCESSED_DIR / "feature_data.csv"
ANOMALY_FILE = PROCESSED_DIR / "anomaly_result.csv"
LABEL_FILE = PROCESSED_DIR / "risk_label.csv"

DEFAULT_TRAIN_CUTOFF = "2026-06-30"
VALIDATION_RATIO = 0.2
PURGE_GAP_DAYS = TARGET_WINDOW_DAYS
TOP_N = 10


def load_backtest_data() -> pd.DataFrame:
    features = pd.read_csv(FEATURE_FILE, dtype={"stock_code": str}, encoding="utf-8-sig")
    anomalies = pd.read_csv(ANOMALY_FILE, dtype={"stock_code": str}, encoding="utf-8-sig")
    labels = pd.read_csv(LABEL_FILE, dtype={"stock_code": str}, encoding="utf-8-sig")

    for frame in [features, anomalies, labels]:
        frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)

    features["trade_date"] = pd.to_datetime(features["date"])
    anomalies["trade_date"] = pd.to_datetime(anomalies["trade_date"])
    labels["trade_date"] = pd.to_datetime(labels["trade_date"])

    label_columns = [
        "stock_code",
        "trade_date",
        "target_window_days",
        "future_5d_return",
        "future_min_return_1d",
        "risk_label",
        "is_label_available",
    ]
    missing = sorted(set(label_columns) - set(labels.columns))
    if missing:
        raise ValueError(f"{LABEL_FILE} is missing required columns: {', '.join(missing)}")

    data = features[
        [
            "stock_code",
            "trade_date",
            *PRICE_FEATURE_COLUMNS,
        ]
    ].merge(
        anomalies[
            [
                "stock_code",
                "trade_date",
                *ANOMALY_FEATURE_COLUMNS,
            ]
        ],
        on=["stock_code", "trade_date"],
        how="inner",
    ).merge(
        labels[label_columns],
        on=["stock_code", "trade_date"],
        how="left",
    )

    data = data[data["target_window_days"].fillna(TARGET_WINDOW_DAYS) == TARGET_WINDOW_DAYS]
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=FEATURE_COLUMNS)
    data = data.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    data["label_available_date"] = data.groupby("stock_code")["trade_date"].shift(-TARGET_WINDOW_DAYS)
    return data.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)


def split_training_data(trainable: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(trainable["trade_date"].unique()))
    split_index = int(len(dates) * (1 - VALIDATION_RATIO))
    validation_start = dates[split_index]
    train_end_index = max(split_index - PURGE_GAP_DAYS, 0)
    train_end = dates[train_end_index]

    fit_train = trainable[trainable["trade_date"] < train_end].copy()
    validation = trainable[trainable["trade_date"] >= validation_start].copy()
    if fit_train.empty or validation.empty:
        raise ValueError("Not enough training data for a purged validation split.")
    return fit_train, validation


def train_cutoff_model(data: pd.DataFrame, train_cutoff: pd.Timestamp):
    trainable = data[
        (data["trade_date"] <= train_cutoff)
        & (data["label_available_date"] <= train_cutoff)
        & (data["is_label_available"] == 1)
    ].dropna(subset=FEATURE_COLUMNS + ["risk_label"])

    if trainable.empty or trainable["risk_label"].nunique() < 2:
        raise ValueError("Training cutoff does not contain enough observable labeled classes.")

    fit_train, validation = split_training_data(trainable)
    _, _, validation_scale_pos_weight = class_weight(fit_train["risk_label"].astype(int))
    validation_model = train_model(
        fit_train[FEATURE_COLUMNS],
        fit_train["risk_label"].astype(int),
        validation_scale_pos_weight,
    )
    validation_probabilities = validation_model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
    threshold, validation_curve, selection_rule = find_best_threshold(
        validation["risk_label"].astype(int),
        validation_probabilities,
    )
    danger_threshold = select_danger_threshold(threshold, validation_curve)

    _, _, final_scale_pos_weight = class_weight(trainable["risk_label"].astype(int))
    final_model = train_model(
        trainable[FEATURE_COLUMNS],
        trainable["risk_label"].astype(int),
        final_scale_pos_weight,
    )

    train_info = {
        "rows": int(len(trainable)),
        "stocks": int(trainable["stock_code"].nunique()),
        "start_date": str(trainable["trade_date"].min().date()),
        "last_feature_date": str(train_cutoff.date()),
        "last_observable_label_date": str(trainable["trade_date"].max().date()),
        "positive_rate": float(trainable["risk_label"].mean()),
        "threshold": float(threshold),
        "danger_threshold": float(danger_threshold),
        "threshold_selection_rule": selection_rule,
    }
    return final_model, threshold, danger_threshold, train_info


def metric_value(func, y_true: pd.Series, y_pred: pd.Series) -> float | None:
    if y_true.empty:
        return None
    return float(func(y_true, y_pred, zero_division=0))


def summarize_day(frame: pd.DataFrame, day: pd.Timestamp, threshold: float, danger_threshold: float) -> dict[str, object]:
    comparable = frame[frame["is_label_available"] == 1].copy()
    alerts = frame[frame["predicted_alert"]].copy()
    comparable_alerts = comparable[comparable["predicted_alert"]].copy()
    top_rows = frame.sort_values("risk_probability", ascending=False).head(TOP_N)
    comparable_top_rows = top_rows[top_rows["is_label_available"] == 1]

    y_true = comparable["risk_label"].astype(int)
    y_pred = comparable["predicted_alert"].astype(int)

    return {
        "trade_date": str(day.date()),
        "rows": int(len(frame)),
        "comparable_rows": int(len(comparable)),
        "actual_risks": int(y_true.sum()) if not y_true.empty else None,
        "alerts": int(len(alerts)),
        "cautions": int((frame["risk_level"] == "CAUTION").sum()),
        "dangers": int((frame["risk_level"] == "DANGER").sum()),
        "alert_rate": float(len(alerts) / len(frame)) if len(frame) else None,
        "precision": metric_value(precision_score, y_true, y_pred),
        "recall": metric_value(recall_score, y_true, y_pred),
        "f1": metric_value(f1_score, y_true, y_pred),
        "top10_hit_rate": float(comparable_top_rows["risk_label"].mean()) if not comparable_top_rows.empty else None,
        "top10_actual_risks": int(comparable_top_rows["risk_label"].sum()) if not comparable_top_rows.empty else None,
        "mean_probability": float(frame["risk_probability"].mean()),
        "alert_mean_future_5d_return": (
            float(comparable_alerts["future_5d_return"].mean()) if not comparable_alerts.empty else None
        ),
        "alert_mean_future_min_return_1d": (
            float(comparable_alerts["future_min_return_1d"].mean()) if not comparable_alerts.empty else None
        ),
        "risk_threshold": float(threshold),
        "danger_threshold": float(danger_threshold),
    }


def run_backtest(train_cutoff: str, end_date: str) -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp(train_cutoff)
    end = pd.Timestamp(end_date)
    data = load_backtest_data()

    model, threshold, danger_threshold, train_info = train_cutoff_model(data, cutoff)

    prediction_window = data[(data["trade_date"] > cutoff) & (data["trade_date"] <= end)].copy()
    if prediction_window.empty:
        raise ValueError("No prediction rows exist after the training cutoff.")

    prediction_window["risk_probability"] = model.predict_proba(prediction_window[FEATURE_COLUMNS])[:, 1]
    prediction_window["predicted_alert"] = prediction_window["risk_probability"] >= threshold
    prediction_window["risk_level"] = prediction_window["risk_probability"].map(
        lambda probability: risk_level(float(probability), threshold, danger_threshold)
    )

    daily_rows = [
        summarize_day(day_frame.copy(), day, threshold, danger_threshold)
        for day, day_frame in prediction_window.groupby("trade_date", sort=True)
    ]
    daily_summary = pd.DataFrame(daily_rows)

    comparable = prediction_window[prediction_window["is_label_available"] == 1].copy()
    overall = {
        "prediction_rows": int(len(prediction_window)),
        "prediction_days": int(prediction_window["trade_date"].nunique()),
        "first_prediction_date": str(prediction_window["trade_date"].min().date()),
        "last_prediction_date": str(prediction_window["trade_date"].max().date()),
        "comparable_rows": int(len(comparable)),
        "comparable_days": int(comparable["trade_date"].nunique()),
        "last_comparable_date": (
            str(comparable["trade_date"].max().date()) if not comparable.empty else None
        ),
        "alert_count": int(prediction_window["predicted_alert"].sum()),
        "alert_rate": float(prediction_window["predicted_alert"].mean()),
        "danger_count": int((prediction_window["risk_level"] == "DANGER").sum()),
        "precision": metric_value(
            precision_score,
            comparable["risk_label"].astype(int),
            comparable["predicted_alert"].astype(int),
        ),
        "recall": metric_value(
            recall_score,
            comparable["risk_label"].astype(int),
            comparable["predicted_alert"].astype(int),
        ),
        "f1": metric_value(
            f1_score,
            comparable["risk_label"].astype(int),
            comparable["predicted_alert"].astype(int),
        ),
        "top10_mean_hit_rate": float(daily_summary["top10_hit_rate"].dropna().mean()),
        "alert_mean_future_5d_return": (
            float(comparable.loc[comparable["predicted_alert"], "future_5d_return"].mean())
            if comparable["predicted_alert"].any()
            else None
        ),
        "alert_mean_future_min_return_1d": (
            float(comparable.loc[comparable["predicted_alert"], "future_min_return_1d"].mean())
            if comparable["predicted_alert"].any()
            else None
        ),
    }

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    prediction_file = OUTPUT_DIR / f"daily_predictions_{run_id}.csv"
    daily_file = OUTPUT_DIR / f"daily_summary_{run_id}.csv"
    report_file = OUTPUT_DIR / f"daily_backtest_report_{run_id}.json"
    latest_prediction_file = OUTPUT_DIR / "daily_predictions_latest.csv"
    latest_daily_file = OUTPUT_DIR / "daily_summary_latest.csv"
    latest_report_file = OUTPUT_DIR / "daily_backtest_report_latest.json"

    prediction_window.to_csv(prediction_file, index=False, encoding="utf-8-sig")
    prediction_window.to_csv(latest_prediction_file, index=False, encoding="utf-8-sig")
    daily_summary.to_csv(daily_file, index=False, encoding="utf-8-sig")
    daily_summary.to_csv(latest_daily_file, index=False, encoding="utf-8-sig")

    report = {
        "train_info": train_info,
        "overall": overall,
        "files": {
            "prediction_file": str(prediction_file.resolve()),
            "daily_summary_file": str(daily_file.resolve()),
            "report_file": str(report_file.resolve()),
        },
        "notes": [
            "The original data files are preserved. The cutoff is enforced in memory.",
            "Rows after the last observable label date are predicted but not scored yet.",
        ],
    }
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a daily incremental prediction backtest.")
    parser.add_argument("--train-cutoff", default=DEFAULT_TRAIN_CUTOFF)
    parser.add_argument("--end-date", default=date.today().isoformat())
    args = parser.parse_args()

    report = run_backtest(args.train_cutoff, args.end_date)
    train = report["train_info"]
    overall = report["overall"]
    print(f"train_cutoff={train['last_feature_date']}")
    print(f"last_observable_label_date={train['last_observable_label_date']}")
    print(f"prediction_days={overall['prediction_days']}")
    print(f"prediction_rows={overall['prediction_rows']}")
    print(f"last_prediction_date={overall['last_prediction_date']}")
    print(f"comparable_days={overall['comparable_days']}")
    print(f"last_comparable_date={overall['last_comparable_date']}")
    print(f"threshold={train['threshold']:.2f}")
    print(f"danger_threshold={train['danger_threshold']:.2f}")
    print(f"alert_count={overall['alert_count']}")
    print(f"alert_rate={overall['alert_rate']:.4f}")
    print(f"precision={overall['precision']:.4f}")
    print(f"recall={overall['recall']:.4f}")
    print(f"f1={overall['f1']:.4f}")
    print(f"top10_mean_hit_rate={overall['top10_mean_hit_rate']:.4f}")
    print(f"alert_mean_future_min_return_1d={overall['alert_mean_future_min_return_1d']:.4f}")
    print(f"report_file={report['files']['report_file']}")


if __name__ == "__main__":
    main()
