from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from model_explanations import (
    FEATURE_IMPORTANCE_FILE,
    FEATURE_STATS_FILE,
    save_feature_importance,
    save_feature_stats,
)
from model_features import ANOMALY_FEATURE_COLUMNS, FEATURE_COLUMNS, PRICE_FEATURE_COLUMNS
from risk_thresholds import risk_level, select_danger_threshold


PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
METRICS_DIR = Path("data/model_metrics")
MYSQL_IMPORT_DIR = Path("data/mysql_import_ready")

FEATURE_FILE = PROCESSED_DIR / "feature_data.csv"
ANOMALY_FILE = PROCESSED_DIR / "anomaly_result.csv"
LABEL_FILE = PROCESSED_DIR / "risk_label.csv"

TARGET_WINDOW_DAYS = 5
TEST_RATIO = 0.2
VALIDATION_RATIO = 0.2
PURGE_GAP_DAYS = TARGET_WINDOW_DAYS
WALK_FORWARD_FOLDS = 3
WALK_FORWARD_MIN_TRAIN_RATIO = 0.5
RANDOM_STATE = 42
POSITIVE_CLASS_WEIGHT_FACTOR = 1.5
MODEL_N_ESTIMATORS = 500
MODEL_MAX_DEPTH = 2
MODEL_LEARNING_RATE = 0.05
MODEL_MIN_CHILD_WEIGHT = 5
MODEL_GAMMA = 0.5
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01
THRESHOLD_OPTIMIZATION_METRIC = "f1"
MIN_CAUTION_PRECISION = 0.4
TOP_N_VALUES = [5, 10, 20]


def load_training_data() -> pd.DataFrame:
    features = pd.read_csv(FEATURE_FILE, dtype={"stock_code": str}, encoding="utf-8-sig")
    anomalies = pd.read_csv(ANOMALY_FILE, dtype={"stock_code": str}, encoding="utf-8-sig")
    labels = pd.read_csv(LABEL_FILE, dtype={"stock_code": str}, encoding="utf-8-sig")

    features["stock_code"] = features["stock_code"].str.zfill(6)
    anomalies["stock_code"] = anomalies["stock_code"].str.zfill(6)
    labels["stock_code"] = labels["stock_code"].str.zfill(6)

    features["trade_date"] = pd.to_datetime(features["date"])
    anomalies["trade_date"] = pd.to_datetime(anomalies["trade_date"])
    labels["trade_date"] = pd.to_datetime(labels["trade_date"])

    required_label_columns = [
        "stock_code",
        "trade_date",
        "target_window_days",
        "future_5d_return",
        "future_min_return_1d",
        "risk_label",
        "is_label_available",
    ]
    missing_label_columns = sorted(set(required_label_columns) - set(labels.columns))
    if missing_label_columns:
        missing = ", ".join(missing_label_columns)
        raise ValueError(f"{LABEL_FILE} is missing required columns: {missing}")

    labels = labels[labels["is_label_available"] == 1].copy()

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
                "z_score",
                "ewma_value",
                "cusum_pos",
                "cusum_neg",
                "anomaly_score",
            ]
        ],
        on=["stock_code", "trade_date"],
        how="inner",
    ).merge(
        labels[
            [
                "stock_code",
                "trade_date",
                "target_window_days",
                "future_5d_return",
                "future_min_return_1d",
                "risk_label",
                "is_label_available",
            ]
        ],
        on=["stock_code", "trade_date"],
        how="inner",
    )

    data = data[data["target_window_days"] == TARGET_WINDOW_DAYS]
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=FEATURE_COLUMNS + ["risk_label"])
    return data.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)


def split_by_date(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    unique_dates = np.array(sorted(data["trade_date"].unique()))
    test_split_index = int(len(unique_dates) * (1 - TEST_RATIO))
    test_split_date = unique_dates[test_split_index]

    pre_test_dates = unique_dates[:test_split_index]
    validation_split_index = int(len(pre_test_dates) * (1 - VALIDATION_RATIO))
    validation_split_date = pre_test_dates[validation_split_index]
    train_end_index = max(validation_split_index - PURGE_GAP_DAYS, 0)
    validation_end_index = max(test_split_index - PURGE_GAP_DAYS, validation_split_index)
    train_end_date = unique_dates[train_end_index]
    validation_end_date = unique_dates[validation_end_index]

    train = data[data["trade_date"] < train_end_date].copy()
    validation = data[
        (data["trade_date"] >= validation_split_date) & (data["trade_date"] < validation_end_date)
    ].copy()
    test = data[data["trade_date"] >= test_split_date].copy()
    split_info = {
        "purge_gap_days": int(PURGE_GAP_DAYS),
        "validation_split_date": str(pd.Timestamp(validation_split_date).date()),
        "test_split_date": str(pd.Timestamp(test_split_date).date()),
        "train_validation_purged_dates": [
            str(pd.Timestamp(date).date()) for date in unique_dates[train_end_index:validation_split_index]
        ],
        "validation_test_purged_dates": [
            str(pd.Timestamp(date).date()) for date in unique_dates[validation_end_index:test_split_index]
        ],
    }
    return train, validation, test, split_info


def evaluate(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "threshold": float(threshold),
    }


def threshold_candidates() -> np.ndarray:
    return np.round(np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP), 2)


def threshold_curve(y_true: pd.Series, probabilities: np.ndarray) -> list[dict[str, float]]:
    return [evaluate(y_true, probabilities, float(threshold)) for threshold in threshold_candidates()]


def find_best_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, list[dict[str, float]], str]:
    curve = threshold_curve(y_true, probabilities)
    precision_candidates = [
        row for row in curve if row["precision"] >= MIN_CAUTION_PRECISION and row["recall"] > 0
    ]
    if precision_candidates:
        best = max(
            precision_candidates,
            key=lambda row: (
                row[THRESHOLD_OPTIMIZATION_METRIC],
                row["recall"],
                row["threshold"],
            ),
        )
        selection_rule = f"precision_at_least_{MIN_CAUTION_PRECISION:.2f}"
    else:
        best = max(
            curve,
            key=lambda row: (
                row[THRESHOLD_OPTIMIZATION_METRIC],
                row["precision"],
                row["threshold"],
            ),
        )
        selection_rule = f"fallback_max_{THRESHOLD_OPTIMIZATION_METRIC}"
    return float(best["threshold"]), curve, selection_rule


def class_weight(y_train: pd.Series) -> tuple[int, int, float]:
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    class_ratio = negative_count / positive_count if positive_count else 1.0
    return negative_count, positive_count, class_ratio * POSITIVE_CLASS_WEIGHT_FACTOR


def operational_metrics(
    evaluation_data: pd.DataFrame,
    probabilities: np.ndarray,
    risk_threshold: float,
    danger_threshold: float,
) -> dict[str, object]:
    frame = evaluation_data[
        [
            "stock_code",
            "trade_date",
            "risk_label",
            "future_5d_return",
            "future_min_return_1d",
        ]
    ].copy()
    frame["risk_probability"] = probabilities
    frame["predicted_alert"] = frame["risk_probability"] >= risk_threshold
    frame["risk_level"] = frame["risk_probability"].map(
        lambda probability: risk_level(float(probability), risk_threshold, danger_threshold)
    )
    frame["month"] = frame["trade_date"].dt.to_period("M").astype(str)

    daily_alerts = frame.groupby("trade_date")["predicted_alert"].sum()
    alert_frame = frame[frame["predicted_alert"]]

    level_rows = []
    for level, group in frame.groupby("risk_level", sort=False):
        level_rows.append(
            {
                "risk_level": level,
                "rows": int(len(group)),
                "row_rate": float(len(group) / len(frame)),
                "actual_risk_rate": float(group["risk_label"].mean()),
                "avg_probability": float(group["risk_probability"].mean()),
            }
        )

    top_n_rows = []
    for top_n in TOP_N_VALUES:
        top_rows = (
            frame.sort_values(["trade_date", "risk_probability"], ascending=[True, False])
            .groupby("trade_date", group_keys=False)
            .head(top_n)
        )
        daily_hit_rate = top_rows.groupby("trade_date")["risk_label"].mean()
        daily_risk_count = top_rows.groupby("trade_date")["risk_label"].sum()
        top_n_rows.append(
            {
                "top_n": int(top_n),
                "rows": int(len(top_rows)),
                "days": int(daily_hit_rate.size),
                "hit_rate": float(top_rows["risk_label"].mean()),
                "mean_daily_hit_rate": float(daily_hit_rate.mean()),
                "mean_daily_risk_count": float(daily_risk_count.mean()),
                "mean_future_5d_return": float(top_rows["future_5d_return"].mean()),
                "mean_future_min_return_1d": float(top_rows["future_min_return_1d"].mean()),
            }
        )

    if alert_frame.empty:
        monthly_alert_precision = []
        stock_alert_summary = []
        alert_return_summary = {
            "mean_future_5d_return": None,
            "median_future_5d_return": None,
            "mean_future_min_return_1d": None,
            "normal_mean_future_5d_return": float(frame.loc[~frame["predicted_alert"], "future_5d_return"].mean()),
            "alert_vs_normal_future_5d_return_gap": None,
        }
    else:
        monthly_alert_precision = (
            alert_frame.groupby("month")
            .agg(alerts=("risk_label", "size"), actual_risks=("risk_label", "sum"), precision=("risk_label", "mean"))
            .reset_index()
            .tail(24)
            .to_dict(orient="records")
        )
        stock_alert_summary = (
            alert_frame.groupby("stock_code")
            .agg(alerts=("risk_label", "size"), actual_risks=("risk_label", "sum"), precision=("risk_label", "mean"))
            .sort_values(["alerts", "precision"], ascending=[False, False])
            .head(30)
            .reset_index()
            .to_dict(orient="records")
        )
        normal_frame = frame[~frame["predicted_alert"]]
        alert_mean_return = float(alert_frame["future_5d_return"].mean())
        normal_mean_return = float(normal_frame["future_5d_return"].mean()) if not normal_frame.empty else None
        alert_return_summary = {
            "mean_future_5d_return": alert_mean_return,
            "median_future_5d_return": float(alert_frame["future_5d_return"].median()),
            "mean_future_min_return_1d": float(alert_frame["future_min_return_1d"].mean()),
            "normal_mean_future_5d_return": normal_mean_return,
            "alert_vs_normal_future_5d_return_gap": (
                float(alert_mean_return - normal_mean_return) if normal_mean_return is not None else None
            ),
        }

    return {
        "risk_threshold": float(risk_threshold),
        "danger_threshold": float(danger_threshold),
        "rows": int(len(frame)),
        "days": int(frame["trade_date"].nunique()),
        "stocks": int(frame["stock_code"].nunique()),
        "base_risk_rate": float(frame["risk_label"].mean()),
        "alert_count": int(frame["predicted_alert"].sum()),
        "alert_rate": float(frame["predicted_alert"].mean()),
        "avg_daily_alerts": float(daily_alerts.mean()),
        "median_daily_alerts": float(daily_alerts.median()),
        "alert_return_summary": alert_return_summary,
        "level_performance": level_rows,
        "top_n": top_n_rows,
        "monthly_alert_precision": monthly_alert_precision,
        "stock_alert_summary": stock_alert_summary,
    }


def fit_and_evaluate_fold(train: pd.DataFrame, validation: pd.DataFrame, fold_name: str) -> dict[str, object]:
    y_train = train["risk_label"].astype(int)
    y_validation = validation["risk_label"].astype(int)
    _, _, scale_pos_weight = class_weight(y_train)

    model = train_model(train[FEATURE_COLUMNS], y_train, scale_pos_weight)
    probabilities = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
    threshold, _, threshold_selection_rule = find_best_threshold(y_validation, probabilities)

    return {
        "fold": fold_name,
        "threshold": threshold,
        "threshold_selection_rule": threshold_selection_rule,
        "metrics": evaluate(y_validation, probabilities, threshold),
        "rows": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "train_positive": int((y_train == 1).sum()),
            "validation_positive": int((y_validation == 1).sum()),
        },
        "dates": {
            "train_start": str(train["trade_date"].min().date()),
            "train_end": str(train["trade_date"].max().date()),
            "validation_start": str(validation["trade_date"].min().date()),
            "validation_end": str(validation["trade_date"].max().date()),
        },
    }


def walk_forward_validation(data: pd.DataFrame) -> dict[str, object]:
    unique_dates = np.array(sorted(data["trade_date"].unique()))
    first_validation_index = int(len(unique_dates) * WALK_FORWARD_MIN_TRAIN_RATIO)
    validation_window = max(
        (len(unique_dates) - first_validation_index) // WALK_FORWARD_FOLDS,
        1,
    )

    folds = []
    for fold_index in range(WALK_FORWARD_FOLDS):
        validation_start_index = first_validation_index + fold_index * validation_window
        validation_end_index = (
            len(unique_dates)
            if fold_index == WALK_FORWARD_FOLDS - 1
            else min(validation_start_index + validation_window, len(unique_dates))
        )
        train_end_index = validation_start_index - PURGE_GAP_DAYS
        if train_end_index <= 0 or validation_start_index >= len(unique_dates):
            continue

        train = data[data["trade_date"] < unique_dates[train_end_index]].copy()
        validation_mask = data["trade_date"] >= unique_dates[validation_start_index]
        if validation_end_index < len(unique_dates):
            validation_mask &= data["trade_date"] < unique_dates[validation_end_index]
        validation = data[validation_mask].copy()
        if train.empty or validation.empty or train["risk_label"].nunique() < 2 or validation["risk_label"].nunique() < 2:
            continue

        fold = fit_and_evaluate_fold(train, validation, f"fold_{fold_index + 1}")
        fold["purged_dates"] = [
            str(pd.Timestamp(date).date()) for date in unique_dates[train_end_index:validation_start_index]
        ]
        folds.append(fold)

    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    summary = {
        key: float(np.mean([fold["metrics"][key] for fold in folds])) if folds else np.nan
        for key in metric_keys
    }
    return {
        "fold_count": len(folds),
        "configured_folds": int(WALK_FORWARD_FOLDS),
        "min_train_ratio": float(WALK_FORWARD_MIN_TRAIN_RATIO),
        "purge_gap_days": int(PURGE_GAP_DAYS),
        "summary": summary,
        "folds": folds,
    }


def train_model(x_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=MODEL_N_ESTIMATORS,
        max_depth=MODEL_MAX_DEPTH,
        learning_rate=MODEL_LEARNING_RATE,
        min_child_weight=MODEL_MIN_CHILD_WEIGHT,
        gamma=MODEL_GAMMA,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    return model


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MYSQL_IMPORT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_training_data()
    train, validation, test, split_info = split_by_date(data)
    walk_forward = walk_forward_validation(data)

    x_train = train[FEATURE_COLUMNS]
    y_train = train["risk_label"].astype(int)
    x_validation = validation[FEATURE_COLUMNS]
    y_validation = validation["risk_label"].astype(int)
    x_test = test[FEATURE_COLUMNS]
    y_test = test["risk_label"].astype(int)

    negative_count, positive_count, scale_pos_weight = class_weight(y_train)

    model = train_model(x_train, y_train, scale_pos_weight)

    train_probabilities = model.predict_proba(x_train)[:, 1]
    validation_probabilities = model.predict_proba(x_validation)[:, 1]
    test_probabilities = model.predict_proba(x_test)[:, 1]
    decision_threshold, validation_threshold_curve, threshold_selection_rule = find_best_threshold(
        y_validation,
        validation_probabilities,
    )
    danger_threshold = select_danger_threshold(decision_threshold, validation_threshold_curve)
    validation_operational_metrics = operational_metrics(
        validation,
        validation_probabilities,
        decision_threshold,
        danger_threshold,
    )
    test_operational_metrics = operational_metrics(
        test,
        test_probabilities,
        decision_threshold,
        danger_threshold,
    )
    metrics = {
        "train": evaluate(y_train, train_probabilities, decision_threshold),
        "validation": evaluate(y_validation, validation_probabilities, decision_threshold),
        "test": evaluate(y_test, test_probabilities, decision_threshold),
        "operational": {
            "validation": validation_operational_metrics,
            "test": test_operational_metrics,
        },
        "threshold_selection": {
            "source": "validation",
            "selection_rule": threshold_selection_rule,
            "min_caution_precision": float(MIN_CAUTION_PRECISION),
            "optimization_metric": THRESHOLD_OPTIMIZATION_METRIC,
            "threshold_min": float(THRESHOLD_MIN),
            "threshold_max": float(THRESHOLD_MAX),
            "threshold_step": float(THRESHOLD_STEP),
            "selected_threshold": float(decision_threshold),
            "danger_threshold": float(danger_threshold),
            "validation_curve": validation_threshold_curve,
        },
        "walk_forward_validation": walk_forward,
        "rows": {
            "total": int(len(data)),
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
            "train_positive": positive_count,
            "train_negative": negative_count,
            "validation_positive": int((y_validation == 1).sum()),
            "validation_negative": int((y_validation == 0).sum()),
            "test_positive": int((y_test == 1).sum()),
            "test_negative": int((y_test == 0).sum()),
            "scale_pos_weight": float(scale_pos_weight),
            "positive_class_weight_factor": float(POSITIVE_CLASS_WEIGHT_FACTOR),
        },
        "dates": {
            "train_start": str(train["trade_date"].min().date()),
            "train_end": str(train["trade_date"].max().date()),
            "validation_start": str(validation["trade_date"].min().date()),
            "validation_end": str(validation["trade_date"].max().date()),
            "test_start": str(test["trade_date"].min().date()),
            "test_end": str(test["trade_date"].max().date()),
        },
        "split": split_info,
        "feature_columns": FEATURE_COLUMNS,
        "model_params": {
            "n_estimators": int(MODEL_N_ESTIMATORS),
            "max_depth": int(MODEL_MAX_DEPTH),
            "learning_rate": float(MODEL_LEARNING_RATE),
            "min_child_weight": float(MODEL_MIN_CHILD_WEIGHT),
            "gamma": float(MODEL_GAMMA),
            "decision_threshold": decision_threshold,
            "danger_threshold": danger_threshold,
            "validation_ratio": float(VALIDATION_RATIO),
            "test_ratio": float(TEST_RATIO),
            "purge_gap_days": int(PURGE_GAP_DAYS),
            "walk_forward_folds": int(WALK_FORWARD_FOLDS),
        },
    }

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"xgboost_risk_model_{version}.joblib"
    metrics_path = METRICS_DIR / f"xgboost_risk_model_{version}_metrics.json"
    feature_importance_path = METRICS_DIR / f"xgboost_risk_model_{version}_feature_importance.csv"
    feature_stats_path = METRICS_DIR / f"xgboost_risk_model_{version}_feature_stats.json"
    latest_model_path = MODEL_DIR / "xgboost_risk_model_latest.joblib"
    latest_metrics_path = METRICS_DIR / "xgboost_risk_model_latest_metrics.json"

    joblib.dump(model, model_path)
    joblib.dump(model, latest_model_path)
    feature_importance = save_feature_importance(model, FEATURE_COLUMNS, feature_importance_path)
    save_feature_importance(model, FEATURE_COLUMNS, FEATURE_IMPORTANCE_FILE)
    save_feature_stats(x_train, feature_stats_path)
    save_feature_stats(x_train, FEATURE_STATS_FILE)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    model_info = pd.DataFrame(
        [
            {
                "model_name": "xgboost_risk_model",
                "model_type": "XGBoost",
                "model_version": version,
                "target_window_days": TARGET_WINDOW_DAYS,
                "model_path": str(model_path.resolve()),
                "feature_columns": json.dumps(FEATURE_COLUMNS, ensure_ascii=False),
                "train_start_date": metrics["dates"]["train_start"],
                "train_end_date": metrics["dates"]["train_end"],
                "metrics": json.dumps(metrics, ensure_ascii=False),
                "is_active": 1,
            }
        ]
    )
    model_info.to_csv(MYSQL_IMPORT_DIR / "model_info.csv", index=False, encoding="cp949")

    print(f"model_path={model_path.resolve()}")
    print(f"latest_model_path={latest_model_path.resolve()}")
    print(f"metrics_path={metrics_path.resolve()}")
    print(f"feature_importance_path={feature_importance_path.resolve()}")
    print(f"top_feature={feature_importance.iloc[0]['feature']}")
    print(f"model_info_file={(MYSQL_IMPORT_DIR / 'model_info.csv').resolve()}")
    print(f"rows_total={len(data)}")
    print(f"rows_train={len(train)}")
    print(f"rows_validation={len(validation)}")
    print(f"rows_test={len(test)}")
    print(f"train_date={metrics['dates']['train_start']}~{metrics['dates']['train_end']}")
    print(f"validation_date={metrics['dates']['validation_start']}~{metrics['dates']['validation_end']}")
    print(f"test_date={metrics['dates']['test_start']}~{metrics['dates']['test_end']}")
    print(f"validation_precision={metrics['validation']['precision']:.4f}")
    print(f"validation_recall={metrics['validation']['recall']:.4f}")
    print(f"validation_f1={metrics['validation']['f1']:.4f}")
    print(f"test_accuracy={metrics['test']['accuracy']:.4f}")
    print(f"test_precision={metrics['test']['precision']:.4f}")
    print(f"test_recall={metrics['test']['recall']:.4f}")
    print(f"test_f1={metrics['test']['f1']:.4f}")
    print(f"test_roc_auc={metrics['test']['roc_auc']:.4f}")
    print(f"decision_threshold={decision_threshold:.2f}")
    print(f"danger_threshold={danger_threshold:.2f}")
    print(f"test_alert_rate={test_operational_metrics['alert_rate']:.4f}")
    print(f"test_avg_daily_alerts={test_operational_metrics['avg_daily_alerts']:.2f}")
    print(f"test_top10_hit_rate={test_operational_metrics['top_n'][1]['hit_rate']:.4f}")
    print(f"purge_gap_days={PURGE_GAP_DAYS}")
    print(f"walk_forward_folds={walk_forward['fold_count']}")
    print(f"walk_forward_f1_mean={walk_forward['summary']['f1']:.4f}")
    print(f"walk_forward_roc_auc_mean={walk_forward['summary']['roc_auc']:.4f}")
    print(f"positive_class_weight_factor={POSITIVE_CLASS_WEIGHT_FACTOR:.2f}")
    print(f"n_estimators={MODEL_N_ESTIMATORS}")
    print(f"max_depth={MODEL_MAX_DEPTH}")
    print(f"learning_rate={MODEL_LEARNING_RATE:.2f}")
    print(f"min_child_weight={MODEL_MIN_CHILD_WEIGHT}")
    print(f"gamma={MODEL_GAMMA:.2f}")


if __name__ == "__main__":
    main()
