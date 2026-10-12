from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model_explanations import explain_prediction, load_feature_context
from model_features import FEATURE_COLUMNS
from risk_thresholds import load_risk_thresholds, risk_level


PROCESSED_DIR = Path("data/processed")
MYSQL_IMPORT_DIR = Path("data/mysql_import_ready")
MODEL_PATH = Path("models/xgboost_risk_model_latest.joblib")

FEATURE_FILE = PROCESSED_DIR / "feature_data.csv"
ANOMALY_FILE = PROCESSED_DIR / "anomaly_result.csv"
PREDICTION_FILE = PROCESSED_DIR / "prediction_result.csv"
ALERT_FILE = PROCESSED_DIR / "alert_log.csv"
MYSQL_PREDICTION_FILE = MYSQL_IMPORT_DIR / "prediction_result.csv"
MYSQL_ALERT_FILE = MYSQL_IMPORT_DIR / "alert_log.csv"

MODEL_ID = 1
TARGET_WINDOW_DAYS = 5


def load_prediction_data() -> pd.DataFrame:
    features = pd.read_csv(FEATURE_FILE, dtype={"stock_code": str}, encoding="utf-8-sig")
    anomalies = pd.read_csv(ANOMALY_FILE, dtype={"stock_code": str}, encoding="utf-8-sig")

    features["stock_code"] = features["stock_code"].str.zfill(6)
    anomalies["stock_code"] = anomalies["stock_code"].str.zfill(6)
    features["trade_date"] = pd.to_datetime(features["date"])
    anomalies["trade_date"] = pd.to_datetime(anomalies["trade_date"])

    data = features.merge(
        anomalies[
            [
                "stock_code",
                "trade_date",
                "z_score",
                "ewma_value",
                "cusum_pos",
                "cusum_neg",
                "anomaly_score",
                "is_anomaly",
            ]
        ],
        on=["stock_code", "trade_date"],
        how="inner",
    )
    data = data.replace([np.inf, -np.inf], np.nan)
    return data.dropna(subset=FEATURE_COLUMNS).sort_values(["stock_code", "trade_date"]).reset_index(drop=True)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MYSQL_IMPORT_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    risk_threshold, danger_threshold = load_risk_thresholds()
    feature_context = load_feature_context()
    data = load_prediction_data()
    probabilities = model.predict_proba(data[FEATURE_COLUMNS])[:, 1]

    data["risk_probability"] = probabilities
    data["risk_level"] = data["risk_probability"].map(
        lambda probability: risk_level(float(probability), risk_threshold, danger_threshold)
    )

    prediction = data[["stock_code", "trade_date", "risk_probability", "risk_level"]].copy()
    prediction["model_id"] = MODEL_ID
    prediction["target_window_days"] = TARGET_WINDOW_DAYS
    prediction = prediction[
        ["stock_code", "trade_date", "model_id", "risk_probability", "risk_level", "target_window_days"]
    ]
    prediction["trade_date"] = prediction["trade_date"].dt.date

    explanations = data.apply(lambda row: explain_prediction(row, feature_context), axis=1, result_type="expand")
    alert = prediction[["stock_code", "trade_date", "risk_level"]].copy()
    alert["message"] = explanations[0]
    alert["top_factors"] = explanations[1]

    prediction.to_csv(PREDICTION_FILE, index=False, encoding="utf-8-sig")
    prediction.to_csv(MYSQL_PREDICTION_FILE, index=False, encoding="cp949")
    alert.to_csv(ALERT_FILE, index=False, encoding="utf-8-sig")
    alert.to_csv(MYSQL_ALERT_FILE, index=False, encoding="cp949")

    print(f"prediction_rows={len(prediction)}")
    print(prediction["risk_level"].value_counts().to_string())
    print(f"risk_threshold={risk_threshold:.2f}")
    print(f"danger_threshold={danger_threshold:.2f}")
    print(f"prediction_file={PREDICTION_FILE.resolve()}")
    print(f"alert_file={ALERT_FILE.resolve()}")


if __name__ == "__main__":
    main()
