from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from anomaly_features import add_anomaly_features


PROCESSED_DIR = Path("data/processed")
MYSQL_IMPORT_DIR = Path("data/mysql_import_ready")
INPUT_FILE = PROCESSED_DIR / "feature_data.csv"
OUTPUT_FILE = PROCESSED_DIR / "anomaly_result.csv"
MYSQL_OUTPUT_FILE = MYSQL_IMPORT_DIR / "anomaly_result.csv"


def detect_for_stock(group: pd.DataFrame) -> pd.DataFrame:
    return add_anomaly_features(group.sort_values("trade_date"))


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MYSQL_IMPORT_DIR.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(INPUT_FILE, dtype={"stock_code": str}, encoding="utf-8-sig")
    features["stock_code"] = features["stock_code"].str.zfill(6)
    features["trade_date"] = pd.to_datetime(features["date"]).dt.date

    result = pd.concat(
        [detect_for_stock(group) for _, group in features.groupby("stock_code", sort=False)],
        ignore_index=True,
    ).replace([np.inf, -np.inf], np.nan)

    output_columns = [
        "stock_code",
        "trade_date",
        "z_score",
        "ewma_value",
        "cusum_pos",
        "cusum_neg",
        "anomaly_score",
        "is_anomaly",
    ]
    result = result[output_columns].sort_values(["stock_code", "trade_date"]).reset_index(drop=True)

    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    result.to_csv(MYSQL_OUTPUT_FILE, index=False, encoding="cp949")

    print(f"rows={len(result)}")
    print(f"stocks={result['stock_code'].nunique()}")
    print(f"anomalies={int(result['is_anomaly'].sum())}")
    print(f"date_min={result['trade_date'].min()}")
    print(f"date_max={result['trade_date'].max()}")
    print(f"output_file={OUTPUT_FILE.resolve()}")
    print(f"mysql_import_file={MYSQL_OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
