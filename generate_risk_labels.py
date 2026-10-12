from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROCESSED_DIR = Path("data/processed")
MYSQL_IMPORT_DIR = Path("data/mysql_import_ready")
FEATURE_FILE = PROCESSED_DIR / "feature_data.csv"
OUTPUT_FILE = PROCESSED_DIR / "risk_label.csv"
MYSQL_OUTPUT_FILE = MYSQL_IMPORT_DIR / "risk_label.csv"

TARGET_WINDOW_DAYS = 5
CUM_RETURN_DROP_THRESHOLD = -0.08
DAILY_RETURN_DROP_THRESHOLD = -0.06


def future_rolling_min(series: pd.Series, window: int) -> pd.Series:
    return series.shift(-1).rolling(window=window, min_periods=1).min().shift(-(window - 1))


def label_for_stock(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("trade_date").copy()
    close = group["adj_close"]

    future_close = close.shift(-TARGET_WINDOW_DAYS)
    group["future_5d_return"] = future_close / close - 1
    group["future_min_return_1d"] = future_rolling_min(group["return_1d"], TARGET_WINDOW_DAYS)
    group["is_label_available"] = group["future_5d_return"].notna().astype(int)

    cumulative_drop_risk = group["future_5d_return"] <= CUM_RETURN_DROP_THRESHOLD
    daily_drop_risk = group["future_min_return_1d"] <= DAILY_RETURN_DROP_THRESHOLD

    group["risk_label"] = (cumulative_drop_risk | daily_drop_risk).astype(int)

    reasons = np.select(
        [cumulative_drop_risk, daily_drop_risk],
        [
            "향후 5거래일 누적수익률 -8% 이하",
            "향후 5거래일 중 일간수익률 -6% 이하",
        ],
        default="위험 조건 없음",
    )
    group["label_reason"] = reasons
    group.loc[group["future_5d_return"].isna(), ["risk_label", "label_reason"]] = [
        0,
        "미래 5거래일 데이터 부족",
    ]

    return group


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MYSQL_IMPORT_DIR.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(FEATURE_FILE, dtype={"stock_code": str}, encoding="utf-8-sig")

    features["stock_code"] = features["stock_code"].str.zfill(6)
    features["trade_date"] = pd.to_datetime(features["date"]).dt.date

    merged = features[["stock_code", "trade_date", "adj_close", "return_1d"]].copy()

    labeled = pd.concat(
        [label_for_stock(group) for _, group in merged.groupby("stock_code", sort=False)],
        ignore_index=True,
    )

    output_columns = [
        "stock_code",
        "trade_date",
        "target_window_days",
        "future_5d_return",
        "future_min_return_1d",
        "risk_label",
        "is_label_available",
        "label_reason",
    ]
    labeled["target_window_days"] = TARGET_WINDOW_DAYS
    result = labeled[output_columns].sort_values(["stock_code", "trade_date"]).reset_index(drop=True)

    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    result.to_csv(MYSQL_OUTPUT_FILE, index=False, encoding="cp949")

    print(f"rows={len(result)}")
    print(f"stocks={result['stock_code'].nunique()}")
    print(f"risk_labels={int(result['risk_label'].sum())}")
    print(f"risk_rate={result['risk_label'].mean():.4f}")
    print(f"date_min={result['trade_date'].min()}")
    print(f"date_max={result['trade_date'].max()}")
    print(f"output_file={OUTPUT_FILE.resolve()}")
    print(f"mysql_import_file={MYSQL_OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
