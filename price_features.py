from __future__ import annotations

import numpy as np
import pandas as pd


def add_price_features(price_frame: pd.DataFrame) -> pd.DataFrame:
    frame = price_frame.copy()
    has_stock_code = "stock_code" in frame.columns

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
        sort_columns = ["date"]
    else:
        sort_columns = []

    if has_stock_code:
        frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)
        sort_columns = ["stock_code", *sort_columns]
        group_column = "stock_code"
    else:
        group_column = "_feature_group"
        frame[group_column] = 0

    if sort_columns:
        frame = frame.sort_values(sort_columns).reset_index(drop=True)

    grouped = frame.groupby(group_column, group_keys=False)

    frame["return_1d"] = grouped["adj_close"].pct_change()
    frame["return_3d"] = grouped["adj_close"].pct_change(3)
    frame["return_5d"] = grouped["adj_close"].pct_change(5)
    frame["return_10d"] = grouped["adj_close"].pct_change(10)
    frame["return_20d"] = grouped["adj_close"].pct_change(20)

    previous_close = grouped["adj_close"].shift(1)
    frame["gap_open"] = (frame["open"] - previous_close) / previous_close
    frame["intraday_range"] = (frame["high"] - frame["low"]) / frame["adj_close"]
    frame["close_position"] = (frame["adj_close"] - frame["low"]) / (frame["high"] - frame["low"])

    ma5 = grouped["adj_close"].transform(lambda series: series.rolling(5, min_periods=5).mean())
    frame["ma20"] = grouped["adj_close"].transform(lambda series: series.rolling(20, min_periods=20).mean())
    ma60 = grouped["adj_close"].transform(lambda series: series.rolling(60, min_periods=60).mean())
    frame["ma5_gap"] = (frame["adj_close"] - ma5) / ma5
    frame["ma20_gap"] = (frame["adj_close"] - frame["ma20"]) / frame["ma20"]
    frame["ma60_gap"] = (frame["adj_close"] - ma60) / ma60
    frame["trend_strength_5_20"] = ma5 / frame["ma20"] - 1

    frame["volatility_5d"] = grouped["return_1d"].transform(lambda series: series.rolling(5, min_periods=5).std())
    frame["volatility_10d"] = grouped["return_1d"].transform(lambda series: series.rolling(10, min_periods=10).std())
    frame["volatility_20d"] = grouped["return_1d"].transform(lambda series: series.rolling(20, min_periods=20).std())
    frame["volatility_ratio_5_20"] = frame["volatility_5d"] / frame["volatility_20d"]

    frame["volume_change_1d"] = grouped["volume"].pct_change()
    volume_ma5 = grouped["volume"].transform(lambda series: series.rolling(5, min_periods=5).mean())
    volume_ma20 = grouped["volume"].transform(lambda series: series.rolling(20, min_periods=20).mean())
    close_std20 = grouped["adj_close"].transform(lambda series: series.rolling(20, min_periods=20).std())
    frame["volume_ratio_5d"] = frame["volume"] / volume_ma5
    frame["volume_ratio_20d"] = frame["volume"] / volume_ma20
    frame["down_days_5d"] = (
        (frame["return_1d"] < 0)
        .astype(int)
        .groupby(frame[group_column])
        .transform(lambda series: series.rolling(5, min_periods=5).sum())
    )
    frame["down_days_10d"] = (
        (frame["return_1d"] < 0)
        .astype(int)
        .groupby(frame[group_column])
        .transform(lambda series: series.rolling(10, min_periods=10).sum())
    )
    frame["price_volume_shock"] = frame["return_1d"].abs() * frame["volume_ratio_20d"]
    frame["downside_volume_shock"] = frame["return_1d"].clip(upper=0).abs() * frame["volume_ratio_20d"]

    delta = grouped["adj_close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.groupby(frame[group_column]).transform(
        lambda series: series.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    )
    avg_loss = loss.groupby(frame[group_column]).transform(
        lambda series: series.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    )
    rs = avg_gain / avg_loss
    frame["rsi_14"] = 100 - (100 / (1 + rs))
    frame["rsi_change_3d"] = grouped["rsi_14"].diff(3)

    ema12 = grouped["adj_close"].transform(lambda series: series.ewm(span=12, min_periods=12, adjust=False).mean())
    ema26 = grouped["adj_close"].transform(lambda series: series.ewm(span=26, min_periods=26, adjust=False).mean())
    frame["macd"] = ema12 - ema26
    frame["macd_signal"] = grouped["macd"].transform(
        lambda series: series.ewm(span=9, min_periods=9, adjust=False).mean()
    )
    frame["macd_hist"] = frame["macd"] - frame["macd_signal"]
    frame["macd_cross"] = np.sign(frame["macd_hist"]).groupby(frame[group_column]).diff().fillna(0)

    frame["bollinger_upper"] = frame["ma20"] + close_std20 * 2
    frame["bollinger_lower"] = frame["ma20"] - close_std20 * 2
    frame["bollinger_width"] = (frame["bollinger_upper"] - frame["bollinger_lower"]) / frame["ma20"]
    frame["bollinger_position"] = (frame["adj_close"] - frame["bollinger_lower"]) / (
        frame["bollinger_upper"] - frame["bollinger_lower"]
    )

    rolling_high20 = grouped["adj_close"].transform(lambda series: series.rolling(20, min_periods=20).max())
    frame["drawdown_20d"] = frame["adj_close"] / rolling_high20 - 1

    if not has_stock_code:
        frame = frame.drop(columns=[group_column])

    return frame.replace([np.inf, -np.inf], np.nan)
