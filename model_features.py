from __future__ import annotations


PRICE_FEATURE_COLUMNS = [
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "gap_open",
    "intraday_range",
    "close_position",
    "ma5_gap",
    "ma20_gap",
    "ma60_gap",
    "trend_strength_5_20",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "volatility_ratio_5_20",
    "volume_change_1d",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "down_days_5d",
    "down_days_10d",
    "price_volume_shock",
    "downside_volume_shock",
    "rsi_14",
    "rsi_change_3d",
    "macd",
    "macd_signal",
    "macd_hist",
    "macd_cross",
    "bollinger_width",
    "bollinger_position",
    "drawdown_20d",
]

ANOMALY_FEATURE_COLUMNS = [
    "z_score",
    "ewma_value",
    "cusum_pos",
    "cusum_neg",
    "anomaly_score",
]

FEATURE_COLUMNS = PRICE_FEATURE_COLUMNS + ANOMALY_FEATURE_COLUMNS
