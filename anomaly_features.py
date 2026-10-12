from __future__ import annotations

import numpy as np
import pandas as pd


ROLLING_WINDOW = 20
EWMA_SPAN = 20
CUSUM_DRIFT = 0.5
CUSUM_THRESHOLD = 5.0
Z_SCORE_THRESHOLD = 3.0
EWMA_THRESHOLD = 3.0


def standardized_cusum(values: pd.Series) -> pd.DataFrame:
    pos_values: list[float] = []
    neg_values: list[float] = []
    pos = 0.0
    neg = 0.0

    for value in values.fillna(0.0):
        pos = max(0.0, pos + value - CUSUM_DRIFT)
        neg = min(0.0, neg + value + CUSUM_DRIFT)
        pos_values.append(pos)
        neg_values.append(neg)

    return pd.DataFrame({"cusum_pos": pos_values, "cusum_neg": neg_values}, index=values.index)


def add_anomaly_features(feature: pd.DataFrame) -> pd.DataFrame:
    frame = feature.copy()
    returns = frame["return_1d"]

    rolling_mean = returns.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
    rolling_std = returns.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std()
    frame["z_score"] = (returns - rolling_mean) / rolling_std

    ewma_volatility = returns.ewm(span=EWMA_SPAN, min_periods=ROLLING_WINDOW, adjust=False).std()
    frame["ewma_value"] = returns.abs() / ewma_volatility

    standardized_return = (returns - rolling_mean) / rolling_std
    cusum = standardized_cusum(standardized_return)
    frame["cusum_pos"] = cusum["cusum_pos"]
    frame["cusum_neg"] = cusum["cusum_neg"]

    z_component = (frame["z_score"].abs() / Z_SCORE_THRESHOLD).clip(upper=1)
    ewma_component = (frame["ewma_value"] / EWMA_THRESHOLD).clip(upper=1)
    cusum_component = (
        np.maximum(frame["cusum_pos"], frame["cusum_neg"].abs()) / CUSUM_THRESHOLD
    ).clip(upper=1)
    frame["anomaly_score"] = pd.concat([z_component, ewma_component, cusum_component], axis=1).max(axis=1)

    frame["is_anomaly"] = (
        (frame["z_score"].abs() >= Z_SCORE_THRESHOLD)
        | (frame["ewma_value"] >= EWMA_THRESHOLD)
        | (frame["cusum_pos"] >= CUSUM_THRESHOLD)
        | (frame["cusum_neg"] <= -CUSUM_THRESHOLD)
    ).astype(int)

    return frame.replace([np.inf, -np.inf], np.nan)
