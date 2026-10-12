from __future__ import annotations

import json
from pathlib import Path


DEFAULT_RISK_THRESHOLD = 0.65
DEFAULT_DANGER_THRESHOLD = 0.8


def select_danger_threshold(
    risk_threshold: float,
    validation_curve: list[dict[str, float]],
) -> float:
    danger_candidates = [
        row
        for row in validation_curve
        if float(row.get("threshold", 0)) > risk_threshold and float(row.get("recall", 0)) > 0
    ]
    if danger_candidates:
        best = max(
            danger_candidates,
            key=lambda row: (
                float(row.get("precision", 0)),
                float(row.get("f1", 0)),
                float(row.get("threshold", 0)),
            ),
        )
        danger_threshold = float(best["threshold"])
    else:
        danger_threshold = max(DEFAULT_DANGER_THRESHOLD, min(risk_threshold + 0.10, 0.95))

    return max(danger_threshold, risk_threshold)


def load_risk_thresholds(
    metrics_path: Path = Path("data/model_metrics/xgboost_risk_model_latest_metrics.json"),
) -> tuple[float, float]:
    if not metrics_path.exists():
        return DEFAULT_RISK_THRESHOLD, DEFAULT_DANGER_THRESHOLD

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold_selection = metrics.get("threshold_selection", {})
    risk_threshold = float(threshold_selection.get("selected_threshold", DEFAULT_RISK_THRESHOLD))
    validation_curve = threshold_selection.get("validation_curve", [])
    return risk_threshold, select_danger_threshold(risk_threshold, validation_curve)


def risk_level(probability: float, risk_threshold: float, danger_threshold: float) -> str:
    if probability >= danger_threshold:
        return "DANGER"
    if probability >= risk_threshold:
        return "CAUTION"
    return "NORMAL"
