from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


FEATURE_IMPORTANCE_FILE = Path("data/model_metrics/xgboost_risk_model_latest_feature_importance.csv")
FEATURE_STATS_FILE = Path("data/model_metrics/xgboost_risk_model_latest_feature_stats.json")

FEATURE_DISPLAY_NAMES = {
    "return_1d": "오늘 가격 변화",
    "return_3d": "최근 3일 가격 흐름",
    "return_5d": "최근 5일 가격 흐름",
    "return_10d": "최근 10일 가격 흐름",
    "return_20d": "최근 20일 가격 흐름",
    "gap_open": "장 시작 때 벌어진 가격 차이",
    "intraday_range": "하루 중 가격 흔들림",
    "close_position": "마감 가격 위치",
    "ma5_gap": "최근 평균 가격과의 차이",
    "ma20_gap": "20일 평균 가격과의 차이",
    "ma60_gap": "60일 평균 가격과의 차이",
    "trend_strength_5_20": "단기 흐름과 중기 흐름의 차이",
    "volatility_5d": "최근 5일 가격 흔들림",
    "volatility_10d": "최근 10일 가격 흔들림",
    "volatility_20d": "최근 20일 가격 흔들림",
    "volatility_ratio_5_20": "최근 흔들림이 커진 정도",
    "volume_change_1d": "오늘 거래량 변화",
    "volume_ratio_5d": "최근 평균보다 거래가 몰린 정도",
    "volume_ratio_20d": "평소보다 거래가 몰린 정도",
    "down_days_5d": "최근 하락한 날의 수",
    "down_days_10d": "최근 10일 중 하락한 날의 수",
    "price_volume_shock": "가격과 거래량이 함께 크게 움직인 정도",
    "downside_volume_shock": "하락할 때 거래가 몰린 정도",
    "rsi_14": "최근 가격이 과하게 오른/내린 정도",
    "rsi_change_3d": "과열·침체 신호의 최근 변화",
    "macd": "가격 흐름의 방향 변화",
    "macd_signal": "가격 흐름 변화의 기준선",
    "macd_hist": "가격 흐름 변화의 강도",
    "macd_cross": "가격 흐름이 바뀐 신호",
    "bollinger_width": "가격 움직임의 폭",
    "bollinger_position": "최근 가격 범위 안에서의 위치",
    "drawdown_20d": "최근 고점에서 밀린 정도",
    "z_score": "평소와 다르게 움직인 정도",
    "ewma_value": "최근 움직임이 평소보다 큰 정도",
    "cusum_pos": "상승 움직임이 누적된 정도",
    "cusum_neg": "하락 움직임이 누적된 정도",
    "anomaly_score": "평소와 다른 움직임의 크기",
}

LEVEL_TEXT = {
    "NORMAL": "보통",
    "CAUTION": "주의",
    "DANGER": "위험",
}

LEVEL_SUMMARY = {
    "NORMAL": "주의 기준보다는 낮아 현재는 단기 급락 신호가 강하지 않은 편입니다.",
    "CAUTION": "최근 움직임이 불안정해 단기 급락 가능성을 확인할 필요가 있습니다.",
    "DANGER": "최근 흐름이 과거 급락 직전 모습과 꽤 비슷합니다.",
}


def save_feature_importance(model, feature_columns: list[str], output_path: Path) -> pd.DataFrame:
    gain_scores = model.get_booster().get_score(importance_type="gain")
    rows = []
    for feature in feature_columns:
        rows.append({"feature": feature, "importance": float(gain_scores.get(feature, 0.0))})

    importance = pd.DataFrame(rows)
    total = importance["importance"].sum()
    if total > 0:
        importance["importance"] = importance["importance"] / total
    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
    importance["rank"] = importance.index + 1
    importance.to_csv(output_path, index=False, encoding="utf-8-sig")
    return importance


def save_feature_stats(features: pd.DataFrame, output_path: Path) -> dict[str, dict[str, float]]:
    stats = {}
    for feature in features.columns:
        series = pd.to_numeric(features[feature], errors="coerce")
        stats[feature] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "median": float(series.median()),
        }
    output_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def load_feature_context(
    importance_path: Path = FEATURE_IMPORTANCE_FILE,
    stats_path: Path = FEATURE_STATS_FILE,
) -> dict[str, object]:
    if importance_path.exists():
        importance = pd.read_csv(importance_path, encoding="utf-8-sig").to_dict(orient="records")
    else:
        importance = []

    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        stats = {}

    return {"importance": importance, "stats": stats}


def format_feature_value(feature: str, value: float) -> str:
    if feature.startswith("return_") or "gap" in feature or "volatility" in feature or "drawdown" in feature:
        return f"{value:.2%}"
    if "ratio" in feature or feature in {"ewma_value", "cusum_pos", "cusum_neg", "anomaly_score"}:
        return f"{value:.2f}"
    if feature.startswith("down_days"):
        return f"{value:.0f}일"
    return f"{value:.2f}"


def plain_feature_sentence(feature: str, value: float) -> str:
    if feature in {"volatility_5d", "volatility_10d", "volatility_20d", "volatility_ratio_5_20", "bollinger_width"}:
        return "최근 가격이 평소보다 크게 흔들리고 있습니다."
    if feature in {"volume_change_1d", "volume_ratio_5d", "volume_ratio_20d"}:
        return "거래량이 평소보다 늘어 관심이 몰린 상태입니다."
    if feature in {"down_days_5d", "down_days_10d", "return_3d", "return_5d", "return_10d", "return_20d"}:
        if value < 0:
            return "최근 며칠 동안 가격이 약한 흐름을 보였습니다."
        return "최근 가격 흐름의 변화가 크게 반영되었습니다."
    if feature in {"ma5_gap", "ma20_gap", "ma60_gap", "drawdown_20d"}:
        if value < 0:
            return "최근 평균 가격이나 고점보다 낮은 위치에 있습니다."
        return "최근 평균 가격보다 높은 위치에서 변동이 커졌습니다."
    if feature in {"downside_volume_shock"}:
        return "가격이 내려갈 때 거래가 함께 늘어난 구간이 있습니다."
    if feature in {"price_volume_shock"}:
        return "가격 변화와 거래량 변화가 동시에 크게 나타났습니다."
    if feature in {"rsi_14", "rsi_change_3d"}:
        return "최근 가격 움직임이 한쪽으로 치우친 신호가 있습니다."
    if feature in {"macd", "macd_signal", "macd_hist", "macd_cross", "trend_strength_5_20"}:
        return "짧은 기간의 가격 흐름이 바뀌는 신호가 감지되었습니다."
    if feature in {"z_score", "ewma_value", "cusum_pos", "cusum_neg", "anomaly_score"}:
        return "평소 패턴과 다른 움직임이 나타났습니다."
    if feature in {"gap_open", "intraday_range", "close_position"}:
        return "하루 안의 가격 움직임이 평소와 다르게 나타났습니다."
    return f"{FEATURE_DISPLAY_NAMES.get(feature, feature)} 값이 평소와 다르게 나타났습니다."


def feature_reason(row: pd.Series, feature: str, importance: float, stats: dict[str, dict[str, float]]) -> dict[str, object]:
    value = float(row.get(feature, 0.0))
    feature_stats = stats.get(feature, {})
    mean = float(feature_stats.get("mean", 0.0))
    std = abs(float(feature_stats.get("std", 0.0)))
    deviation = abs((value - mean) / std) if std > 0 else abs(value)
    score = importance * deviation

    name = FEATURE_DISPLAY_NAMES.get(feature, feature)
    description = plain_feature_sentence(feature, value)
    return {
        "name": name,
        "feature": feature,
        "value": value,
        "formatted_value": format_feature_value(feature, value),
        "importance": round(float(importance), 6),
        "deviation": round(float(deviation), 4),
        "score": round(float(score), 6),
        "description": description,
    }


def top_model_factors(
    row: pd.Series,
    context: dict[str, object],
    limit: int = 3,
    candidate_limit: int = 12,
) -> list[dict[str, object]]:
    importance_rows = context.get("importance", [])
    stats = context.get("stats", {})
    factors = []
    for item in importance_rows[:candidate_limit]:
        feature = str(item["feature"])
        if feature not in row or pd.isna(row[feature]):
            continue
        factors.append(feature_reason(row, feature, float(item.get("importance", 0.0)), stats))

    factors = sorted(factors, key=lambda item: item["score"], reverse=True)
    return factors[:limit]


def explain_prediction(row: pd.Series, context: dict[str, object]) -> tuple[str, str]:
    factors = top_model_factors(row, context)
    sentences = []
    for item in factors:
        description = str(item["description"])
        if description not in sentences:
            sentences.append(description)
        if len(sentences) == 3:
            break
    factor_text = " ".join(sentences) or "특별히 두드러진 신호는 많지 않습니다."
    level = LEVEL_TEXT.get(str(row["risk_level"]), str(row["risk_level"]))
    summary = LEVEL_SUMMARY.get(str(row["risk_level"]), "현재 상태를 확인하세요.")
    probability = float(row["risk_probability"])

    message = (
        f"{level} 단계입니다. 단기 급락 가능성은 {probability:.1%}입니다. "
        f"{summary} {factor_text}"
    )
    return message, json.dumps(factors, ensure_ascii=False)
