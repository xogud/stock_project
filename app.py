from __future__ import annotations

import json
import io
import os
import re
import urllib.parse
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from functools import lru_cache
from datetime import timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    import FinanceDataReader as fdr
except ModuleNotFoundError:
    fdr = None

from anomaly_features import add_anomaly_features
from database.mysql_repository import mysql_enabled, ping as mysql_ping, stock_listing_frame, table_counts, top_risks_frame
from model_explanations import explain_prediction, load_feature_context, top_model_factors
from model_features import FEATURE_COLUMNS
from price_features import add_price_features
from risk_thresholds import load_risk_thresholds, risk_level


MODEL_PATH = Path("models/xgboost_risk_model_latest.joblib")
STOCK_INFO_FILE = Path("data/processed/stock_info.csv")
FEATURE_FILE = Path("data/processed/feature_data.csv")
MARKET_CONTEXT_FILE = Path("data/processed/market_context_daily.csv")
MARKET_INDEX_FILE = Path("data/processed/market_index.csv")
PREDICTION_FILE = Path("data/processed/prediction_result.csv")
BACKTEST_REPORT_FILE = Path("data/processed/daily_backtest/daily_backtest_report_latest.json")
DART_CORP_CODE_FILE = Path("data/processed/dart_corp_codes.csv")
RAW_DIR = Path("data/raw")
FRONTEND_DIST = Path("frontend/dist")
FRONTEND_INDEX = FRONTEND_DIST / "index.html"
START_DATE = "2020-01-01"

LEVEL_TEXT = {
    "NORMAL": "보통",
    "CAUTION": "주의",
    "DANGER": "위험",
}

LEVEL_DESCRIPTIONS = {
    "NORMAL": "최근 급락 위험 신호는 크지 않습니다.",
    "CAUTION": "단기 변동성이 커졌습니다. 신규 진입이나 추가 매수는 신중히 확인하는 구간입니다.",
    "DANGER": "과거 급락 사례와 유사한 패턴입니다. 보유자는 손실 관리 기준을 점검하는 것이 좋습니다.",
}

FEATURE_LABELS = {
    "return_1d": "하루 수익률",
    "return_3d": "3일 수익률",
    "return_5d": "5일 수익률",
    "return_10d": "10일 수익률",
    "return_20d": "20일 수익률",
    "gap_open": "장 시작 가격 차이",
    "intraday_range": "하루 중 가격 흔들림",
    "close_position": "마감 가격 위치",
    "ma5_gap": "5일 평균 가격과의 차이",
    "ma20_gap": "20일 평균 대비 위치",
    "ma60_gap": "60일 평균 가격과의 차이",
    "trend_strength_5_20": "짧은 흐름과 중간 흐름의 차이",
    "volatility_5d": "최근 5일 가격 흔들림",
    "volatility_10d": "최근 10일 가격 흔들림",
    "volatility_20d": "최근 가격 흔들림",
    "volatility_ratio_5_20": "최근 흔들림 증가 정도",
    "volume_change_1d": "오늘 거래량 변화",
    "volume_ratio_5d": "최근 평균 대비 거래량",
    "volume_ratio_20d": "평소 대비 거래량",
    "down_days_5d": "최근 하락일수",
    "down_days_10d": "최근 10일 하락일수",
    "price_volume_shock": "가격과 거래량의 동시 변화",
    "downside_volume_shock": "하락 거래 충격",
    "rsi_14": "과열·침체 정도",
    "rsi_change_3d": "과열·침체 신호 변화",
    "macd": "가격 흐름 방향",
    "macd_signal": "가격 흐름 기준선",
    "macd_hist": "단기 추세 변화",
    "macd_cross": "가격 흐름 전환 신호",
    "bollinger_width": "가격 변동 범위",
    "bollinger_position": "최근 가격 범위 안의 위치",
    "drawdown_20d": "최근 고점 대비 하락폭",
    "z_score": "평소 대비 이탈 정도",
    "ewma_value": "최근 변동 반영 이상 신호",
    "cusum_pos": "상승 누적 변화",
    "cusum_neg": "하락 누적 변화",
    "anomaly_score": "평소와 다른 움직임",
}

app = FastAPI(title="Stock Risk API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):517[0-9]$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: clean_value(value) for key, value in row.items()} for row in frame.to_dict(orient="records")]


def dart_api_key() -> str:
    return os.getenv("DART_API_KEY", "").strip() or os.getenv("OPENDART_API_KEY", "").strip()


def dart_risk_tags(report_name: str) -> list[str]:
    keyword_map = {
        "실적": ["잠정실적", "매출액", "영업손실", "손익구조", "적자"],
        "자금조달": ["유상증자", "전환사채", "신주인수권", "사채권", "증자"],
        "지배구조": ["최대주주", "대표이사", "경영권", "임원"],
        "소송/제재": ["소송", "벌금", "제재", "횡령", "배임"],
        "거래/상장": ["거래정지", "상장폐지", "관리종목", "불성실공시"],
        "계약": ["공급계약", "단일판매", "수주"],
    }
    return [tag for tag, keywords in keyword_map.items() if any(keyword in report_name for keyword in keywords)]


DISCLOSURE_ANALYSIS_KEYWORDS = {
    "실적 악화": ["영업손실", "당기순손실", "적자", "매출액 감소", "손익구조", "실적부진"],
    "자금 조달": ["유상증자", "전환사채", "신주인수권부사채", "사채권", "차입", "담보제공"],
    "지배구조": ["최대주주", "대표이사", "경영권", "임원", "주식변동"],
    "소송/제재": ["소송", "제재", "벌금", "과징금", "횡령", "배임"],
    "거래/상장": ["거래정지", "상장폐지", "관리종목", "불성실공시", "조회공시"],
    "회계/감사": ["감사의견", "한정", "부적정", "의견거절", "계속기업"],
    "계약/수주": ["공급계약", "단일판매", "수주", "계약체결"],
}

DISCLOSURE_RISK_WEIGHTS = {
    "실적 악화": 3,
    "자금 조달": 2,
    "지배구조": 1,
    "소송/제재": 4,
    "거래/상장": 4,
    "회계/감사": 4,
    "계약/수주": 0,
}


def compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


@lru_cache(maxsize=256)
def dart_document_text(receipt_no: str, key: str) -> str:
    params = {"crtfc_key": key, "rcept_no": receipt_no}
    url = "https://opendart.fss.or.kr/api/document.xml?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=20) as response:
        payload = response.read()

    text_parts: list[str] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for xml_name in archive.namelist():
            raw = archive.read(xml_name)
            decoded = None
            for encoding in ("utf-8", "cp949", "euc-kr"):
                try:
                    decoded = raw.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if decoded is None:
                decoded = raw.decode("utf-8", errors="ignore")

            try:
                root = ET.fromstring(decoded)
                text_parts.append(" ".join(root.itertext()))
            except ET.ParseError:
                text_parts.append(re.sub(r"<[^>]+>", " ", decoded))

    return compact_text(" ".join(text_parts))


def disclosure_snippet(text: str, keyword: str, radius: int = 55) -> str | None:
    index = text.find(keyword)
    if index < 0:
        return None
    start = max(index - radius, 0)
    end = min(index + len(keyword) + radius, len(text))
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


def analyze_disclosure_content(report_name: str, receipt_no: str, key: str) -> dict[str, Any]:
    base_text = compact_text(report_name)
    try:
        document_text = dart_document_text(receipt_no, key)
    except Exception as error:
        title_tags = dart_risk_tags(report_name)
        return {
            "available": False,
            "summary": f"본문을 가져오지 못했습니다: {error}",
            "risk_score": sum(DISCLOSURE_RISK_WEIGHTS.get(tag, 0) for tag in title_tags),
            "matched_categories": title_tags,
            "snippets": [],
        }

    combined = f"{base_text} {document_text[:80000]}"
    matched_categories = []
    snippets = []
    risk_score = 0

    for category, keywords in DISCLOSURE_ANALYSIS_KEYWORDS.items():
        matched_keywords = [keyword for keyword in keywords if keyword in combined]
        if not matched_keywords:
            continue
        matched_categories.append(category)
        risk_score += DISCLOSURE_RISK_WEIGHTS.get(category, 0)
        snippet = disclosure_snippet(combined, matched_keywords[0])
        if snippet:
            snippets.append({"category": category, "keyword": matched_keywords[0], "text": snippet})

    if risk_score >= 6:
        summary = "본문에서 여러 위험 키워드가 확인됩니다. 공시 원문 확인이 필요합니다."
    elif risk_score >= 3:
        summary = "본문에서 주의할 만한 공시 키워드가 확인됩니다."
    elif matched_categories:
        summary = "본문에서 참고할 만한 공시 키워드가 확인됩니다."
    else:
        summary = "본문 기준으로 뚜렷한 위험 키워드는 확인되지 않았습니다."

    return {
        "available": True,
        "summary": summary,
        "risk_score": risk_score,
        "matched_categories": matched_categories,
        "snippets": snippets[:3],
        "text_length": len(document_text),
    }


@lru_cache(maxsize=1)
def dart_corp_codes() -> pd.DataFrame:
    if DART_CORP_CODE_FILE.exists():
        return pd.read_csv(DART_CORP_CODE_FILE, dtype={"corp_code": str, "stock_code": str}, encoding="utf-8-sig")

    key = dart_api_key()
    if not key:
        return pd.DataFrame(columns=["corp_code", "corp_name", "stock_code", "modify_date"])

    url = "https://opendart.fss.or.kr/api/corpCode.xml?" + urllib.parse.urlencode({"crtfc_key": key})
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            payload = response.read()
    except Exception:
        return pd.DataFrame(columns=["corp_code", "corp_name", "stock_code", "modify_date"])

    import io

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        xml_name = archive.namelist()[0]
        xml_bytes = archive.read(xml_name)

    root = ET.fromstring(xml_bytes)
    rows = []
    for item in root.findall("list"):
        stock_code = (item.findtext("stock_code") or "").strip()
        if not stock_code:
            continue
        rows.append(
            {
                "corp_code": (item.findtext("corp_code") or "").strip(),
                "corp_name": (item.findtext("corp_name") or "").strip(),
                "stock_code": stock_code.zfill(6),
                "modify_date": (item.findtext("modify_date") or "").strip(),
            }
        )

    frame = pd.DataFrame(rows)
    DART_CORP_CODE_FILE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DART_CORP_CODE_FILE, index=False, encoding="utf-8-sig")
    return frame


def summarize_disclosures(items: list[dict[str, Any]], risk_probability: float | None = None) -> dict[str, Any]:
    total_count = len(items)
    analyzed_count = sum(1 for item in items if item.get("analysis"))
    risk_scores = [
        int(item.get("analysis", {}).get("risk_score") or 0)
        for item in items
        if item.get("analysis")
    ]
    total_risk_score = sum(risk_scores)
    caution_count = sum(1 for score in risk_scores if score >= 3)
    high_risk_count = sum(1 for score in risk_scores if score >= 6)

    keyword_counts: dict[str, int] = {}
    for item in items:
        for tag in item.get("tags") or []:
            keyword_counts[str(tag)] = keyword_counts.get(str(tag), 0) + 1

    top_keywords = [
        keyword
        for keyword, _ in sorted(keyword_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:5]
    ]

    if high_risk_count > 0 or total_risk_score >= 8:
        level = "높음"
    elif caution_count > 0 or total_risk_score >= 3:
        level = "주의"
    else:
        level = "낮음"

    stock_risk_high = risk_probability is not None and float(risk_probability) >= 0.86
    stock_risk_danger = risk_probability is not None and float(risk_probability) >= 0.93

    if stock_risk_danger and level in {"주의", "높음"}:
        interpretation = "주가 흐름과 최근 공시에서 모두 주의 신호가 확인됩니다. 공시 원문과 가격 변동을 함께 확인해야 합니다."
    elif stock_risk_high and level == "낮음":
        interpretation = "주가 위험 신호는 높지만, 최근 공시 본문에서는 뚜렷한 위험 키워드가 강하게 확인되지는 않았습니다."
    elif not stock_risk_high and level in {"주의", "높음"}:
        interpretation = "주가 위험 신호는 제한적이지만, 최근 공시에 주의할 만한 내용이 있어 확인이 필요합니다."
    elif total_count == 0:
        interpretation = "최근 조회 기간에 확인된 공시가 없습니다."
    else:
        interpretation = "최근 공시 기준으로 강한 위험 키워드는 제한적입니다."

    return {
        "total_count": total_count,
        "analyzed_count": analyzed_count,
        "caution_count": caution_count,
        "high_risk_count": high_risk_count,
        "risk_score": total_risk_score,
        "risk_level": level,
        "top_keywords": top_keywords,
        "interpretation": interpretation,
    }


def recent_disclosures(
    stock_code: str,
    base_date: Any,
    days: int = 30,
    limit: int = 8,
    risk_probability: float | None = None,
) -> dict[str, Any]:
    key = dart_api_key()
    if not key:
        return {
            "available": False,
            "reason": "DART_API_KEY 환경변수가 설정되지 않았습니다.",
            "items": [],
        }

    try:
        corp_codes = dart_corp_codes()
    except Exception as error:
        return {"available": False, "reason": f"DART 회사코드 조회 실패: {error}", "items": []}

    if corp_codes.empty:
        return {"available": False, "reason": "DART 회사코드 데이터를 가져오지 못했습니다.", "items": []}
    normalized_code = str(stock_code).zfill(6)
    matched = corp_codes[corp_codes["stock_code"].astype(str).str.zfill(6) == normalized_code]
    if matched.empty:
        return {"available": False, "reason": "DART 회사코드를 찾지 못했습니다.", "items": []}

    corp_code = str(matched.iloc[0]["corp_code"])
    end_date = pd.Timestamp(base_date).date()
    begin_date = end_date - timedelta(days=days)
    params = {
        "crtfc_key": key,
        "corp_code": corp_code,
        "bgn_de": begin_date.strftime("%Y%m%d"),
        "end_de": end_date.strftime("%Y%m%d"),
        "page_no": "1",
        "page_count": str(limit),
        "sort": "date",
        "sort_mth": "desc",
    }
    url = "https://opendart.fss.or.kr/api/list.json?" + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(url, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        return {"available": False, "reason": f"DART 조회 실패: {error}", "items": []}

    status = str(payload.get("status", ""))
    if status not in {"000", "013"}:
        return {"available": False, "reason": payload.get("message") or "DART 응답 오류", "items": []}

    items = []
    for index, item in enumerate(payload.get("list", [])[:limit]):
        report_name = str(item.get("report_nm", ""))
        receipt_no = str(item.get("rcept_no", ""))
        tags = dart_risk_tags(report_name)
        analysis = analyze_disclosure_content(report_name, receipt_no, key) if receipt_no and index < 3 else None
        if analysis and analysis.get("matched_categories"):
            tags = sorted(set(tags) | set(analysis["matched_categories"]))
        items.append(
            {
                "date": item.get("rcept_dt"),
                "corp_name": item.get("corp_name"),
                "report_name": report_name,
                "submitter": item.get("flr_nm"),
                "receipt_no": receipt_no,
                "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt_no}" if receipt_no else None,
                "tags": tags,
                "analysis": analysis,
            }
        )

    return {
        "available": True,
        "corp_code": corp_code,
        "corp_name": clean_value(matched.iloc[0].get("corp_name")),
        "period": {"from": begin_date.isoformat(), "to": end_date.isoformat()},
        "summary": summarize_disclosures(items, risk_probability),
        "items": items,
    }


def feature_label(feature_name: str) -> str:
    return FEATURE_LABELS.get(feature_name, feature_name)


def format_feature_value(feature_name: str, value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    if feature_name in {
        "return_1d",
        "return_3d",
        "return_5d",
        "return_10d",
        "return_20d",
        "ma20_gap",
        "volatility_20d",
        "bollinger_width",
        "drawdown_20d",
    }:
        return f"{value:.2%}"
    if feature_name == "volume_ratio_20d":
        return f"{value:.2f}배"
    if feature_name == "down_days_5d":
        return f"{value:.0f}일"
    return f"{value:.2f}"


def market_index_context(date_value: Any, stock_return: float | None = None) -> dict[str, Any]:
    target_date = pd.Timestamp(date_value).date()
    if not MARKET_INDEX_FILE.exists():
        return {"available": False}

    index_frame = pd.read_csv(MARKET_INDEX_FILE, encoding="utf-8-sig")
    if index_frame.empty:
        return {"available": False}

    index_frame["date"] = pd.to_datetime(index_frame["date"], errors="coerce")
    index_frame = index_frame.dropna(subset=["date"]).sort_values("date")
    matched = index_frame[index_frame["date"].dt.date <= target_date].tail(1)
    if matched.empty:
        return {"available": False}

    row = matched.iloc[0]
    index_return_1d = clean_value(row.get("return_1d"))
    relative_return = None
    if stock_return is not None and index_return_1d is not None:
        relative_return = float(stock_return) - float(index_return_1d)
    return {
        "available": True,
        "date": clean_value(row["date"]),
        "index_name": clean_value(row.get("index_name")) or "종합주가지수",
        "close": clean_value(row.get("close")),
        "return_1d": index_return_1d,
        "return_5d": clean_value(row.get("return_5d")),
        "return_20d": clean_value(row.get("return_20d")),
        "volatility_20d": clean_value(row.get("volatility_20d")),
        "relative_return_1d": relative_return,
        "source": clean_value(row.get("source")),
    }


def market_context_for_date(date_value: Any, stock_code: str | None = None) -> dict[str, Any]:
    context_file = MARKET_CONTEXT_FILE if MARKET_CONTEXT_FILE.exists() else FEATURE_FILE
    if not context_file.exists():
        return {"available": False}

    target_date = pd.Timestamp(date_value).date().isoformat()
    features = pd.read_csv(
        context_file,
        dtype={"stock_code": str},
        usecols=["date", "stock_code", "sector", "return_1d"],
        encoding="utf-8-sig",
    )
    features["date"] = pd.to_datetime(features["date"], errors="coerce").dt.date.astype(str)
    day = features[features["date"] == target_date].copy()
    day["return_1d"] = pd.to_numeric(day["return_1d"], errors="coerce")
    day["sector"] = day["sector"].fillna("기타").astype(str)
    day = day.dropna(subset=["return_1d"])
    if day.empty:
        return {"available": False, "date": target_date}

    total_count = int(len(day))
    down_count = int((day["return_1d"] < 0).sum())
    sharp_drop_count = int((day["return_1d"] <= -0.03).sum())
    severe_drop_count = int((day["return_1d"] <= -0.06).sum())
    down_ratio = down_count / total_count if total_count else 0
    sharp_drop_ratio = sharp_drop_count / total_count if total_count else 0
    severe_drop_ratio = severe_drop_count / total_count if total_count else 0

    risk_signal_count = None
    risk_signal_ratio = None
    prediction_day = pd.DataFrame()
    if PREDICTION_FILE.exists():
        predictions = pd.read_csv(
            PREDICTION_FILE,
            dtype={"stock_code": str},
            usecols=["trade_date", "stock_code", "risk_level"],
            encoding="utf-8-sig",
        )
        predictions["trade_date"] = pd.to_datetime(predictions["trade_date"], errors="coerce").dt.date.astype(str)
        prediction_day = predictions[predictions["trade_date"] == target_date]
        if not prediction_day.empty:
            risk_signal_count = int(prediction_day["risk_level"].isin(["CAUTION", "DANGER"]).sum())
            risk_signal_ratio = risk_signal_count / len(prediction_day)

    normalized_code = str(stock_code).zfill(6) if stock_code else None
    stock_return = None
    stock_sector = None
    if normalized_code:
        matched = day[day["stock_code"].astype(str).str.zfill(6) == normalized_code]
        if not matched.empty:
            stock_return = float(matched.iloc[0]["return_1d"])
            stock_sector = str(matched.iloc[0]["sector"])

    sector_group = day.groupby("sector")["return_1d"].agg(
        total_count="count",
        mean_return="mean",
        median_return="median",
        down_count=lambda values: int((values < 0).sum()),
        sharp_drop_count=lambda values: int((values <= -0.03).sum()),
        severe_drop_count=lambda values: int((values <= -0.06).sum()),
    )
    sector_group["down_ratio"] = sector_group["down_count"] / sector_group["total_count"]
    sector_group["sharp_drop_ratio"] = sector_group["sharp_drop_count"] / sector_group["total_count"]
    sector_group["severe_drop_ratio"] = sector_group["severe_drop_count"] / sector_group["total_count"]

    sector_risk_counts = pd.Series(dtype="int64")
    if not prediction_day.empty:
        risk_codes = prediction_day[prediction_day["risk_level"].isin(["CAUTION", "DANGER"])]["stock_code"].astype(str).str.zfill(6)
        risk_day = day[day["stock_code"].astype(str).str.zfill(6).isin(set(risk_codes))]
        if not risk_day.empty:
            sector_risk_counts = risk_day.groupby("sector").size()

    def build_sector_context(sector_name: str | None) -> dict[str, Any] | None:
        if not sector_name or sector_name not in sector_group.index:
            return None
        row = sector_group.loc[sector_name]
        total = int(row["total_count"])
        sector_risk_count = int(sector_risk_counts.get(sector_name, 0)) if not sector_risk_counts.empty else None
        return {
            "name": sector_name,
            "total_count": total,
            "down_count": int(row["down_count"]),
            "down_ratio": float(row["down_ratio"]),
            "sharp_drop_count": int(row["sharp_drop_count"]),
            "sharp_drop_ratio": float(row["sharp_drop_ratio"]),
            "severe_drop_count": int(row["severe_drop_count"]),
            "severe_drop_ratio": float(row["severe_drop_ratio"]),
            "mean_return": float(row["mean_return"]),
            "median_return": float(row["median_return"]),
            "risk_signal_count": sector_risk_count,
            "risk_signal_ratio": sector_risk_count / total if sector_risk_count is not None and total else None,
        }

    top_sector_rows = sector_group[sector_group["total_count"] >= 3].copy()
    top_sector_context = None
    if not top_sector_rows.empty:
        top_sector_rows = top_sector_rows.sort_values(
            ["sharp_drop_ratio", "down_ratio", "mean_return"],
            ascending=[False, False, True],
        )
        top_sector_context = build_sector_context(str(top_sector_rows.index[0]))

    focused_sector_context = build_sector_context(stock_sector) if stock_sector else top_sector_context
    if focused_sector_context:
        focused_sector_context["is_stock_sector"] = bool(stock_sector)

    index_context = market_index_context(target_date, stock_return)
    index_return_1d = index_context.get("return_1d") if index_context.get("available") else None
    index_return_5d = index_context.get("return_5d") if index_context.get("available") else None

    external_likely = (
        down_ratio >= 0.65
        or sharp_drop_ratio >= 0.20
        or severe_drop_ratio >= 0.08
        or (risk_signal_ratio is not None and risk_signal_ratio >= 0.25)
        or (index_return_1d is not None and float(index_return_1d) <= -0.015)
        or (index_return_5d is not None and float(index_return_5d) <= -0.03)
    )
    individual_likely = (
        stock_return is not None
        and stock_return <= -0.03
        and down_ratio < 0.45
        and sharp_drop_ratio < 0.08
    )
    sector_likely = (
        focused_sector_context is not None
        and bool(focused_sector_context.get("is_stock_sector"))
        and stock_return is not None
        and stock_return <= -0.03
        and down_ratio < 0.65
        and float(focused_sector_context["down_ratio"]) >= 0.55
        and (
            float(focused_sector_context["sharp_drop_ratio"]) >= 0.10
            or float(focused_sector_context["mean_return"]) <= -0.015
        )
    )

    if external_likely:
        label = "외부 요인 가능성 높음"
        message = "같은 날 여러 종목 또는 종합주가지수에서 약세 신호가 함께 나타났습니다. 개별 종목보다 시장 전체나 외부 요인의 영향일 수 있습니다."
    elif individual_likely:
        label = "개별 종목 요인 가능성"
        message = "시장 전체의 동반 하락은 크지 않은데 해당 종목의 하락이 두드러집니다. 개별 종목 이슈 가능성을 함께 확인해야 합니다."
    else:
        label = "동반 변동 제한적"
        message = "같은 날 시장 전체가 크게 흔들린 신호는 제한적입니다. 개별 종목 지표와 최근 흐름을 중심으로 확인하는 구간입니다."

    if sector_likely and not external_likely:
        label = "섹터 요인 가능성"
        sector_name = focused_sector_context.get("name", "같은 섹터") if focused_sector_context else "같은 섹터"
        message = f"{sector_name} 안에서 여러 종목이 같이 하락했습니다. 개별 종목 문제만 보기보다 같은 업종의 수급, 정책, 실적 이슈를 함께 확인해야 합니다."

    return {
        "available": True,
        "date": target_date,
        "label": label,
        "message": message,
        "total_count": total_count,
        "down_count": down_count,
        "down_ratio": down_ratio,
        "sharp_drop_count": sharp_drop_count,
        "sharp_drop_ratio": sharp_drop_ratio,
        "severe_drop_count": severe_drop_count,
        "severe_drop_ratio": severe_drop_ratio,
        "risk_signal_count": risk_signal_count,
        "risk_signal_ratio": risk_signal_ratio,
        "market_mean_return": float(day["return_1d"].mean()),
        "market_median_return": float(day["return_1d"].median()),
        "stock_return": stock_return,
        "sector": focused_sector_context,
        "top_weak_sector": top_sector_context,
        "index": index_context,
    }


@lru_cache(maxsize=1)
def model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file does not exist: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def feature_context() -> dict[str, object]:
    return load_feature_context()


@lru_cache(maxsize=1)
def thresholds() -> tuple[float, float]:
    return load_risk_thresholds()


@lru_cache(maxsize=1)
def stock_listing() -> pd.DataFrame:
    if mysql_enabled():
        try:
            listing = stock_listing_frame()
            if not listing.empty:
                listing["Code"] = listing["Code"].astype(str).str.zfill(6)
                listing["Name"] = listing["Name"].astype(str)
                listing["Market"] = listing["Market"].astype(str)
                listing["Sector"] = listing.get("Sector", pd.Series("기타", index=listing.index)).fillna("기타").astype(str)
                listing["Ticker"] = listing["Ticker"].astype(str)
                return listing.sort_values(["Market", "Name"]).reset_index(drop=True)
        except Exception:
            pass

    if not STOCK_INFO_FILE.exists():
        raise RuntimeError(f"Stock info file does not exist: {STOCK_INFO_FILE}")

    local = pd.read_csv(STOCK_INFO_FILE, dtype={"code": str}, encoding="utf-8-sig")
    local = local.rename(columns={"code": "Code", "name": "Name", "market": "Market", "sector": "Sector"})
    if "Sector" not in local.columns:
        local["Sector"] = "기타"
    local = local[["Code", "Name", "Market", "Sector"]].copy()

    try:
        krx = fdr.StockListing("KRX")[["Code", "Name", "Market"]].copy() if fdr else pd.DataFrame()
    except Exception:
        krx = pd.DataFrame(columns=["Code", "Name", "Market"])
    if not krx.empty:
        krx["Sector"] = "기타"

    listing = pd.concat([local, krx], ignore_index=True)
    listing["Code"] = listing["Code"].astype(str).str.zfill(6)
    listing["Name"] = listing["Name"].astype(str)
    listing["Market"] = listing["Market"].astype(str)
    listing["Sector"] = listing["Sector"].fillna("기타").astype(str)
    listing = listing.drop_duplicates(subset=["Code"], keep="first")
    listing = listing[listing["Market"].isin(["KOSPI", "KOSDAQ"])]
    listing["Ticker"] = listing["Code"] + "." + listing["Market"].map({"KOSPI": "KS", "KOSDAQ": "KQ"})
    return listing.sort_values(["Market", "Name"]).reset_index(drop=True)


def normalize_price_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = frame.columns.get_level_values(0)

    normalized_columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    if "date" not in normalized_columns:
        frame = frame.reset_index()
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]

    if "adj_close" not in frame.columns:
        if "close" not in frame.columns:
            return pd.DataFrame()
        frame["adj_close"] = frame["close"]
    if "date" not in frame.columns:
        return pd.DataFrame()

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "adj_close", "volume"]:
        if column not in frame.columns:
            return pd.DataFrame()
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["date", "open", "high", "low", "close", "adj_close", "volume"])
    frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    frame["ticker"] = ticker
    return frame.reset_index(drop=True)


def load_cached_price(ticker: str) -> pd.DataFrame:
    code = ticker.split(".", maxsplit=1)[0]
    path = RAW_DIR / f"{code}_{ticker}.csv"
    if not path.exists():
        return pd.DataFrame()

    frame = pd.read_csv(path, encoding="utf-8-sig")
    return normalize_price_frame(frame, ticker)


def download_price(ticker: str) -> pd.DataFrame:
    yf.set_tz_cache_location(str(Path(".cache/yfinance").resolve()))
    try:
        frame = yf.download(
            ticker,
            start=START_DATE,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return load_cached_price(ticker)

    normalized = normalize_price_frame(frame, ticker)
    if normalized.empty:
        return load_cached_price(ticker)
    return normalized


def build_prediction_frame(ticker: str) -> pd.DataFrame:
    price = download_price(ticker)
    if price.empty:
        return pd.DataFrame()

    data = add_anomaly_features(add_price_features(price)).dropna(subset=FEATURE_COLUMNS).copy()
    if data.empty:
        return pd.DataFrame()

    risk_threshold, danger_threshold = thresholds()
    data["risk_probability"] = model().predict_proba(data[FEATURE_COLUMNS])[:, 1]
    data["risk_level"] = data["risk_probability"].map(
        lambda probability: risk_level(float(probability), risk_threshold, danger_threshold)
    )
    data["risk_level_ko"] = data["risk_level"].map(LEVEL_TEXT)
    data["description"] = data["risk_level"].map(LEVEL_DESCRIPTIONS)
    data["explanation"] = data.apply(lambda row: explain_prediction(row, feature_context())[0], axis=1)
    return data.sort_values("date").reset_index(drop=True)


def selected_reasons(row: pd.Series) -> list[dict[str, Any]]:
    model_factors = top_model_factors(row, feature_context(), limit=5)
    if model_factors:
        return [
            {
                "label": feature_label(str(item["feature"])),
                "feature": str(item["feature"]),
                "value": clean_value(item.get("formatted_value")) or format_feature_value(
                    str(item["feature"]),
                    float(item["value"]),
                ),
                "summary": clean_value(item.get("description")),
                "score": clean_value(item.get("score")),
            }
            for item in model_factors
        ]

    fallback_features = ["volatility_20d", "volume_ratio_20d", "drawdown_20d", "down_days_5d", "anomaly_score"]
    return [
        {
            "label": feature_label(feature),
            "feature": feature,
            "value": format_feature_value(feature, clean_value(row.get(feature))),
            "summary": "현재 판단에 참고한 주요 움직임입니다.",
            "score": None,
        }
        for feature in fallback_features
    ]


@app.get("/api/health")
def health() -> dict[str, Any]:
    risk_threshold, danger_threshold = thresholds()
    database = mysql_ping() if mysql_enabled() else {"available": False, "enabled": False, "error": None}
    return {
        "status": "ok",
        "model_exists": MODEL_PATH.exists(),
        "stock_info_exists": STOCK_INFO_FILE.exists(),
        "risk_threshold": risk_threshold,
        "danger_threshold": danger_threshold,
        "database": database,
    }


@app.get("/api/database/status")
def database_status() -> dict[str, Any]:
    status = mysql_ping()
    status["enabled"] = mysql_enabled()
    status["table_counts"] = table_counts() if status["available"] else {}
    return status


@app.get("/api/stocks")
def search_stocks(q: str = Query("", max_length=50), limit: int = Query(20, ge=1, le=50)) -> dict[str, Any]:
    listing = stock_listing()
    query = q.strip()
    if not query:
        matched = listing.head(limit)
    elif query.isdigit():
        matched = listing[listing["Code"].str.startswith(query)].head(limit)
    else:
        exact = listing[listing["Name"] == query]
        partial = listing[listing["Name"].str.contains(query, case=False, regex=False, na=False)]
        matched = pd.concat([exact, partial], ignore_index=True).drop_duplicates(subset=["Code"]).head(limit)

    matched = matched.rename(
        columns={"Code": "code", "Name": "name", "Market": "market", "Sector": "sector", "Ticker": "ticker"}
    )
    return {"items": records(matched)}


@app.get("/api/predict/{ticker}")
def predict(ticker: str, days: int = Query(365, ge=30, le=1800)) -> dict[str, Any]:
    listing = stock_listing()
    stock = listing[listing["Ticker"] == ticker]
    if stock.empty:
        raise HTTPException(status_code=404, detail="Unknown ticker")

    result = build_prediction_frame(ticker)
    if result.empty or "date" not in result.columns:
        raise HTTPException(status_code=422, detail="No analyzable price data")

    selected = result.tail(days).copy()
    latest = selected.tail(1).iloc[0]
    risk_threshold, danger_threshold = thresholds()

    series_columns = ["date", "close", "volume", "risk_probability", "risk_level", "risk_level_ko"]
    recent_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "risk_probability",
        "risk_level",
        "risk_level_ko",
    ]
    return {
        "stock": {
            "code": str(stock.iloc[0]["Code"]),
            "name": str(stock.iloc[0]["Name"]),
            "market": str(stock.iloc[0]["Market"]),
            "sector": str(stock.iloc[0].get("Sector", "기타")),
            "ticker": ticker,
        },
        "thresholds": {
            "caution": risk_threshold,
            "danger": danger_threshold,
        },
        "latest": {
            "date": clean_value(latest["date"]),
            "close": clean_value(latest["close"]),
            "volume": clean_value(latest["volume"]),
            "return_1d": clean_value(latest["return_1d"]),
            "risk_probability": clean_value(latest["risk_probability"]),
            "risk_level": clean_value(latest["risk_level"]),
            "risk_level_ko": clean_value(latest["risk_level_ko"]),
            "description": clean_value(latest["description"]),
            "explanation": clean_value(latest["explanation"]),
            "reasons": selected_reasons(latest),
            "market_context": market_context_for_date(latest["date"], str(stock.iloc[0]["Code"])),
            "disclosures": recent_disclosures(
                str(stock.iloc[0]["Code"]),
                latest["date"],
                risk_probability=clean_value(latest["risk_probability"]),
            ),
        },
        "series": records(selected[series_columns]),
        "recent": records(selected.sort_values("date", ascending=False).head(15)[recent_columns]),
    }


@app.get("/api/backtest-summary")
def backtest_summary() -> dict[str, Any]:
    if not BACKTEST_REPORT_FILE.exists():
        return {"available": False}
    return {"available": True, **json.loads(BACKTEST_REPORT_FILE.read_text(encoding="utf-8"))}


def top_risks_from_csv(limit: int) -> dict[str, Any]:
    if not PREDICTION_FILE.exists():
        return {"items": [], "latest_date": None, "source": "csv"}

    predictions = pd.read_csv(PREDICTION_FILE, dtype={"stock_code": str}, encoding="utf-8-sig")
    if predictions.empty:
        return {"items": [], "latest_date": None, "source": "csv"}

    predictions["trade_date"] = pd.to_datetime(predictions["trade_date"])
    latest_date = predictions["trade_date"].max()
    latest = predictions[predictions["trade_date"] == latest_date].copy()

    listing = stock_listing().rename(
        columns={"Code": "stock_code", "Name": "stock_name", "Market": "market", "Sector": "sector"}
    )
    latest = latest.merge(listing[["stock_code", "stock_name", "market", "sector", "Ticker"]], on="stock_code", how="left")
    latest["risk_level_ko"] = latest["risk_level"].map(LEVEL_TEXT)
    latest = latest.sort_values("risk_probability", ascending=False).head(limit)
    latest = latest.rename(columns={"trade_date": "date", "Ticker": "ticker"})
    return {
        "latest_date": latest_date.date().isoformat(),
        "source": "csv",
        "market_context": market_context_for_date(latest_date),
        "items": records(
            latest[
                [
                    "date",
                    "stock_code",
                    "stock_name",
                    "market",
                    "sector",
                    "ticker",
                    "risk_probability",
                    "risk_level",
                    "risk_level_ko",
                ]
            ]
        ),
    }


@app.get("/api/top-risks")
def top_risks(limit: int = Query(10, ge=1, le=50)) -> dict[str, Any]:
    csv_payload = top_risks_from_csv(limit)
    csv_latest = csv_payload.get("latest_date")

    if mysql_enabled():
        try:
            latest_date, latest = top_risks_frame(limit)
            mysql_is_fresher = (
                latest_date
                and (not csv_latest or pd.Timestamp(latest_date) >= pd.Timestamp(str(csv_latest)))
            )
            if mysql_is_fresher and not latest.empty:
                latest["risk_level_ko"] = latest["risk_level"].map(LEVEL_TEXT)
                return {
                    "latest_date": latest_date,
                    "source": "mysql",
                    "market_context": market_context_for_date(latest_date),
                    "items": records(latest),
                }
        except Exception:
            pass

    return csv_payload


if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/", include_in_schema=False)
    def serve_frontend_index() -> FileResponse:
        return FileResponse(FRONTEND_INDEX)

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_frontend(full_path: str) -> FileResponse:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")

        target = FRONTEND_DIST / full_path
        if target.is_file():
            return FileResponse(target)
        return FileResponse(FRONTEND_INDEX)


