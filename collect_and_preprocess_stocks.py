from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import yfinance as yf

from price_features import add_price_features


START_DATE = "2020-01-01"
DEFAULT_UNIVERSE = "krx"
DEFAULT_MAX_STOCKS = 300
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MYSQL_IMPORT_DIR = Path("data/mysql_import_ready")
CACHE_DIR = Path(".cache/yfinance")


@dataclass(frozen=True)
class Stock:
    no: int
    name: str
    code: str
    ticker: str
    market: str
    sector: str


STOCKS: tuple[Stock, ...] = (
    Stock(1, "삼성전자", "005930", "005930.KS", "KOSPI", "반도체"),
    Stock(2, "SK하이닉스", "000660", "000660.KS", "KOSPI", "반도체"),
    Stock(3, "LG에너지솔루션", "373220", "373220.KS", "KOSPI", "2차전지"),
    Stock(4, "삼성바이오로직스", "207940", "207940.KS", "KOSPI", "바이오"),
    Stock(5, "현대차", "005380", "005380.KS", "KOSPI", "자동차"),
    Stock(6, "기아", "000270", "000270.KS", "KOSPI", "자동차"),
    Stock(7, "셀트리온", "068270", "068270.KS", "KOSPI", "바이오"),
    Stock(8, "NAVER", "035420", "035420.KS", "KOSPI", "플랫폼"),
    Stock(9, "카카오", "035720", "035720.KS", "KOSPI", "플랫폼"),
    Stock(10, "KB금융", "105560", "105560.KS", "KOSPI", "금융"),
    Stock(11, "신한지주", "055550", "055550.KS", "KOSPI", "금융"),
    Stock(12, "하나금융지주", "086790", "086790.KS", "KOSPI", "금융"),
    Stock(13, "우리금융지주", "316140", "316140.KS", "KOSPI", "금융"),
    Stock(14, "삼성물산", "028260", "028260.KS", "KOSPI", "지주/건설"),
    Stock(15, "삼성생명", "032830", "032830.KS", "KOSPI", "보험"),
    Stock(16, "삼성화재", "000810", "000810.KS", "KOSPI", "보험"),
    Stock(17, "LG화학", "051910", "051910.KS", "KOSPI", "화학/2차전지"),
    Stock(18, "삼성SDI", "006400", "006400.KS", "KOSPI", "2차전지"),
    Stock(19, "POSCO홀딩스", "005490", "005490.KS", "KOSPI", "철강"),
    Stock(20, "포스코퓨처엠", "003670", "003670.KS", "KOSPI", "2차전지 소재"),
    Stock(21, "현대모비스", "012330", "012330.KS", "KOSPI", "자동차부품"),
    Stock(22, "현대글로비스", "086280", "086280.KS", "KOSPI", "물류"),
    Stock(23, "LG전자", "066570", "066570.KS", "KOSPI", "전자"),
    Stock(24, "삼성전기", "009150", "009150.KS", "KOSPI", "전자부품"),
    Stock(25, "LG이노텍", "011070", "011070.KS", "KOSPI", "전자부품"),
    Stock(26, "SK텔레콤", "017670", "017670.KS", "KOSPI", "통신"),
    Stock(27, "KT", "030200", "030200.KS", "KOSPI", "통신"),
    Stock(28, "KT&G", "033780", "033780.KS", "KOSPI", "소비재"),
    Stock(29, "SK이노베이션", "096770", "096770.KS", "KOSPI", "에너지"),
    Stock(30, "S-Oil", "010950", "010950.KS", "KOSPI", "정유"),
    Stock(31, "두산에너빌리티", "034020", "034020.KS", "KOSPI", "에너지/기계"),
    Stock(32, "HD현대중공업", "329180", "329180.KS", "KOSPI", "조선"),
    Stock(33, "HD한국조선해양", "009540", "009540.KS", "KOSPI", "조선"),
    Stock(34, "한화오션", "042660", "042660.KS", "KOSPI", "조선"),
    Stock(35, "한화에어로스페이스", "012450", "012450.KS", "KOSPI", "방산"),
    Stock(36, "한국항공우주", "047810", "047810.KS", "KOSPI", "방산/항공"),
    Stock(37, "유한양행", "000100", "000100.KS", "KOSPI", "제약"),
    Stock(38, "삼성에스디에스", "018260", "018260.KS", "KOSPI", "IT서비스"),
    Stock(39, "HMM", "011200", "011200.KS", "KOSPI", "해운"),
    Stock(40, "고려아연", "010130", "010130.KS", "KOSPI", "비철금속"),
    Stock(41, "에코프로비엠", "247540", "247540.KQ", "KOSDAQ", "2차전지 소재"),
    Stock(42, "에코프로", "086520", "086520.KQ", "KOSDAQ", "2차전지 지주"),
    Stock(43, "알테오젠", "196170", "196170.KQ", "KOSDAQ", "바이오"),
    Stock(44, "HLB", "028300", "028300.KQ", "KOSDAQ", "바이오"),
    Stock(45, "레인보우로보틱스", "277810", "277810.KQ", "KOSDAQ", "로봇"),
    Stock(46, "리노공업", "058470", "058470.KQ", "KOSDAQ", "반도체 장비"),
    Stock(47, "삼천당제약", "000250", "000250.KQ", "KOSDAQ", "제약"),
    Stock(48, "JYP Ent.", "035900", "035900.KQ", "KOSDAQ", "엔터"),
    Stock(49, "에스엠", "041510", "041510.KQ", "KOSDAQ", "엔터"),
    Stock(50, "HPSP", "403870", "403870.KQ", "KOSDAQ", "반도체 장비"),
)


def fallback_stocks(limit: int | None = None) -> list[Stock]:
    stocks = list(STOCKS)
    return stocks[:limit] if limit else stocks


def load_mysql_cached_stocks(max_stocks: int) -> list[Stock]:
    try:
        from database.mysql_repository import mysql_enabled, query_frame

        if not mysql_enabled():
            return []
        listing = query_frame(
            """
            SELECT
                stock_code AS code,
                ticker,
                stock_name AS name,
                market,
                COALESCE(NULLIF(sector, ''), 'UNKNOWN') AS sector
            FROM stock_info
            WHERE market IN ('KOSPI', 'KOSDAQ')
            ORDER BY stock_code
            LIMIT %s
            """,
            (int(max_stocks),),
        )
    except Exception as exc:
        print(f"mysql_stock_universe_failed reason={exc}")
        return []

    stocks: list[Stock] = []
    for index, row in listing.iterrows():
        code = str(row["code"]).zfill(6)
        stocks.append(
            Stock(
                no=index + 1,
                name=str(row["name"]),
                code=code,
                ticker=str(row["ticker"]),
                market=str(row["market"]),
                sector=str(row.get("sector", "UNKNOWN") or "UNKNOWN"),
            )
        )
    return stocks


def load_raw_cached_stocks(max_stocks: int) -> list[Stock]:
    market_by_suffix = {"KS": "KOSPI", "KQ": "KOSDAQ"}
    stocks: list[Stock] = []
    for path in sorted(RAW_DIR.glob("*_*.csv")):
        if len(stocks) >= max_stocks:
            break
        stem = path.stem
        if "_" not in stem:
            continue
        code, ticker = stem.split("_", 1)
        suffix = ticker.rsplit(".", 1)[-1] if "." in ticker else ""
        market = market_by_suffix.get(suffix)
        if not market:
            continue
        stocks.append(
            Stock(
                no=len(stocks) + 1,
                name=code,
                code=code.zfill(6),
                ticker=ticker,
                market=market,
                sector="UNKNOWN",
            )
        )
    return stocks


def load_krx_stocks(max_stocks: int) -> list[Stock]:
    listing = fdr.StockListing("KRX").copy()
    listing["Code"] = listing["Code"].astype(str).str.zfill(6)
    listing = listing[listing["Market"].isin(["KOSPI", "KOSDAQ"])].copy()
    listing = listing.drop_duplicates(subset=["Code"], keep="first")

    sort_columns = [column for column in ["Marcap", "Amount", "Volume"] if column in listing.columns]
    for column in sort_columns:
        listing[column] = pd.to_numeric(listing[column], errors="coerce")
    if sort_columns:
        listing = listing.sort_values(sort_columns, ascending=[False] * len(sort_columns))
    else:
        listing = listing.sort_values(["Market", "Name"])

    selected = listing.head(max_stocks).reset_index(drop=True)
    market_suffix = {"KOSPI": "KS", "KOSDAQ": "KQ"}
    stocks = []
    for index, row in selected.iterrows():
        market = str(row["Market"])
        code = str(row["Code"]).zfill(6)
        stocks.append(
            Stock(
                no=index + 1,
                name=str(row["Name"]),
                code=code,
                ticker=f"{code}.{market_suffix[market]}",
                market=market,
                sector=str(row.get("Sector", "UNKNOWN") or "UNKNOWN"),
            )
        )
    return stocks


def select_stocks(universe: str, max_stocks: int) -> list[Stock]:
    if universe == "static":
        return fallback_stocks(max_stocks)
    if universe == "cached":
        stocks = load_mysql_cached_stocks(max_stocks)
        if stocks:
            print(f"stock_universe=mysql_cached count={len(stocks)}")
            return stocks
        stocks = load_raw_cached_stocks(max_stocks)
        if stocks:
            print(f"stock_universe=raw_cached count={len(stocks)}")
            return stocks
        print("stock_universe_fallback=static reason=empty_cached_universe")
        return fallback_stocks(max_stocks)

    try:
        stocks = load_krx_stocks(max_stocks)
    except Exception as exc:
        print(f"stock_universe_fallback=cached reason={exc}")
        stocks = load_mysql_cached_stocks(max_stocks)
        if stocks:
            print(f"stock_universe_fallback=mysql_cached count={len(stocks)}")
            return stocks
        stocks = load_raw_cached_stocks(max_stocks)
        if stocks:
            print(f"stock_universe_fallback=raw_cached count={len(stocks)}")
            return stocks
        return fallback_stocks(max_stocks)

    if not stocks:
        print("stock_universe_fallback=static reason=empty_krx_listing")
        return fallback_stocks(max_stocks)
    return stocks


def stock_info_frame(stocks: list[Stock]) -> pd.DataFrame:
    return pd.DataFrame([stock.__dict__ for stock in stocks])


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = frame.columns.get_level_values(0)

    frame = frame.reset_index()
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    rename_map = {
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj_close": "adj_close",
        "volume": "volume",
    }
    return frame.rename(columns=rename_map)


def raw_cache_path(stock: Stock) -> Path:
    return RAW_DIR / f"{stock.code}_{stock.ticker}.csv"


def load_cached_raw(stock: Stock) -> pd.DataFrame:
    path = raw_cache_path(stock)
    if not path.exists():
        return pd.DataFrame()

    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as error:
        print(f"raw_cache_load_failed={stock.ticker} reason={error}")
        return pd.DataFrame()

    if frame.empty:
        return pd.DataFrame()

    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    if "date" not in frame.columns:
        return pd.DataFrame()

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame["stock_code"] = stock.code
    frame["ticker"] = stock.ticker
    frame["stock_name"] = stock.name
    frame["market"] = stock.market
    frame["sector"] = stock.sector
    print(f"raw_cache_used={stock.ticker} rows={len(frame)}")
    return frame


def download_one(stock: Stock, prefer_cache: bool = False, timeout: int = 8) -> pd.DataFrame:
    if prefer_cache:
        cached = load_cached_raw(stock)
        if not cached.empty:
            return cached

    try:
        frame = yf.download(
            stock.ticker,
            start=START_DATE,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=timeout,
        )
    except Exception as error:
        print(f"download_failed={stock.ticker} reason={error}")
        return load_cached_raw(stock)

    if frame.empty:
        return load_cached_raw(stock)

    frame = normalize_columns(frame)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame["stock_code"] = stock.code
    frame["ticker"] = stock.ticker
    frame["stock_name"] = stock.name
    frame["market"] = stock.market
    frame["sector"] = stock.sector
    return frame


def clean_price_data(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "stock_code", "ticker", "open", "high", "low", "close", "volume"]
    frame = frame.copy()
    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame["close"]

    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for column in numeric_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=required)
    frame = frame[frame["volume"] >= 0]
    frame = frame.sort_values(["stock_code", "date"])
    frame = frame.drop_duplicates(subset=["stock_code", "date"], keep="last")
    return frame[
        [
            "date",
            "stock_code",
            "ticker",
            "stock_name",
            "market",
            "sector",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        ]
    ].reset_index(drop=True)


def add_features(price_frame: pd.DataFrame) -> pd.DataFrame:
    frame = price_frame.copy()
    grouped = frame.groupby("stock_code", group_keys=False)

    frame["return_1d"] = grouped["adj_close"].pct_change()
    frame["ma20"] = grouped["adj_close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    frame["ma20_gap"] = (frame["adj_close"] - frame["ma20"]) / frame["ma20"]
    frame["volatility_20d"] = grouped["return_1d"].transform(lambda s: s.rolling(20, min_periods=20).std())
    frame["volume_change_1d"] = grouped["volume"].pct_change()
    volume_ma20 = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    close_std20 = grouped["adj_close"].transform(lambda s: s.rolling(20, min_periods=20).std())
    frame["volume_ratio_20d"] = frame["volume"] / volume_ma20

    delta = grouped["adj_close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.groupby(frame["stock_code"]).transform(lambda s: s.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean())
    avg_loss = loss.groupby(frame["stock_code"]).transform(lambda s: s.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean())
    rs = avg_gain / avg_loss
    frame["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = grouped["adj_close"].transform(lambda s: s.ewm(span=12, min_periods=12, adjust=False).mean())
    ema26 = grouped["adj_close"].transform(lambda s: s.ewm(span=26, min_periods=26, adjust=False).mean())
    frame["macd"] = ema12 - ema26
    frame["macd_signal"] = frame.groupby("stock_code")["macd"].transform(
        lambda s: s.ewm(span=9, min_periods=9, adjust=False).mean()
    )
    frame["macd_hist"] = frame["macd"] - frame["macd_signal"]

    frame["bollinger_upper"] = frame["ma20"] + (close_std20 * 2)
    frame["bollinger_lower"] = frame["ma20"] - (close_std20 * 2)
    frame["bollinger_width"] = (frame["bollinger_upper"] - frame["bollinger_lower"]) / frame["ma20"]

    feature_columns = [
        "return_1d",
        "ma20",
        "ma20_gap",
        "volatility_20d",
        "volume_change_1d",
        "volume_ratio_20d",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bollinger_upper",
        "bollinger_lower",
        "bollinger_width",
    ]
    base_columns = [
        "date",
        "stock_code",
        "ticker",
        "stock_name",
        "market",
        "sector",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]

    return frame[base_columns + feature_columns].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Korean stock price data and build features.")
    parser.add_argument("--universe", choices=["krx", "static", "cached"], default=DEFAULT_UNIVERSE)
    parser.add_argument("--max-stocks", type=int, default=DEFAULT_MAX_STOCKS)
    parser.add_argument("--prefer-cache", action="store_true", help="Use existing raw CSV cache before calling yfinance.")
    parser.add_argument("--download-timeout", type=int, default=8, help="Per-symbol yfinance timeout in seconds.")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MYSQL_IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(CACHE_DIR.resolve()))

    stocks = select_stocks(args.universe, args.max_stocks)
    info = stock_info_frame(stocks)
    info.to_csv(PROCESSED_DIR / "stock_info.csv", index=False, encoding="utf-8-sig")

    raw_frames: list[pd.DataFrame] = []
    summary: list[dict[str, object]] = []
    for stock in stocks:
        raw = download_one(stock, prefer_cache=args.prefer_cache, timeout=args.download_timeout)
        if raw.empty:
            summary.append({**stock.__dict__, "rows": 0, "status": "empty"})
            continue
        raw.to_csv(raw_cache_path(stock), index=False, encoding="utf-8-sig")
        raw_frames.append(raw)
        summary.append({**stock.__dict__, "rows": len(raw), "status": "ok"})

    if not raw_frames:
        raise RuntimeError("No stock data was downloaded from yfinance.")

    raw_all = pd.concat(raw_frames, ignore_index=True)
    price = clean_price_data(raw_all)
    features = add_price_features(price)

    price.to_csv(PROCESSED_DIR / "stock_price.csv", index=False, encoding="utf-8-sig")
    features.to_csv(PROCESSED_DIR / "feature_data.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(PROCESSED_DIR / "collection_summary.csv", index=False, encoding="utf-8-sig")

    info.rename(columns={"code": "stock_code"})[
        ["stock_code", "ticker", "name", "market", "sector"]
    ].rename(columns={"name": "stock_name"}).to_csv(
        MYSQL_IMPORT_DIR / "stock_info.csv",
        index=False,
        encoding="cp949",
    )
    price.rename(
        columns={
            "date": "trade_date",
            "open": "open_price",
            "high": "high_price",
            "low": "low_price",
            "close": "close_price",
            "adj_close": "adj_close_price",
        }
    )[
        [
            "stock_code",
            "trade_date",
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "adj_close_price",
            "volume",
        ]
    ].to_csv(MYSQL_IMPORT_DIR / "stock_price.csv", index=False, encoding="cp949")

    print(f"stock_universe={args.universe}")
    print(f"stocks_requested={len(stocks)}")
    print(f"stocks_downloaded={sum(item['status'] == 'ok' for item in summary)}")
    print(f"stock_price_rows={len(price)}")
    print(f"feature_data_rows={len(features)}")
    print(f"date_min={price['date'].min()}")
    print(f"date_max={price['date'].max()}")
    print(f"output_dir={PROCESSED_DIR.resolve()}")


if __name__ == "__main__":
    main()
