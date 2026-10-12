from __future__ import annotations

from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd
import yfinance as yf


START_DATE = "2020-01-01"
PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE = PROCESSED_DIR / "market_index.csv"


def normalize(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = frame.columns.get_level_values(0)

    frame = frame.reset_index()
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    if "date" not in frame.columns:
        return pd.DataFrame()

    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame.get("close")

    required = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    for column in required:
        if column not in frame.columns:
            frame[column] = 0 if column == "volume" else pd.NA

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "adj_close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    frame = frame.drop_duplicates(subset=["date"], keep="last")
    frame["index_code"] = "KOSPI"
    frame["index_name"] = "종합주가지수"
    frame["source"] = source
    frame["return_1d"] = frame["close"].pct_change()
    frame["return_5d"] = frame["close"].pct_change(5)
    frame["return_20d"] = frame["close"].pct_change(20)
    frame["volatility_20d"] = frame["return_1d"].rolling(20, min_periods=20).std()
    return frame[
        [
            "date",
            "index_code",
            "index_name",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "return_1d",
            "return_5d",
            "return_20d",
            "volatility_20d",
            "source",
        ]
    ].reset_index(drop=True)


def download_yfinance() -> pd.DataFrame:
    yf.set_tz_cache_location(str(Path(".cache/yfinance").resolve()))
    frame = yf.download(
        "^KS11",
        start=START_DATE,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=10,
    )
    return normalize(frame, "yfinance")


def download_fdr() -> pd.DataFrame:
    frame = fdr.DataReader("KS11", START_DATE)
    return normalize(frame, "FinanceDataReader")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    for downloader in [download_yfinance, download_fdr]:
        try:
            frame = downloader()
            if not frame.empty:
                frame.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
                print(f"rows={len(frame)}")
                print(f"date_min={frame['date'].min().date()}")
                print(f"date_max={frame['date'].max().date()}")
                print(f"source={frame['source'].iloc[-1]}")
                print(f"output_file={OUTPUT_FILE.resolve()}")
                return
        except Exception as error:
            errors.append(f"{downloader.__name__}: {error}")

    if OUTPUT_FILE.exists():
        frame = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig")
        print("market_index_cache_used=1")
        print(f"rows={len(frame)}")
        print(f"date_max={frame['date'].max()}")
        return

    raise RuntimeError("Could not download KOSPI index. " + " | ".join(errors))


if __name__ == "__main__":
    main()
