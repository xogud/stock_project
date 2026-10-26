"""Refresh the deployment-sized market context file after sector mapping."""

from pathlib import Path

import pandas as pd


def export_market_context(directory=Path("data/processed")):
    columns = ["date", "stock_code", "sector", "return_1d"]
    frame = pd.read_csv(directory / "feature_data.csv", usecols=columns,
                        dtype={"stock_code": str}, encoding="utf-8-sig")
    if frame.empty:
        raise ValueError("Cannot publish empty market context")
    listing = pd.read_csv(directory / "stock_info.csv", usecols=["code", "sector"],
                          dtype={"code": str}, encoding="utf-8-sig")
    sectors = listing.set_index("code")["sector"]
    frame["sector"] = frame["stock_code"].map(sectors).fillna(frame["sector"]).fillna("기타")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
    target = directory / "market_context_daily.csv"
    temporary = target.with_suffix(".tmp")
    frame[columns].to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(target)
    return {"rows": len(frame), "latest_date": frame["date"].max(), "output": str(target)}


if __name__ == "__main__":
    print(export_market_context())
