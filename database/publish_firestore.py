"""Publish a small atomic dashboard snapshot. Defaults to a no-network dry run."""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def build_snapshot(root=ROOT):
    stocks = pd.read_csv(root / "data/processed/stock_info.csv", dtype={"code": str})
    stocks = stocks[["code", "name", "market", "sector", "ticker"]].fillna("")
    predictions = pd.read_csv(root / "data/processed/prediction_result.csv", dtype={"stock_code": str})
    if predictions.empty or stocks.empty:
        raise ValueError("Stock and prediction data must not be empty")
    predictions["trade_date"] = pd.to_datetime(predictions["trade_date"], errors="raise")
    latest = predictions["trade_date"].max()
    predictions = predictions[predictions["trade_date"] == latest].copy()
    predictions = predictions.sort_values(["risk_probability", "model_id"], ascending=False).drop_duplicates("stock_code")
    if not predictions["risk_probability"].between(0, 1).all():
        raise ValueError("Invalid prediction probabilities")
    predictions = predictions.merge(stocks, left_on="stock_code", right_on="code", validate="many_to_one")
    if predictions.empty:
        raise ValueError("No matching stocks for predictions")
    predictions["date"] = latest.strftime("%Y-%m-%d")
    predictions = predictions.rename(columns={"name": "stock_name"})
    payload = {
        "schema_version": 1,
        "latest_date": latest.strftime("%Y-%m-%d"),
        "stocks": stocks.to_dict("records"),
        "predictions": predictions[["date", "stock_code", "stock_name", "market", "sector", "ticker", "risk_probability", "risk_level"]].to_dict("records"),
    }
    # Leave space for Firestore's field/document overhead under its 1 MiB limit.
    size = len(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode())
    if size > 700_000:
        raise ValueError("Dashboard snapshot exceeds the 700 KB safety limit")
    return payload, size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write the snapshot to the configured Firebase project")
    args = parser.parse_args()
    payload, size = build_snapshot()
    print(json.dumps({"latest_date": payload["latest_date"], "stocks": len(payload["stocks"]), "predictions": len(payload["predictions"]), "bytes": size, "writes": 1, "apply": args.apply}))
    if args.apply:
        from database.firebase_client import firestore_db
        from google.cloud import firestore
        db = firestore_db()
        ref = db.collection("app_metadata").document("dashboard")

        @firestore.transactional
        def publish(transaction):
            current = ref.get(transaction=transaction, timeout=10).to_dict() or {}
            if current.get("latest_date", "") > payload["latest_date"]:
                raise ValueError("Refusing to overwrite a newer snapshot")
            transaction.set(ref, payload)

        publish(db.transaction(max_attempts=3))
        print("Dashboard snapshot published")


if __name__ == "__main__":
    main()
