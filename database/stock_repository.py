"""Read dashboard snapshots from Firestore, or the existing MySQL repository."""

import os
import threading
import time

import pandas as pd

from database import mysql_repository
from database.firebase_client import firestore_db

_cache = {}
_lock = threading.Lock()


def backend():
    return os.getenv("STOCK_DB_BACKEND", "mysql")


def enabled():
    return backend() == "firebase" or (backend() == "mysql" and mysql_repository.mysql_enabled())


def snapshot():
    # One bounded document read per minute instead of reading every stock per visit.
    with _lock:
        if time.monotonic() >= _cache.get("expires", 0):
            try:
                document = firestore_db().collection("app_metadata").document("dashboard").get(timeout=5, retry=None)
                _cache["data"] = document.to_dict() or {}
                _cache["error"] = None
            except Exception:
                _cache["data"] = {}
                _cache["error"] = "Firestore is unavailable"
            _cache["expires"] = time.monotonic() + 60
        if _cache.get("error"):
            raise RuntimeError(_cache["error"])
        return _cache["data"]


def ping():
    if backend() != "firebase":
        return mysql_repository.ping()
    try:
        snapshot()
        return {"available": True, "enabled": True, "source": "firebase", "error": None}
    except Exception:
        return {"available": False, "enabled": True, "source": "firebase", "error": "Firestore is unavailable"}


def stock_listing_frame():
    if backend() != "firebase":
        return mysql_repository.stock_listing_frame()
    return pd.DataFrame(snapshot().get("stocks", [])).rename(columns={
        "code": "Code", "name": "Name", "market": "Market", "sector": "Sector", "ticker": "Ticker",
    })


def top_risks_frame(limit):
    if backend() != "firebase":
        return mysql_repository.top_risks_frame(limit)
    data = snapshot()
    return data.get("latest_date"), pd.DataFrame(data.get("predictions", [])[:limit])


def table_counts():
    if backend() != "firebase":
        return mysql_repository.table_counts()
    data = snapshot()
    return {"stock_info_snapshot": len(data.get("stocks", [])), "prediction_snapshot": len(data.get("predictions", []))}
