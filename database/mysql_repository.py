from __future__ import annotations

import os
from typing import Any

import pandas as pd
import pymysql
from pymysql.cursors import DictCursor


DEFAULT_DATABASE = "stock_risk_project"


def mysql_config(include_database: bool = True) -> dict[str, Any]:
    config: dict[str, Any] = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "charset": "utf8mb4",
        "cursorclass": DictCursor,
        "autocommit": True,
    }
    if include_database:
        config["database"] = os.getenv("MYSQL_DATABASE", DEFAULT_DATABASE)
    return config


def mysql_enabled() -> bool:
    value = os.getenv("MYSQL_ENABLED", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    return bool(os.getenv("MYSQL_PASSWORD"))


def connect(include_database: bool = True):
    return pymysql.connect(**mysql_config(include_database=include_database))


def ping() -> dict[str, Any]:
    try:
        with connect(include_database=False) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1 AS ok")
                row = cursor.fetchone()
        return {
            "available": bool(row and row.get("ok") == 1),
            "host": os.getenv("MYSQL_HOST", "localhost"),
            "port": int(os.getenv("MYSQL_PORT", "3306")),
            "database": os.getenv("MYSQL_DATABASE", DEFAULT_DATABASE),
            "user": os.getenv("MYSQL_USER", "root"),
            "error": None,
        }
    except Exception as error:
        return {
            "available": False,
            "host": os.getenv("MYSQL_HOST", "localhost"),
            "port": int(os.getenv("MYSQL_PORT", "3306")),
            "database": os.getenv("MYSQL_DATABASE", DEFAULT_DATABASE),
            "user": os.getenv("MYSQL_USER", "root"),
            "error": str(error),
        }


def query_frame(sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    with connect(include_database=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
    return pd.DataFrame(rows)


def stock_listing_frame() -> pd.DataFrame:
    return query_frame(
        """
        SELECT
            stock_code AS Code,
            stock_name AS Name,
            market AS Market,
            sector AS Sector,
            ticker AS Ticker
        FROM stock_info
        WHERE market IN ('KOSPI', 'KOSDAQ')
        ORDER BY market, stock_name
        """
    )


def top_risks_frame(limit: int) -> tuple[str | None, pd.DataFrame]:
    latest_date = query_frame("SELECT MAX(trade_date) AS latest_date FROM prediction_result")
    if latest_date.empty or pd.isna(latest_date.iloc[0]["latest_date"]):
        return None, pd.DataFrame()

    date_value = latest_date.iloc[0]["latest_date"]
    frame = query_frame(
        """
        SELECT
            p.trade_date AS date,
            p.stock_code,
            s.stock_name,
            s.market,
            s.sector,
            s.ticker,
            p.risk_probability,
            p.risk_level
        FROM prediction_result p
        JOIN stock_info s ON s.stock_code = p.stock_code
        WHERE p.trade_date = %s
        ORDER BY p.risk_probability DESC
        LIMIT %s
        """,
        (date_value, int(limit)),
    )
    return str(pd.Timestamp(date_value).date()), frame


def table_counts() -> dict[str, int | None]:
    tables = [
        "stock_info",
        "stock_price",
        "feature_data",
        "anomaly_result",
        "risk_label",
        "model_info",
        "prediction_result",
        "alert_log",
    ]
    counts: dict[str, int | None] = {}
    for table in tables:
        try:
            result = query_frame(f"SELECT COUNT(*) AS count FROM `{table}`")
            counts[table] = int(result.iloc[0]["count"]) if not result.empty else 0
        except Exception:
            counts[table] = None
    return counts
