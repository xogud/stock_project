from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import pymysql


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPORT_DIR = PROJECT_ROOT / "data" / "mysql_import_ready"
SCHEMA_PATH = PROJECT_ROOT / "database" / "schema.sql"


TABLES = {
    "stock_info": {
        "file": "stock_info.csv",
        "columns": ["stock_code", "ticker", "stock_name", "market", "sector"],
        "numeric": set(),
        "update": ["ticker", "stock_name", "market", "sector"],
    },
    "stock_price": {
        "file": "stock_price.csv",
        "columns": [
            "stock_code",
            "trade_date",
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "adj_close_price",
            "volume",
        ],
        "numeric": {
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "adj_close_price",
            "volume",
        },
        "update": [
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "adj_close_price",
            "volume",
        ],
    },
    "feature_data": {
        "file": "feature_data.csv",
        "columns": [
            "stock_code",
            "trade_date",
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
        ],
        "numeric": {
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
        },
        "update": [
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
        ],
    },
    "anomaly_result": {
        "file": "anomaly_result.csv",
        "columns": [
            "stock_code",
            "trade_date",
            "z_score",
            "ewma_value",
            "cusum_pos",
            "cusum_neg",
            "anomaly_score",
            "is_anomaly",
        ],
        "numeric": {
            "z_score",
            "ewma_value",
            "cusum_pos",
            "cusum_neg",
            "anomaly_score",
            "is_anomaly",
        },
        "update": [
            "z_score",
            "ewma_value",
            "cusum_pos",
            "cusum_neg",
            "anomaly_score",
            "is_anomaly",
        ],
    },
    "risk_label": {
        "file": "risk_label.csv",
        "columns": [
            "stock_code",
            "trade_date",
            "target_window_days",
            "risk_label",
            "label_reason",
        ],
        "numeric": {
            "target_window_days",
            "risk_label",
        },
        "update": [
            "risk_label",
            "label_reason",
        ],
    },
    "model_info": {
        "file": "model_info.csv",
        "columns": [
            "model_name",
            "model_type",
            "model_version",
            "target_window_days",
            "model_path",
            "feature_columns",
            "train_start_date",
            "train_end_date",
            "metrics",
            "is_active",
        ],
        "numeric": {
            "target_window_days",
            "is_active",
        },
        "update": [
            "model_type",
            "target_window_days",
            "model_path",
            "feature_columns",
            "train_start_date",
            "train_end_date",
            "metrics",
            "is_active",
        ],
    },
    "prediction_result": {
        "file": "prediction_result.csv",
        "columns": [
            "stock_code",
            "trade_date",
            "model_id",
            "risk_probability",
            "risk_level",
            "target_window_days",
        ],
        "numeric": {
            "model_id",
            "risk_probability",
            "target_window_days",
        },
        "update": [
            "risk_probability",
            "risk_level",
            "target_window_days",
        ],
    },
    "alert_log": {
        "file": "alert_log.csv",
        "columns": [
            "stock_code",
            "trade_date",
            "risk_level",
            "message",
            "top_factors",
        ],
        "numeric": set(),
        "update": [
            "risk_level",
            "message",
            "top_factors",
        ],
    },
}

DEFAULT_TABLE_ORDER = [
    "stock_info",
    "stock_price",
    "feature_data",
    "anomaly_result",
    "risk_label",
    "model_info",
    "prediction_result",
    "alert_log",
]

DELETE_TABLE_ORDER = [
    "alert_log",
    "prediction_result",
    "model_info",
    "risk_label",
    "anomaly_result",
    "feature_data",
    "stock_price",
    "stock_info",
]


def mysql_client_path() -> str:
    configured = os.getenv("MYSQL_CLIENT")
    if configured:
        return configured
    found = shutil.which("mysql")
    if found:
        return found
    workbench_client = Path(r"C:\Program Files\MySQL\MySQL Workbench 8.0\mysql.exe")
    if workbench_client.exists():
        return str(workbench_client)
    raise RuntimeError("mysql client was not found. Set MYSQL_CLIENT to the mysql.exe path.")


def run_mysql(sql: str, args: argparse.Namespace) -> None:
    cmd = [
        mysql_client_path(),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--user",
        args.user,
        "--default-character-set=utf8mb4",
    ]
    if args.database:
        cmd.extend(["--database", args.database])

    env = os.environ.copy()
    if args.password:
        env["MYSQL_PWD"] = args.password

    completed = subprocess.run(
        cmd,
        input=sql,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())


def sql_value(value: str | None, numeric: bool) -> str:
    if value is None:
        return "NULL"

    value = value.strip()
    if value == "":
        return "NULL"

    if numeric:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return "NULL"
        if number.is_integer():
            return str(int(number))
        return repr(number)

    escaped = value.replace("\\", "\\\\").replace("'", "''")
    return f"'{escaped}'"


def chunks(items: list[dict[str, str]], size: int) -> Iterable[list[dict[str, str]]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="cp949", newline="") as file:
        return list(csv.DictReader(file))


def python_value(value: str | None, numeric: bool, column: str) -> object:
    if value is None:
        return None

    value = value.strip()
    if value == "":
        return None

    if column == "stock_code":
        return value.zfill(6)

    if numeric:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        if number.is_integer():
            return int(number)
        return number

    return value


def mysql_connection(args: argparse.Namespace):
    return pymysql.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database,
        charset="utf8mb4",
        autocommit=False,
    )


def active_model_id(args: argparse.Namespace) -> int:
    with mysql_connection(args) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id
                FROM model_info
                WHERE is_active = 1
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()
    if not row:
        raise RuntimeError("No active model_info row exists. Load model_info before prediction_result.")
    return int(row[0])


def build_insert_statement(table: str, table_config: dict[str, object]) -> str:
    columns = table_config["columns"]
    update_columns = table_config["update"]
    column_sql = ", ".join(f"`{column}`" for column in columns)
    placeholders = ", ".join(["%s"] * len(columns))
    update_sql = ", ".join(f"`{column}` = VALUES(`{column}`)" for column in update_columns)
    return (
        f"INSERT INTO `{table}` ({column_sql}) "
        f"VALUES ({placeholders}) "
        f"ON DUPLICATE KEY UPDATE {update_sql}"
    )


def row_values(row: dict[str, str], table_config: dict[str, object]) -> tuple[object, ...]:
    columns = table_config["columns"]
    numeric_columns = table_config["numeric"]
    return tuple(python_value(row.get(column), column in numeric_columns, column) for column in columns)


def build_insert_sql(table: str, table_config: dict[str, object], rows: list[dict[str, str]]) -> str:
    columns = table_config["columns"]
    numeric_columns = table_config["numeric"]
    update_columns = table_config["update"]

    value_rows = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if column == "stock_code" and value:
                value = value.strip().zfill(6)
            values.append(sql_value(value, column in numeric_columns))
        value_rows.append("(" + ", ".join(values) + ")")

    column_sql = ", ".join(f"`{column}`" for column in columns)
    update_sql = ", ".join(f"`{column}` = VALUES(`{column}`)" for column in update_columns)
    return (
        f"INSERT INTO `{table}` ({column_sql})\n"
        f"VALUES\n{',\n'.join(value_rows)}\n"
        f"ON DUPLICATE KEY UPDATE {update_sql};\n"
    )


def load_table(table: str, args: argparse.Namespace) -> int:
    table_config = TABLES[table]
    path = IMPORT_DIR / table_config["file"]
    rows = read_rows(path)
    if table == "prediction_result":
        model_id = active_model_id(args)
        for row in rows:
            row["model_id"] = str(model_id)

    statement = build_insert_statement(table, table_config)
    with mysql_connection(args) as connection:
        with connection.cursor() as cursor:
            for batch in chunks(rows, args.batch_size):
                cursor.executemany(statement, [row_values(row, table_config) for row in batch])
                connection.commit()

    return len(rows)


def replace_tables(tables: list[str], args: argparse.Namespace) -> None:
    selected = set(tables)
    with mysql_connection(args) as connection:
        with connection.cursor() as cursor:
            for table in DELETE_TABLE_ORDER:
                if table not in selected:
                    continue
                cursor.execute(f"DELETE FROM `{table}`")
                print(f"{table}: deleted {cursor.rowcount} rows")
            connection.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Load processed CSV files into MySQL.")
    parser.add_argument("--host", default=os.getenv("MYSQL_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MYSQL_PORT", "3306")))
    parser.add_argument("--user", default=os.getenv("MYSQL_USER", "root"))
    parser.add_argument("--password", default=os.getenv("MYSQL_PASSWORD", ""))
    parser.add_argument("--database", default=os.getenv("MYSQL_DATABASE", "stock_risk_project"))
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--init-schema", action="store_true", help="Run database/schema.sql before loading CSV files.")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete selected import tables before loading so MySQL matches the current CSV outputs.",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=DEFAULT_TABLE_ORDER,
        default=DEFAULT_TABLE_ORDER,
        help="Tables to load. Defaults to all available import tables.",
    )
    args = parser.parse_args()

    if args.init_schema:
        run_mysql(SCHEMA_PATH.read_text(encoding="utf-8"), argparse.Namespace(**{**vars(args), "database": ""}))

    if args.replace:
        replace_tables(args.tables, args)

    total = 0
    for table in args.tables:
        count = load_table(table, args)
        total += count
        print(f"{table}: loaded {count} rows")

    print(f"done: loaded {total} rows")


if __name__ == "__main__":
    main()
