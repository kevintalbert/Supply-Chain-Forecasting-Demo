#!/usr/bin/env python3
"""
Load logistics procurement CSVs into Impala database `logistics`.
Table names match filenames without .csv (customer warehouse layout).

Requires CML Data connection (cml.data_v1) and workload password auth.
"""

import os
import sys
from datetime import datetime

import pandas as pd

try:
    import cml.data_v1 as cmldata
except ImportError:
    cmldata = None

CONNECTION_NAME = os.environ.get("LOGISTICS_IMPALA_CONN", "default-impala-aws")
DATABASE_NAME = "logistics"
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")

TABLE_SCHEMAS = {
    "supplier_shipping_performance": {
        "columns": [
            ("supplier_id", "STRING"),
            ("supplier_name", "STRING"),
            ("report_month", "STRING"),
            ("avg_lead_time_days", "DOUBLE"),
            ("on_time_delivery_rate", "DOUBLE"),
            ("defect_rate", "DOUBLE"),
            ("primary_shipping_code", "STRING"),
            ("region", "STRING"),
            ("shipments_completed", "INT"),
            ("expedite_rate", "DOUBLE"),
            ("avg_freight_cost", "DOUBLE"),
            ("risk_score", "DOUBLE"),
        ],
        "description": "Monthly supplier shipping KPIs",
    },
    "item_price_history_forecasting": {
        "columns": [
            ("nsn", "STRING"),
            ("item_description", "STRING"),
            ("date", "STRING"),
            ("unit_price", "DOUBLE"),
            ("demand_quantity", "DOUBLE"),
            ("market_index", "DOUBLE"),
            ("supplier_id", "STRING"),
            ("contract_id", "STRING"),
            ("data_density_class", "STRING"),
            ("recommended_modeling_approach", "STRING"),
        ],
        "description": "Price history with dense vs sparse modeling hints",
    },
    "procurement_transactions": {
        "columns": [
            ("order_id", "STRING"),
            ("order_date", "STRING"),
            ("nsn", "STRING"),
            ("item_description", "STRING"),
            ("quantity", "DOUBLE"),
            ("unit_price", "DOUBLE"),
            ("total_price", "DOUBLE"),
            ("supplier_id", "STRING"),
            ("supplier_name", "STRING"),
            ("shipping_code", "STRING"),
            ("priority_code", "STRING"),
            ("destination", "STRING"),
            ("contract_id", "STRING"),
            ("days_to_deliver", "INT"),
            ("order_status", "STRING"),
        ],
        "description": "Order-level procurement transactions",
    },
}


def connect():
    if cmldata is None:
        raise RuntimeError("cml.data_v1 not available — run inside Cloudera AI Workbench")
    return cmldata.get_connection(CONNECTION_NAME)


def create_database(conn):
    cur = conn.get_cursor()
    try:
        conn.get_pandas_dataframe("SHOW DATABASES")
    except Exception:
        pass
    try:
        cur.execute(f"DROP DATABASE IF EXISTS {DATABASE_NAME} CASCADE")
    except Exception:
        pass
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}")
    cur.execute(f"USE {DATABASE_NAME}")
    cur.close()


def create_table(conn, table_name, schema_info):
    cur = conn.get_cursor()
    cur.execute(f"USE {DATABASE_NAME}")
    cols = ",\n    ".join(f"{n} {t}" for n, t in schema_info["columns"])
    cur.execute(
        f"""
        CREATE TABLE {DATABASE_NAME}.{table_name} (
            {cols}
        )
        STORED AS TEXTFILE
        """
    )
    cur.close()


def load_csv(conn, table_name, csv_path, schema_info):
    if not os.path.exists(csv_path):
        print(f"Skip missing file: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    expected = [c[0] for c in schema_info["columns"]]
    df = df[[c for c in expected if c in df.columns]]
    cur = conn.get_cursor()
    cur.execute(f"USE {DATABASE_NAME}")
    ph = ", ".join(["%s"] * len(df.columns))
    sql = f"INSERT INTO {DATABASE_NAME}.{table_name} VALUES ({ph})"
    for _, row in df.iterrows():
        cur.execute(sql, tuple(None if pd.isna(x) else x for x in row))
    cur.close()
    print(f"Loaded {len(df)} rows -> {table_name}")


def main():
    print(f"Target database: {DATABASE_NAME}")
    print(f"CSV path: {DATA_PATH}")
    conn = connect()
    create_database(conn)
    for tname, schema in TABLE_SCHEMAS.items():
        create_table(conn, tname, schema)
    for tname in TABLE_SCHEMAS:
        load_csv(conn, tname, os.path.join(DATA_PATH, f"{tname}.csv"), TABLE_SCHEMAS[tname])
    print("Done at", datetime.now().isoformat())
    conn.close()


if __name__ == "__main__":
    main()
