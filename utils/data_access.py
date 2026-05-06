"""Load procurement data from the Impala warehouse (``logistics``) or local CSV fallback."""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

try:
    import cml.data_v1 as cmldata
except ImportError:
    cmldata = None

DEFAULT_DATA_DIR = os.environ.get(
    "LOGISTICS_DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw"),
)

WAREHOUSE_DB = os.environ.get("LOGISTICS_DATABASE", "logistics")
WAREHOUSE_CONN = os.environ.get("LOGISTICS_IMPALA_CONN", "default-impala-aws")

TABLE_PRICE = "item_price_history_forecasting"
TABLE_SHIP = "supplier_shipping_performance"
TABLE_TX = "procurement_transactions"


def _csv_mode_explicit() -> Optional[str]:
    return os.environ.get("LOGISTICS_DATA_SOURCE", "").strip().lower() or None


def using_csv_files() -> bool:
    """
    True when readers use CSV under ``LOGISTICS_DATA_DIR`` / ``data/raw``.
    False when ``cml.data_v1`` is used against Impala (default on Cloudera AI).
    """
    mode = _csv_mode_explicit()
    if mode in ("csv", "file", "local"):
        return True
    if mode in ("warehouse", "impala", "dw", "hive"):
        return False
    # Default: warehouse if the CML Data library is available; otherwise CSV for laptops.
    return cmldata is None


def _query_warehouse(sql: str) -> pd.DataFrame:
    if cmldata is None:
        raise RuntimeError(
            "Warehouse reads require cml.data_v1 (Cloudera AI). "
            "For local CSV, unset LOGISTICS_DATA_SOURCE=warehouse or run without that env."
        )
    conn = cmldata.get_connection(WAREHOUSE_CONN)
    try:
        return conn.get_pandas_dataframe(sql)
    finally:
        conn.close()


def _load_table_warehouse(table: str) -> pd.DataFrame:
    fq = f"{WAREHOUSE_DB}.{table}"
    return _query_warehouse(f"SELECT * FROM {fq}")


def load_price_history(data_dir: Optional[str] = None) -> pd.DataFrame:
    if using_csv_files():
        base = data_dir or DEFAULT_DATA_DIR
        path = os.path.join(base, f"{TABLE_PRICE}.csv")
        df = pd.read_csv(path)
    else:
        df = _load_table_warehouse(TABLE_PRICE)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["nsn", "date"])


def load_transactions(data_dir: Optional[str] = None) -> pd.DataFrame:
    if using_csv_files():
        base = data_dir or DEFAULT_DATA_DIR
        path = os.path.join(base, f"{TABLE_TX}.csv")
        df = pd.read_csv(path)
    else:
        df = _load_table_warehouse(TABLE_TX)
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def load_supplier_shipping(data_dir: Optional[str] = None) -> pd.DataFrame:
    if using_csv_files():
        base = data_dir or DEFAULT_DATA_DIR
        path = os.path.join(base, f"{TABLE_SHIP}.csv")
        df = pd.read_csv(path)
    else:
        df = _load_table_warehouse(TABLE_SHIP)
    if "report_month" in df.columns:
        df["report_month"] = pd.to_datetime(df["report_month"])
    return df


load_procurement_transactions = load_transactions
