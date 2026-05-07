"""Load procurement data: try Impala ``logistics`` tables first; fall back to local CSVs (read-only)."""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

try:
    import cml.data_v1 as cmldata
except ImportError:
    cmldata = None

DEFAULT_DATA_DIR = os.environ.get("LOGISTICS_DATA_DIR") or os.path.join("data", "raw")

WAREHOUSE_DB = os.environ.get("LOGISTICS_DATABASE", "logistics")
WAREHOUSE_CONN = os.environ.get("LOGISTICS_IMPALA_CONN", "default-impala-aws")

TABLE_PRICE = "item_price_history_forecasting"
TABLE_SHIP = "supplier_shipping_performance"
TABLE_TX = "procurement_transactions"

_fallback_notice_shown = False


def _csv_mode_explicit() -> Optional[str]:
    return os.environ.get("LOGISTICS_DATA_SOURCE", "").strip().lower() or None


def using_csv_files() -> bool:
    """
    True when ``LOGISTICS_DATA_SOURCE`` forces CSV only (skip warehouse reads).
    False otherwise — warehouse is attempted first when ``cml.data_v1`` exists, then CSV if empty/unavailable.
    """
    mode = _csv_mode_explicit()
    if mode in ("csv", "file", "local"):
        return True
    if mode in ("warehouse", "impala", "dw", "hive"):
        return False
    return False


def _warn_fallback_csv(base: str, reason: str) -> None:
    global _fallback_notice_shown
    if _fallback_notice_shown:
        return
    _fallback_notice_shown = True
    print(
        f"Procurement data: using local CSVs under {base} ({reason}). "
        "No database writes."
    )


def _query_warehouse(sql: str) -> pd.DataFrame:
    if cmldata is None:
        raise RuntimeError("cml.data_v1 not available")
    conn = cmldata.get_connection(WAREHOUSE_CONN)
    try:
        return conn.get_pandas_dataframe(sql)
    finally:
        conn.close()


def _load_table_warehouse(table: str) -> pd.DataFrame:
    fq = f"{WAREHOUSE_DB}.{table}"
    return _query_warehouse(f"SELECT * FROM {fq}")


def _try_warehouse(table: str) -> Optional[pd.DataFrame]:
    """Return DataFrame from Impala, or None if empty or SELECT fails."""
    try:
        df = _load_table_warehouse(table)
        if df is None or len(df) == 0:
            return None
        return df
    except Exception:
        return None


def _read_csv(base: str, filename: str) -> pd.DataFrame:
    path = os.path.join(base, filename)
    return pd.read_csv(path)


def load_price_history(data_dir: Optional[str] = None) -> pd.DataFrame:
    base = data_dir or DEFAULT_DATA_DIR
    if using_csv_files():
        df = _read_csv(base, f"{TABLE_PRICE}.csv")
    elif cmldata is None:
        _warn_fallback_csv(base, "cml.data_v1 not available")
        df = _read_csv(base, f"{TABLE_PRICE}.csv")
    else:
        df = _try_warehouse(TABLE_PRICE)
        if df is None:
            _warn_fallback_csv(base, "warehouse empty or SELECT failed")
            df = _read_csv(base, f"{TABLE_PRICE}.csv")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["nsn", "date"])


def load_transactions(data_dir: Optional[str] = None) -> pd.DataFrame:
    base = data_dir or DEFAULT_DATA_DIR
    if using_csv_files():
        df = _read_csv(base, f"{TABLE_TX}.csv")
    elif cmldata is None:
        _warn_fallback_csv(base, "cml.data_v1 not available")
        df = _read_csv(base, f"{TABLE_TX}.csv")
    else:
        df = _try_warehouse(TABLE_TX)
        if df is None:
            _warn_fallback_csv(base, "warehouse empty or SELECT failed")
            df = _read_csv(base, f"{TABLE_TX}.csv")
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def load_supplier_shipping(data_dir: Optional[str] = None) -> pd.DataFrame:
    base = data_dir or DEFAULT_DATA_DIR
    if using_csv_files():
        df = _read_csv(base, f"{TABLE_SHIP}.csv")
    elif cmldata is None:
        _warn_fallback_csv(base, "cml.data_v1 not available")
        df = _read_csv(base, f"{TABLE_SHIP}.csv")
    else:
        df = _try_warehouse(TABLE_SHIP)
        if df is None:
            _warn_fallback_csv(base, "warehouse empty or SELECT failed")
            df = _read_csv(base, f"{TABLE_SHIP}.csv")
    if "report_month" in df.columns:
        df["report_month"] = pd.to_datetime(df["report_month"])
    return df


load_procurement_transactions = load_transactions
