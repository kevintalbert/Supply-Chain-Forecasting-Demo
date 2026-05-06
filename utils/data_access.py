"""Load procurement datasets from CSV (local) or optional Impala."""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

DEFAULT_DATA_DIR = os.environ.get(
    "LOGISTICS_DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw"),
)


def _resolve(path: Optional[str], filename: str) -> str:
    base = path or DEFAULT_DATA_DIR
    return os.path.join(base, filename)


def load_price_history(data_dir: Optional[str] = None) -> pd.DataFrame:
    path = _resolve(data_dir, "item_price_history_forecasting.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["nsn", "date"])


def load_transactions(data_dir: Optional[str] = None) -> pd.DataFrame:
    path = _resolve(data_dir, "procurement_transactions.csv")
    df = pd.read_csv(path)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


load_procurement_transactions = load_transactions


def load_supplier_shipping(data_dir: Optional[str] = None) -> pd.DataFrame:
    path = _resolve(data_dir, "supplier_shipping_performance.csv")
    df = pd.read_csv(path)
    df["report_month"] = pd.to_datetime(df["report_month"])
    return df
