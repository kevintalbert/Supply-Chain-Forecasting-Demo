#!/usr/bin/env python3
"""
Read-only check: row counts for the three ``logistics`` Impala tables.

Does **not** INSERT, TRUNCATE, or otherwise modify the warehouse. Training and the model API
load data via ``utils/data_access.py`` (warehouse SELECT with automatic fallback to CSV).

Env: ``LOGISTICS_IMPALA_CONN``, ``LOGISTICS_DATABASE``.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_UTILS_DIR = os.path.join(_SCRIPT_DIR, "utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

try:
    import cml.data_v1 as cmldata
except ImportError:
    cmldata = None

CONNECTION_NAME = os.environ.get("LOGISTICS_IMPALA_CONN", "default-impala-aws")
DATABASE_NAME = os.environ.get("LOGISTICS_DATABASE", "logistics")

TABLES = (
    "supplier_shipping_performance",
    "item_price_history_forecasting",
    "procurement_transactions",
)


def _count_rows(conn, table: str) -> int:
    df = conn.get_pandas_dataframe(
        f"SELECT COUNT(*) AS row_count FROM {DATABASE_NAME}.{table}"
    )
    n = df.iloc[0, 0] if len(df) else 0
    try:
        return int(n)
    except (TypeError, ValueError):
        return 0


def _print_zero_diagnostics(conn) -> None:
    print("\n--- Diagnostics: all counts zero ---")
    try:
        show_df = conn.get_pandas_dataframe(f"SHOW TABLES IN {DATABASE_NAME}")
        if show_df is not None and len(show_df):
            print(f"Tables in `{DATABASE_NAME}`:")
            print(show_df.to_string(index=False))
        print(
            "``utils/data_access.py`` will fall back to CSV under LOGISTICS_DATA_DIR / data/raw "
            "when warehouse tables are empty."
        )
    except Exception as e:
        print(f"SHOW TABLES failed: {e}")
    print("---\n")


def main() -> None:
    if cmldata is None:
        raise SystemExit("cml.data_v1 not available — run inside Cloudera AI Workbench")

    print(f"Database: {DATABASE_NAME}  connection: {CONNECTION_NAME}")
    conn = cmldata.get_connection(CONNECTION_NAME)
    try:
        counts = []
        for t in TABLES:
            n = _count_rows(conn, t)
            counts.append(n)
            print(f"  {t}: {n} rows")
        if sum(counts) == 0:
            _print_zero_diagnostics(conn)
        print("OK at", datetime.now().isoformat())
    finally:
        conn.close()


if __name__ == "__main__":
    main()
