#!/usr/bin/env python3
"""
Optional: verify read access to existing ``logistics`` warehouse tables.

Training and model serving read the same tables via ``utils/data_access.py``
(``SELECT * FROM logistics.<table>``). This script only prints row counts — it does
not create tables or load CSVs.

Example::

    import cml.data_v1 as cmldata

    CONNECTION_NAME = "default-impala-aws"
    conn = cmldata.get_connection(CONNECTION_NAME)
    conn.get_pandas_dataframe("USE logistics;")
    dataframe = conn.get_pandas_dataframe(
        "SELECT COUNT(*) AS n FROM logistics.procurement_transactions"
    )
    print(dataframe)
    conn.close()

Env: ``LOGISTICS_IMPALA_CONN`` (default ``default-impala-aws``),
``LOGISTICS_DATABASE`` (default ``logistics``).
"""

from __future__ import annotations

import os
from datetime import datetime

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


def main():
    if cmldata is None:
        raise SystemExit("cml.data_v1 not available — run inside Cloudera AI Workbench")

    print(f"Database: {DATABASE_NAME}  connection: {CONNECTION_NAME}")
    conn = cmldata.get_connection(CONNECTION_NAME)
    try:
        conn.get_pandas_dataframe(f"USE {DATABASE_NAME};")
        for t in TABLES:
            df = conn.get_pandas_dataframe(
                f"SELECT COUNT(*) AS row_count FROM {DATABASE_NAME}.{t}"
            )
            n = df.iloc[0, 0] if len(df) else "?"
            print(f"  {t}: {n} rows")
        print("OK at", datetime.now().isoformat())
    finally:
        conn.close()


if __name__ == "__main__":
    main()
