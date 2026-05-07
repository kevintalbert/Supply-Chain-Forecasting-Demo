"""
CML model entrypoint — two forecasting modes (dense GBM, sparse GBM). Same pattern as churn AMP:
load joblib artifacts and return a dict from ``predict``.
"""

from __future__ import annotations

import os
import time
import traceback
from datetime import datetime
from typing import Any, Dict

import joblib
import pandas as pd

try:
    os.chdir("code")
except Exception:
    pass

try:
    import cml.models_v1 as models
    import cml.metrics_v1 as metrics

    CML_AVAILABLE = True
except ImportError:

    class _M:
        def cml_model(self, metrics=True):
            def deco(fn):
                return fn

            return deco

    class _Met:
        def track_metric(self, k, v):
            pass

    models = _M()
    metrics = _Met()
    CML_AVAILABLE = False

MODEL_PATH = os.path.join("..", "models")
DATA_RAW = os.path.join("..", "data", "raw")

from data_access import load_price_history, load_supplier_shipping
from forecasting_pipeline import (
    DENSE_DEMO_NSN,
    SPARSE_DEMO_NSN,
    iterative_gbm_forecast,
    predict_next_sparse,
)

dense_gbm_bundle = None
sparse_bundle = None
init_time = None


def initialize_model():
    global dense_gbm_bundle, sparse_bundle, init_time
    t0 = time.time()
    dense_gbm_bundle = joblib.load(os.path.join(MODEL_PATH, "dense_gbm.joblib"))
    sparse_bundle = joblib.load(os.path.join(MODEL_PATH, "sparse_gbm.joblib"))
    init_time = time.time() - t0


if __name__ == "__main__" or os.getenv("CDSW_ENGINE_TYPE") == "model":
    initialize_model()


def _data_dir():
    return DATA_RAW if os.path.isdir(DATA_RAW) else None


def handle_forecast_dense(args: Dict[str, Any]) -> Dict[str, Any]:
    nsn = args.get("nsn", DENSE_DEMO_NSN)
    horizon = int(args.get("horizon_months", 6))
    price_df = load_price_history(_data_dir())
    sub = price_df[price_df["nsn"] == nsn].sort_values("date")
    if sub.empty:
        return {"error": f"No series for NSN {nsn}"}
    forecast = iterative_gbm_forecast(sub, dense_gbm_bundle, horizon=horizon)
    return {
        "nsn": nsn,
        "model": "hist_gradient_boosting_regressor",
        "forecast": forecast,
        "last_history_month": str(sub["date"].max().date()),
    }


def handle_forecast_sparse(args: Dict[str, Any]) -> Dict[str, Any]:
    nsn = args.get("nsn", SPARSE_DEMO_NSN)
    price_df = load_price_history(_data_dir())
    ship_df = load_supplier_shipping(_data_dir())
    sub = price_df[price_df["nsn"] == nsn].sort_values("date")
    if len(sub) < 2:
        return {"error": "need sparse history"}
    prev = sub.iloc[-2]
    cur = sub.iloc[-1]
    gap_days = int((cur["date"] - prev["date"]).days)
    supplier_id = cur["supplier_id"]
    sh = ship_df[ship_df["supplier_id"] == supplier_id]
    last_row = {
        "log_gap": float(__import__("numpy").log1p(gap_days)),
        "prev_price": float(prev["unit_price"]),
        "market_index": float(cur["market_index"]),
        "demand_quantity": float(cur["demand_quantity"]),
        "delta_market": float(cur["market_index"] - prev["market_index"]),
        "supplier_avg_lead_3m": float(sh["avg_lead_time_days"].tail(3).mean())
        if len(sh)
        else 15.0,
        "supplier_expedite_3m": float(sh["expedite_rate"].tail(3).mean())
        if len(sh)
        else 0.05,
    }
    pred = predict_next_sparse(
        sparse_bundle["model"], sparse_bundle["feature_names"], last_row
    )
    return {
        "nsn": nsn,
        "model": "hist_gradient_boosting_regressor",
        "last_observed_price": float(cur["unit_price"]),
        "forecast_next_purchase_price": pred,
        "gap_days_between_last_observations": gap_days,
    }


@models.cml_model(metrics=True)
def predict(args: Dict[str, Any]) -> Dict[str, Any]:
    start = time.time()
    try:
        action = args.get("action", "forecast_dense")
        metrics.track_metric("action", action)
        if dense_gbm_bundle is None:
            initialize_model()

        if action == "forecast_dense":
            out = handle_forecast_dense(args)
        elif action == "forecast_sparse":
            out = handle_forecast_sparse(args)
        elif action == "health":
            out = {
                "status": "ok",
                "cml": CML_AVAILABLE,
                "init_time_s": init_time,
                "models_loaded": dense_gbm_bundle is not None,
            }
        else:
            out = {"error": f"unknown action {action}"}

        out["prediction_timestamp"] = datetime.now().isoformat()
        out["response_time_ms"] = round((time.time() - start) * 1000, 2)
        return out
    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "response_time_ms": round((time.time() - start) * 1000, 2),
        }


def health_check():
    return predict({"action": "health"})
