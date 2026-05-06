"""
Cloudera AI model entrypoint: procurement price forecasting + contract RAG explanations.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

import joblib
import pandas as pd

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

if os.path.exists("/home/cdsw"):
    PROJECT_PATH = os.getenv("CDSW_PROJECT_PATH", "/home/cdsw")
    MODEL_PATH = os.getenv("CDSW_MODEL_PATH", "/home/cdsw/models")
else:
    PROJECT_PATH = os.getcwd()
    MODEL_PATH = os.path.join(PROJECT_PATH, "models")

UTILS_PATH = os.path.join(PROJECT_PATH, "utils")
DATA_RAW = os.path.join(PROJECT_PATH, "data", "raw")
sys.path.insert(0, UTILS_PATH)

from contract_rag import (  # noqa: E402
    load_index,
    retrieve,
    structured_price_context,
)
from data_access import (  # noqa: E402
    load_price_history,
    load_procurement_transactions,
    load_supplier_shipping,
)
from forecasting_pipeline import (  # noqa: E402
    DENSE_DEMO_NSN,
    SPARSE_DEMO_NSN,
    forecast_arima,
    iterative_gbm_forecast,
    lstm_forecast_next,
    predict_next_sparse,
)

dense_arima = None
dense_gbm_bundle = None
dense_lstm = None
dense_lstm_meta = None
sparse_bundle = None
rag_index = None
init_time = None


def _load_keras(path: str):
    try:
        from tensorflow import keras

        return keras.models.load_model(path)
    except Exception:
        return None


def initialize_model():
    global dense_arima, dense_gbm_bundle, dense_lstm, dense_lstm_meta
    global sparse_bundle, rag_index, init_time
    t0 = time.time()
    dense_arima = joblib.load(os.path.join(MODEL_PATH, "dense_arima_fit.joblib"))
    dense_gbm_bundle = joblib.load(os.path.join(MODEL_PATH, "dense_gbm.joblib"))
    sparse_bundle = joblib.load(os.path.join(MODEL_PATH, "sparse_gbm.joblib"))
    lstm_path = os.path.join(MODEL_PATH, "dense_lstm.keras")
    meta_path = os.path.join(MODEL_PATH, "dense_lstm_meta.joblib")
    dense_lstm = _load_keras(lstm_path) if os.path.exists(lstm_path) else None
    dense_lstm_meta = joblib.load(meta_path) if os.path.exists(meta_path) else None
    rag_index = load_index(MODEL_PATH)
    init_time = time.time() - t0


if __name__ == "__main__" or os.getenv("CDSW_ENGINE_TYPE") == "model":
    initialize_model()


def _data_dir():
    return DATA_RAW if os.path.isdir(DATA_RAW) else None


def handle_forecast_dense(args: Dict[str, Any]) -> Dict[str, Any]:
    nsn = args.get("nsn", DENSE_DEMO_NSN)
    horizon = int(args.get("horizon_months", 6))
    price_df = load_price_history(_data_dir())
    sub = price_df[price_df["nsn"] == nsn]
    if sub.empty:
        return {"error": f"No series for NSN {nsn}"}
    series = sub.set_index("date")["unit_price"].asfreq("MS").interpolate()
    arima_future = forecast_arima(dense_arima, steps=horizon)
    arima_dates = pd.date_range(series.index.max() + pd.offsets.MonthBegin(), periods=horizon, freq="MS")
    arima_payload = [
        {"month": str(d.date()), "forecast_price": float(v)}
        for d, v in zip(arima_dates, arima_future)
    ]
    gbm_roll = iterative_gbm_forecast(sub, dense_gbm_bundle, horizon=horizon)
    lstm_next = lstm_forecast_next(sub, dense_lstm, dense_lstm_meta)
    return {
        "nsn": nsn,
        "models": {
            "arima": {"holdout_metric": "mae in forecasting_metadata.json", "forecast": arima_payload},
            "gradient_boosting": {"forecast": gbm_roll},
            "lstm_next_step": lstm_next,
        },
        "note": "ARIMA + gradient boosting + LSTM cover continuous monthly lubricant pricing.",
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
        "sparse_strategy": "HistGradientBoosting with gap_days and supplier context (not classical ARIMA)",
        "last_observed_price": float(cur["unit_price"]),
        "forecast_next_purchase_price": pred,
        "gap_days_between_last_observations": gap_days,
        "why": "Classical ARIMA/LSTM fail with multi-year gaps; CAI uses feature-based boosting "
        "pooling intermittent demand signals.",
    }


def handle_explain_spike(args: Dict[str, Any]) -> Dict[str, Any]:
    nsn = args.get("nsn", DENSE_DEMO_NSN)
    contract_id = args.get("contract_id", "CON-7781")
    spike_month = args.get("spike_month")  # optional YYYY-MM-DD
    price_df = load_price_history(_data_dir())
    tx_df = load_procurement_transactions(_data_dir())
    ctx = structured_price_context(
        nsn, contract_id, price_df, tx_df, spike_month=spike_month
    )
    if "error" in ctx:
        return ctx
    query = args.get(
        "rag_query",
        "energy surcharge escalation lubricant index quarterly price adjustment",
    )
    chunks_out = []
    if rag_index:
        hits = retrieve(query, rag_index, top_k=3)
        chunks_out = [{"score": h.score, "text": h.text} for h in hits]
    narrative = []
    if ctx.get("mom_pct"):
        narrative.append(
            f"Month-over-month unit price moved {ctx['mom_pct']:.1%} around {ctx['spike_month']}."
        )
    narrative.append(
        f"Recorded market_index={ctx.get('market_index')} and demand_quantity={ctx.get('demand_quantity')}."
    )
    if chunks_out:
        narrative.append(
            "Contract retrieval highlights surcharge clauses that may explain pass-through when indices jump."
        )
    return {
        "structured_context": ctx,
        "retrieved_contract_clauses": chunks_out,
        "fusion_summary": " ".join(narrative),
        "rag_query": query,
    }


@models.cml_model(metrics=True)
def predict(args: Dict[str, Any]) -> Dict[str, Any]:
    start = time.time()
    try:
        action = args.get("action", "forecast_dense")
        metrics.track_metric("action", action)
        if dense_arima is None:
            initialize_model()

        if action == "forecast_dense":
            out = handle_forecast_dense(args)
        elif action == "forecast_sparse":
            out = handle_forecast_sparse(args)
        elif action == "explain_spike":
            out = handle_explain_spike(args)
        elif action == "health":
            out = {
                "status": "ok",
                "cml": CML_AVAILABLE,
                "init_time_s": init_time,
                "models_loaded": dense_arima is not None,
                "rag_loaded": rag_index is not None,
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
