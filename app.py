#!/usr/bin/env python3
"""
Visual MLOps explorer for the Supply Chain Forecasting demo.

Calls your **deployed** model HTTP endpoint (production path) or **model_api.predict**
locally when no URL is set (development path).

**Local parity** (optional LSTM + richer RAG): install the full stack, then train so ``models/``
exists::

  pip install -r requirements.txt
  streamlit run app.py

The deployed CML model uses a **minimal** image (no TensorFlow / torch); local ``predict`` can
still load ``dense_lstm.keras`` or MiniLM-based RAG indexes when those artifacts exist and the
full packages are installed.

On Cloudera AI, set the prediction URL from the deployed model’s **Invoke** / API tab::

  export CML_MODEL_PREDICT_URL="https://<host>/.../predict"
  export CDSW_APIV2_KEY="<api-key>"   # if the endpoint requires a bearer token

Some gateways wrap the JSON body; this app unwraps common ``prediction`` / ``result`` keys.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "utils") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "utils"))

from forecasting_pipeline import DENSE_DEMO_NSN, SPARSE_DEMO_NSN

DEFAULT_MODEL_URL_ENV = "CML_MODEL_PREDICT_URL"


def _unwrap_gateway_payload(body: Any) -> Dict[str, Any]:
    if not isinstance(body, dict):
        return {"error": "unexpected response shape", "raw": body}
    for key in ("prediction", "result", "data", "output"):
        if key in body and isinstance(body[key], dict):
            return body[key]
    return body


def call_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke deployed endpoint or local ``predict``."""
    use_local = st.session_state.get("use_local", True)
    url = (st.session_state.get("model_url") or "").strip()

    if use_local or not url:
        from model_api import predict as model_predict

        return model_predict(payload)

    headers = {"Content-Type": "application/json"}
    token = (st.session_state.get("api_key") or "").strip() or os.getenv(
        "CDSW_APIV2_KEY", ""
    )
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=180)
        r.raise_for_status()
        return _unwrap_gateway_payload(r.json())
    except requests.RequestException as e:
        return {"error": str(e), "hint": "Check CML_MODEL_PREDICT_URL and CDSW_APIV2_KEY"}


def _dense_chart_df(result: Dict[str, Any]) -> pd.DataFrame | None:
    models = result.get("models") or {}
    arima = (models.get("arima") or {}).get("forecast") or []
    gbm = (models.get("gradient_boosting") or {}).get("forecast") or []
    if not arima and not gbm:
        return None
    rows: Dict[str, list] = {"month": [], "ARIMA": [], "Gradient boosting": []}
    for i, a in enumerate(arima):
        m = a.get("month", "")
        rows["month"].append(m[:10] if m else f"t+{i+1}")
        rows["ARIMA"].append(float(a.get("forecast_price", 0)))
        rows["Gradient boosting"].append(
            float(gbm[i]["forecast_price"]) if i < len(gbm) else float("nan")
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(
        page_title="Supply Chain · MLOps Explorer",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Supply Chain Forecasting · MLOps Explorer")
    st.caption(
        "Interact with the **deployed** CAI model (or run **locally**) to see training artifacts, "
        "dense vs sparse strategies, and RAG fusion."
    )

    if "model_url" not in st.session_state:
        st.session_state.model_url = os.getenv(DEFAULT_MODEL_URL_ENV, "")
    if "use_local" not in st.session_state:
        st.session_state.use_local = not bool(st.session_state.model_url.strip())

    with st.sidebar:
        st.header("Connection")
        st.session_state.use_local = st.toggle(
            "Use local model_api (no HTTP)",
            value=st.session_state.use_local,
            help="Uses ``model_api.predict`` in this project. Requires ``models/`` "
            "and running from the repo root.",
        )
        st.session_state.model_url = st.text_input(
            "Deployed model predict URL",
            value=st.session_state.model_url,
            placeholder="https://…/predict",
            disabled=st.session_state.use_local,
            help=f"Paste from the model’s API / Invoke page. Env: {DEFAULT_MODEL_URL_ENV}",
        )
        st.session_state.api_key = st.text_input(
            "API key (optional)",
            type="password",
            value=os.getenv("CDSW_APIV2_KEY", ""),
            help="Often CDSW_APIV2_KEY if the gateway expects Bearer auth.",
        )

        if st.button("Health check"):
            with st.spinner("Calling predict …"):
                hc = call_model({"action": "health"})
            st.json(hc)

    # --- Overview ---
    with st.expander("What am I looking at? (MLOps map)", expanded=True):
        st.markdown(
            """
1. **Train** (Job or ``main.py``) writes artifacts to ``models/`` — that is **offline** ML.
2. **Deploy** (``create_model.py``) packages ``model_api.predict`` — that is **serving**.
3. **This app** is a **client**: it sends JSON to the **same** ``predict`` contract your dashboard would use.
4. **Dense forecast** = classic monthly series → ARIMA + gradient boosting + optional LSTM.
5. **Sparse forecast** = rare purchases → boosting on **gap** and market features (not monthly ARIMA).
6. **Explain spike** = **RAG** on the contract PDF + **structured** Impala/CSV context.
            """
        )

    tab_dense, tab_sparse, tab_rag = st.tabs(
        [
            "Dense forecasts (continuous data)",
            "Sparse demand (rare buys)",
            "RAG + structured spike",
        ]
    )

    with tab_dense:
        st.subheader("Predictive modeling — dense NSN")
        st.info(
            "Demonstrates **ARIMA**, **gradient boosting** (HistGradientBoosting), and optional "
            "**LSTM** on a lubricant line with regular history."
        )
        c1, c2 = st.columns(2)
        with c1:
            nsn_dense = st.text_input("NSN (dense)", value=DENSE_DEMO_NSN, key="nsn_d")
        with c2:
            horizon = st.slider("Forecast horizon (months)", 3, 24, 6)

        if st.button("Run dense forecast", key="btn_dense"):
            with st.spinner("forecast_dense …"):
                out = call_model(
                    {
                        "action": "forecast_dense",
                        "nsn": nsn_dense.strip(),
                        "horizon_months": horizon,
                    }
                )
            if out.get("error") and "models" not in out:
                st.error(out.get("error"))
                st.json(out)
            else:
                chart_df = _dense_chart_df(out)
                if chart_df is not None:
                    st.line_chart(
                        chart_df.set_index("month")[["ARIMA", "Gradient boosting"]]
                    )
                lstm = (out.get("models") or {}).get("lstm_next_step")
                if lstm is not None:
                    st.metric("LSTM next-step forecast (scaled pipeline)", f"{lstm:.4f}")
                elif lstm is None:
                    st.caption("LSTM omitted if TensorFlow model not trained or not loaded.")
                st.json(out)

    with tab_sparse:
        st.subheader('The "sparse data" problem')
        st.info(
            "Items bought every few **years** break monthly ARIMA/LSTM. We use **HistGradientBoosting** "
            "on **gap_days**, prices, market moves, and supplier KPIs."
        )
        nsn_sp = st.text_input("NSN (sparse)", value=SPARSE_DEMO_NSN, key="nsn_s")

        if st.button("Run sparse forecast", key="btn_sparse"):
            with st.spinner("forecast_sparse …"):
                out = call_model({"action": "forecast_sparse", "nsn": nsn_sp.strip()})
            if out.get("error"):
                st.error(out["error"])
                st.json(out)
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Gap (days) between last two observations",
                    out.get("gap_days_between_last_observations", "—"),
                )
                c2.metric("Last observed price", f"{out.get('last_observed_price', 0):.4f}")
                c3.metric(
                    "Forecast next purchase price",
                    f"{out.get('forecast_next_purchase_price', 0):.4f}",
                )
                st.markdown(f"**Strategy:** {out.get('sparse_strategy', '')}")
                st.markdown(f"**Why:** {out.get('why', '')}")
                st.json(out)

    with tab_rag:
        st.subheader("Unstructured + structured fusion (RAG)")
        st.info(
            "Retrieves **contract clauses** (PDF index) for your query and joins **warehouse-style** "
            "price/order facts for the spike month."
        )
        c1, c2 = st.columns(2)
        with c1:
            nsn_r = st.text_input("NSN", value=DENSE_DEMO_NSN, key="nsn_r")
            cid = st.text_input("Contract ID", value="CON-7781")
        with c2:
            spike_m = st.text_input(
                "Spike month (optional YYYY-MM-DD)",
                placeholder="leave blank for largest recent MoM jump",
            )
        rag_q = st.text_area(
            "RAG query",
            value="energy surcharge escalation lubricant index quarterly price adjustment",
            height=90,
        )

        if st.button("Explain spike", key="btn_rag"):
            payload: Dict[str, Any] = {
                "action": "explain_spike",
                "nsn": nsn_r.strip(),
                "contract_id": cid.strip(),
                "rag_query": rag_q.strip(),
            }
            if spike_m.strip():
                payload["spike_month"] = spike_m.strip()
            with st.spinner("explain_spike …"):
                out = call_model(payload)
            if out.get("error"):
                st.error(str(out.get("error")))
                st.json(out)
            else:
                st.markdown("### Fusion summary")
                st.write(out.get("fusion_summary", ""))
                st.markdown("### Structured context")
                st.json(out.get("structured_context", {}))
                st.markdown("### Retrieved contract clauses")
                for i, ch in enumerate(out.get("retrieved_contract_clauses") or [], 1):
                    with st.expander(f"Chunk {i} (score {ch.get('score', 0):.4f})"):
                        st.text(ch.get("text", ""))
                with st.expander("Raw JSON"):
                    st.json(out)


if __name__ == "__main__":
    main()
