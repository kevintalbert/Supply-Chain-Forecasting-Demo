#!/usr/bin/env python3
"""Simple Streamlit client for the forecasting model (local ``predict`` or deployed HTTP URL)."""

from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st

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


def _forecast_chart_rows(result: Dict[str, Any]) -> pd.DataFrame | None:
    fc = result.get("forecast") or []
    if not fc:
        return None
    return pd.DataFrame(
        {
            "month": [x.get("reference_month", "")[:10] for x in fc],
            "forecast_price": [float(x.get("forecast_price", 0)) for x in fc],
        }
    )


def main() -> None:
    st.set_page_config(page_title="Supply Chain Forecasting", layout="wide")
    st.title("Supply Chain Forecasting")
    st.caption("Two **HistGradientBoostingRegressor** models: dense monthly series vs sparse intermittent buys.")

    if "model_url" not in st.session_state:
        st.session_state.model_url = os.getenv(DEFAULT_MODEL_URL_ENV, "")
    if "use_local" not in st.session_state:
        st.session_state.use_local = not bool(st.session_state.model_url.strip())

    with st.sidebar:
        st.header("Connection")
        st.session_state.use_local = st.toggle(
            "Use local model_api",
            value=st.session_state.use_local,
        )
        st.session_state.model_url = st.text_input(
            "Deployed predict URL",
            value=st.session_state.model_url,
            disabled=st.session_state.use_local,
        )
        st.session_state.api_key = st.text_input(
            "API key (optional)",
            type="password",
            value=os.getenv("CDSW_APIV2_KEY", ""),
        )
        if st.button("Health check"):
            st.json(call_model({"action": "health"}))

    tab_d, tab_s = st.tabs(["Dense forecast (regular buys)", "Sparse forecast (rare buys)"])

    with tab_d:
        st.subheader("Dense NSN — rolling multi-step forecast")
        nsn = st.text_input("NSN", value=DENSE_DEMO_NSN, key="dnsn")
        horizon = st.slider("Horizon (months)", 3, 24, 6)
        if st.button("Run"):
            out = call_model(
                {
                    "action": "forecast_dense",
                    "nsn": nsn.strip(),
                    "horizon_months": horizon,
                }
            )
            if out.get("error") and "forecast" not in out:
                st.error(out.get("error"))
            else:
                df = _forecast_chart_rows(out)
                if df is not None:
                    st.line_chart(df.set_index("month"))
                st.json(out)

    with tab_s:
        st.subheader("Sparse NSN — next purchase price")
        nsn_s = st.text_input("NSN", value=SPARSE_DEMO_NSN, key="snsn")
        if st.button("Run sparse forecast", key="rb"):
            out = call_model({"action": "forecast_sparse", "nsn": nsn_s.strip()})
            if out.get("error"):
                st.error(out["error"])
            else:
                st.metric("Forecast next price", f"{out.get('forecast_next_purchase_price', 0):.4f}")
                st.json(out)


if __name__ == "__main__":
    main()
