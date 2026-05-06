# Supply Chain Price Forecasting Demo (Cloudera AI)

Customer-ready companion to the Student Loan Risk demo pattern: **warehouse-backed procurement data**, **forecasting models that match real data challenges**, and **RAG over a mock contract** fused with structured pricing to narrate spikes.

## What this proves

| Ask | How it is shown |
|-----|-----------------|
| **Predictive modeling (ARIMA / LSTM / Gradient Boosting)** | Monthly **Turbine Oil** NSN `9150-01-123-4567`: `statsmodels` **ARIMA**, **HistGradientBoostingRegressor** on lags + demand + market features, optional **TensorFlow LSTM** next-step forecast. |
| **Sparse / intermittent demand** | **Legacy Valve** NSN `4820-00-111-2222` (and related sparse SKUs): irregular multi-year timestamps → **feature-based HistGradientBoosting** using `gap_days`, market deltas, and supplier shipping KPIs — avoids pretending ARIMA works on three observations per decade. |
| **Unstructured + structured fusion (RAG)** | Mock PDF `contracts/CON-7781_Turbine_Oil_Supply_Agreement.pdf` → chunked retrieval (**sentence-transformers** or TF-IDF fallback) → joined with **Impala-style** price/order context to explain a lubricant price spike (e.g., surcharge clause + rising `market_index`). |

## Data layout (your warehouse)

Database: **`logistics`**  

Tables (same names as CSV files without `.csv`):

- `supplier_shipping_performance`
- `item_price_history_forecasting`
- `procurement_transactions`

Training and **`model_api`** load these tables through **`utils/data_access.py`** via Impala (`SELECT * FROM logistics.<table>`) when **`cml.data_v1`** is available (default on Cloudera AI). For laptop runs without Impala, set **`LOGISTICS_DATA_SOURCE=csv`** and put the three CSVs under **`data/raw/`** (or point **`LOGISTICS_DATA_DIR`** at them). Optional smoke check: **`python load_logistics_data.py`** prints row counts only.

## Quickstart

On **Cloudera AI**, omit **`LOGISTICS_DATA_SOURCE`** so reads use **`cml.data_v1`** against **`logistics`** (no project CSVs needed).

```bash
cd Supply-Chain-Forecasting-Demo
pip install -r requirements.txt

# Local CSV fallback only:
export LOGISTICS_DATA_SOURCE=csv
export LOGISTICS_DATA_DIR=/path/to/folder_with_three_csvs   # or copy into data/raw/

python main.py --all          # builds PDF + trains + builds RAG index
```

Artifacts land in `models/` (`dense_arima_fit.joblib`, `dense_gbm.joblib`, optional `dense_lstm.keras` + `dense_lstm_meta.joblib`, `sparse_gbm.joblib`, `contract_rag_index.joblib`, `forecasting_metadata.json`).

If Hugging Face downloads are blocked, **semantic embeddings fall back to TF-IDF** automatically (still demonstrably RAG: chunk → retrieve → fuse).

## Model serving (pipeline behind the dashboard)

Entry point: **`model_api.predict`**

Example payloads:

```json
{"action": "forecast_dense", "nsn": "9150-01-123-4567", "horizon_months": 6}
```

```json
{"action": "forecast_sparse", "nsn": "4820-00-111-2222"}
```

```json
{"action": "explain_spike", "nsn": "9150-01-123-4567", "contract_id": "CON-7781", "rag_query": "energy surcharge index lubricant"}
```

```json
{"action": "health"}
```

Deploy with `create_model.py` inside Cloudera AI (requires `CDSW_API_URL`, `CDSW_APIV2_KEY`, `CDSW_PROJECT_ID`).

## Notebooks

Open `notebooks/supply_chain_forecasting_walkthrough.ipynb` for a narrated walkthrough of training metrics, sparse vs dense plots, and RAG fusion — suitable to wire to **CDV dashboards** or Apps serving the deployed model.

## Cloudera AI touchpoints

- **Workbench / Jobs**: `main.py` training job; scheduled retrains on fresh warehouse extracts.  
- **Model Registry / Serving**: `model_api.py` multi-action API used by dashboards.  
- **Data Warehouse**: Impala `logistics` database via **`utils/data_access.py`** (`cml.data_v1`); optional **`load_logistics_data.py`** to verify row counts.  
- **Experiments**: Track `forecasting_metadata.json` MAEs per build.

Synthetic mock data only; safe for public demos.
