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

Training and **`model_api`** read Impala ``logistics`` tables via **`utils/data_access.py`** when **`cml.data_v1`** is available; **if tables are empty or reads fail, data loads automatically from CSV** under **`data/raw/`** or **`LOGISTICS_DATA_DIR`** (no warehouse writes). Force CSV-only with **`LOGISTICS_DATA_SOURCE=csv`**. **`python load_logistics_data.py`** prints Impala row counts only (read-only).

## Quickstart

On **Cloudera AI**, keep the three CSVs in **`data/raw/`** (or **`LOGISTICS_DATA_DIR`**) for automatic fallback when warehouse tables have no rows. Omit **`LOGISTICS_DATA_SOURCE`** to try Impala **`SELECT`** first (**no warehouse writes**).

```bash
cd Supply-Chain-Forecasting-Demo
pip install -r requirements.txt

# Optional: force CSV-only reads (skip Impala):
export LOGISTICS_DATA_SOURCE=csv
export LOGISTICS_DATA_DIR=/path/to/folder_with_three_csvs   # or rely on data/raw/

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

Deploy with **`create_model.py`** inside Cloudera AI (requires `CDSW_API_URL`, `CDSW_APIV2_KEY`, `CDSW_PROJECT_ID`). Schedule training with **`create_training_job.py`** / **`submit_experiment_jobs.py`** (same credentials).

## CML API v2: Jobs (training) vs Models (serving)

| Capability | What exists in this repo | Credentials |
|------------|--------------------------|-------------|
| **Jobs API** — schedule / trigger training (`main.py`) | **`create_training_job.py`** creates a Job + optional run; **`submit_experiment_jobs.py`** creates multiple Jobs with different **`EXPERIMENT_NAME`** env vars | `CDSW_API_URL`, `CDSW_APIV2_KEY`, `CDSW_PROJECT_ID` |
| **Models API** — HTTP deployment of `model_api.predict` | **`create_model.py`** builds and deploys the registered model | Same env vars + optional `CML_RUNTIME_ID` |
| **Experiments** | Training writes **`models/forecasting_metadata.json`** (MAEs, etc.). Set **`EXPERIMENT_NAME`** on the Job environment so each run tags metadata; compare Job runs or artifacts in the UI | — |

Training picks **`DENSE_DEMO_NSN`** from the environment when set (defaults unchanged).

Before **Models**: run **`python main.py --all`** (or a Job that runs it) so **`models/`** contains artifacts the served model loads.

## Notebooks

Open `notebooks/supply_chain_forecasting_walkthrough.ipynb` for a narrated walkthrough of training metrics, sparse vs dense plots, and RAG fusion — suitable to wire to **CDV dashboards** or Apps serving the deployed model.

## Cloudera AI touchpoints

- **Workbench / Jobs**: **`create_training_job.py`** / **`submit_experiment_jobs.py`** (API v2); or run **`main.py`** manually.  
- **Model Registry / Serving**: **`create_model.py`** + **`model_api.py`** (`predict`).  
- **Data Warehouse**: **`utils/data_access.py`** reads Impala `logistics` or falls back to CSV (no writes); **`load_logistics_data.py`** row-count checks only.  
- **Experiments**: **`EXPERIMENT_NAME`** env → **`forecasting_metadata.json`**; Job comparison via **`submit_experiment_jobs.py`** or separate Job definitions.

Synthetic mock data only; safe for public demos.
