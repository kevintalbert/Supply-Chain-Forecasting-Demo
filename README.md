# Supply Chain Price Forecasting Demo (Cloudera AI)

Customer-ready companion to the Student Loan Risk demo pattern: **warehouse-backed procurement data**, **forecasting models that match real data challenges**, and **RAG over a mock contract** fused with structured pricing to narrate spikes.

This README is written so you can understand the project **without prior ML background**. Skip ahead using the table of contents.

### What this proves (at a glance)

| Ask | How it is shown |
|-----|-----------------|
| **Predictive modeling (ARIMA / LSTM / Gradient Boosting)** | Monthly **Turbine Oil** NSN `9150-01-123-4567`: **ARIMA**, **HistGradientBoostingRegressor** on lags + demand + market features, optional **TensorFlow LSTM**. |
| **Sparse / intermittent demand** | **Legacy Valve** NSN `4820-00-111-2222`: irregular timestamps → **HistGradientBoosting** with `gap_days`, market deltas, supplier KPIs. |
| **Unstructured + structured fusion (RAG)** | Mock PDF → chunked retrieval (**sentence-transformers** or TF-IDF fallback) → joined with warehouse-style price/order context. |

---

## Table of contents

1. [Big picture — two phases](#big-picture--two-phases-training-vs-serving)
2. [Machine learning in plain language](#machine-learning-in-plain-language)
3. [What each model type actually does](#what-each-model-type-actually-does)
4. [How the pieces connect](#how-the-pieces-connect)
5. [Data: warehouse vs CSV](#data-warehouse-vs-csv)
6. [Artifacts in `models/`](#artifacts-in-models)
7. [Quickstart (train locally or in Workbench)](#quickstart)
8. [Serving API (`model_api.predict`)](#model-serving-pipeline-behind-the-dashboard)
9. [Cloudera AI: Jobs vs Models](#cloudera-ai-jobs-training-vs-models-serving)
10. [Recommended order on CML](#recommended-order-on-cloudera-ai)
11. [Troubleshooting](#troubleshooting)
12. [Notebooks & touchpoints](#notebooks)

---

## Big picture — two phases (training vs serving)

| Phase | What runs | What it produces / uses |
|--------|-----------|-------------------------|
| **Training** | `main.py` → `utils/forecasting_pipeline.py`, optional PDF + `utils/contract_rag.py` | Files under **`models/`** (saved models, indexes, metadata) |
| **Serving** | **`model_api.predict`** (after you deploy with `create_model.py`) | Loads **`models/`**, answers HTTP/JSON requests (forecasts, explanations, health) |

**Training** = learn patterns from **past** data and write files to disk.  
**Serving** = load those files and answer **new** questions using **fresh** data where needed.

You must **train first** (so `models/` exists), then **deploy** the model. They solve different problems.

---

## Machine learning in plain language

- **Supervised learning:** show the computer many examples of inputs and the correct answer; it finds rules that **predict** the answer for new inputs. Here the “answers” are things like **next month’s price** or **price after a long gap**.
- **Time series:** data ordered by time (monthly prices). Some methods assume **regular** spacing (every month); others handle **irregular** gaps.
- **Artifacts:** after training, learned rules are saved as **files** (`*.joblib`, optional `*.keras`). At serving time, Python **loads** those files — it is not re-training on every API call.
- **RAG (retrieval-augmented generation):** the contract PDF is split into chunks and indexed. At query time, the system **retrieves** relevant chunks and combines them with **structured** price/order data so explanations cite **both** numbers and contract text.

Nothing here “thinks” like a human; it **fits patterns** from history and applies them with **math**.

---

## What each model type actually does

### Dense demand (example NSN: Turbine Oil `9150-01-123-4567`)

This part is ordered **often**, so there is a **long run of monthly-ish prices**.

Three strategies are trained on the same history:

| Model | Idea (non-technical) |
|--------|----------------------|
| **ARIMA** | “Prices tend to drift and wobble in a stable way.” Mostly uses **the price line over time** to extrapolate. |
| **Gradient boosting (HistGradientBoosting)** | “Guess next month’s price from **recent prices**, **order volume**, **market index**, and **calendar month**.” Learns weighted rules from examples. |
| **LSTM** (optional, needs TensorFlow) | Looks at the **last several months** of prices as a **sequence** and predicts the next step — useful when patterns are nonlinear. |

When you ask for a **dense forecast**, the API uses these saved models plus **current** price history from `data_access`.

### Sparse / intermittent demand (example: Legacy Valve `4820-00-111-2222`)

Some parts appear **rarely**, with **years** between purchases. A “every month” model is a poor fit.

Instead, each **gap between purchases** becomes one training row: days since last price, market change, supplier shipping stats, etc. A **single gradient boosting model** learns: “in situations like this, what price showed up next?” Predictions are **event-style**, not a full calendar of months.

### Contract RAG (PDF)

Separate from the numeric forecasts: text chunks from **`contracts/CON-7781_Turbine_Oil_Supply_Agreement.pdf`** are embedded (or TF-IDF fallback). **`explain_spike`** retrieves relevant clauses and joins them with structured context from your tables.

---

## How the pieces connect

```mermaid
flowchart LR
  subgraph data [Data]
    WH[(Impala logistics)]
    CSV[data/raw CSVs]
  end
  subgraph train [Training]
    main[main.py]
    fp[forecasting_pipeline]
    rag[contract_rag]
    mdir[models/ folder]
  end
  subgraph cml_api [Cloudera AI APIs]
    job[Job: create_training_job.py]
    deploy[create_model.py]
    api[model_api.predict]
  end
  WH --> da[data_access.py]
  CSV --> da
  da --> main
  main --> fp
  main --> rag
  fp --> mdir
  rag --> mdir
  job --> main
  mdir --> deploy
  deploy --> api
  da --> api
```

- **`utils/data_access.py`** is the **single gate** for procurement tables: try Impala first, fall back to CSV (read-only, **no warehouse writes**).
- **`main.py`** orchestrates training and optional PDF/RAG index build.
- **`model_api.py`** loads **`models/`** at startup and calls the same **`data_access`** helpers when a request needs fresh series.

---

## Data (warehouse vs CSV)

Database: **`logistics`**  

Tables (same names as CSV files without `.csv`):

- `supplier_shipping_performance`
- `item_price_history_forecasting`
- `procurement_transactions`

Training and **`model_api`** read Impala via **`cml.data_v1`** when available; **if tables are empty or reads fail**, data loads from **`data/raw/`** or **`LOGISTICS_DATA_DIR`**. Set **`LOGISTICS_DATA_SOURCE=csv`** to skip Impala. **`python load_logistics_data.py`** prints Impala row counts only (read-only).

---

## Artifacts in `models/`

After training you should see examples such as:

| File | Role |
|------|------|
| `dense_arima_fit.joblib` | Saved ARIMA model for the dense NSN |
| `dense_gbm.joblib` | Gradient boosting bundle (lags + features) |
| `dense_lstm.keras`, `dense_lstm_meta.joblib` | Optional LSTM + scaler metadata |
| `sparse_gbm.joblib` | Sparse-demand gradient boosting |
| `contract_rag_index.joblib` | RAG index (if PDF/index built) |
| `forecasting_metadata.json` | Metrics and experiment tag (`EXPERIMENT_NAME` if set) |

If Hugging Face downloads are blocked, **semantic embeddings fall back to TF-IDF** automatically.

---

## Quickstart

On **Cloudera AI**, keep the three CSVs in **`data/raw/`** (or **`LOGISTICS_DATA_DIR`**) when warehouse tables have no rows.

```bash
cd Supply-Chain-Forecasting-Demo
pip install -r requirements.txt

# Optional: force CSV-only reads (skip Impala):
export LOGISTICS_DATA_SOURCE=csv
export LOGISTICS_DATA_DIR=/path/to/folder_with_three_csvs   # or rely on data/raw/

python main.py --all          # builds PDF + trains + builds RAG index (if reportlab / deps available)
```

---

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

---

## Cloudera AI: Jobs (training) vs Models (serving)

| Capability | What exists in this repo | Credentials / runtime |
|------------|--------------------------|------------------------|
| **Jobs API** — run `main.py` on a schedule or on demand | **`create_training_job.py`**; **`submit_experiment_jobs.py`** for multiple **`EXPERIMENT_NAME`** values | `CDSW_API_URL`, `CDSW_APIV2_KEY`, `CDSW_PROJECT_ID`; **`CML_RUNTIME_ID`** or **`--runtime`** on ML Runtime projects |
| **Models API** — HTTP deployment of `model_api.predict` | **`create_model.py`** | Same API vars; **`CML_RUNTIME_ID`** should match the Workbench **Python runtime** you use for Jobs (e.g. Python 3.13 image) |
| **Experiments** | **`models/forecasting_metadata.json`** | Set **`EXPERIMENT_NAME`** on the Job environment |

**Jobs** automate **training**. **Models** expose **`predict`** over the network. Use **the same runtime image** for Jobs and model builds when possible so dependencies and Python versions align.

Training picks **`DENSE_DEMO_NSN`** from the environment when set.

**Before Models:** run **`python main.py --all`** (or a Job that runs it) so **`models/`** contains artifacts the served model loads.

---

## Recommended order on Cloudera AI

1. **Install dependencies** in the project (Workbench terminal): `pip install -r requirements.txt` — required for **deployment** (see [Troubleshooting](#troubleshooting)).
2. **Train:** run **`main.py`** interactively or **`create_training_job.py --run`**.
3. **Verify** `models/` contains the expected files.
4. **Deploy:** **`python create_model.py`** with **`CML_RUNTIME_ID`** set to your site’s ML Runtime if defaults differ.
5. **Test** the deployment URL with **`{"action":"health"}`** then a forecast payload.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'joblib'` (model fails to start)

The model runtime executes **`model_api.py`** in an ML Runtime kernel. That environment includes **base Python** but **not** every package from **`requirements.txt`** until they are installed for the **project**.

**Fix:**

1. In **Workbench**, open a terminal in **this project** and run:
   ```bash
   pip install -r requirements.txt
   ```
   Install at least **`joblib`**, **`pandas`**, **`numpy`**, **`scikit-learn`**, **`statsmodels`**, and any stack your site needs for RAG/LSTM (`pypdf`, optional `tensorflow`, `sentence-transformers`, etc.).

2. If your site uses **project-level dependency settings** (e.g. pinned packages in the UI), add the same dependencies there so **model replicas** see them.

3. **Redeploy** the model (new build) after dependencies are installed.

4. Use the **same `CML_RUNTIME_ID`** for Jobs and **`create_model.py`** so Python versions match.

Logs often show **`Finish start model: failed`** with an **`ename`** such as **`ModuleNotFoundError`**. Messages like **`use of closed network connection`** usually happen **after** the kernel exits because the import failed — fix the **first** Python error, not the websocket line.

### Job / notebook quirks (`__file__`, `ipykernel`, `reportlab`)

If **`main.py`** runs inside Jupyter-style execution, path and argparse quirks were handled in code (project root resolution, `parse_known_args`, optional PDF when **`reportlab`** is missing). Prefer **`python main.py`** from a terminal when possible.

### Runtime ID errors (`runtime ID must be specified`)

On **ML Runtime projects**, Jobs and model builds need **`runtime_identifier`**. Set **`CML_RUNTIME_ID`** to your environment’s image, or rely on the default in **`create_training_job.py`** / **`create_model.py`**.

### Model build: `failed to push ... s2i-registry ... blob upload invalid` / `unknown: unknown error`

The image **build** finished (`exporting layers` succeeded), but **pushing** to the cluster registry failed. That often happens when the image is **too large**: on Linux, **`sentence-transformers`** pulls **`torch`** from PyPI, which defaults to **CUDA builds** and drags in **multi‑gigabyte NVIDIA wheel layers**, on top of TensorFlow and (if present) Jupyter.

**Fix (recommended):**

1. On the **deployed model** (model build settings / environment variables), set:
   **`CDSW_REQUIREMENTS_PROFILE=model`**
2. Redeploy so **`cdsw-build.sh`** uses **`requirements-model.txt`** (slim) and installs **CPU-only PyTorch** before the rest.
3. Keep **`requirements.txt`** for Workbench sessions and **training Jobs** (full stack including Jupyter / Streamlit if you use them).

If it still fails after slimming the image, treat it as a **platform/infrastructure** issue (registry disk/quota, ingress body limits, known Harbor/registry bugs). Open a ticket with your platform team and attach the build log.

---

## Notebooks

Open `notebooks/supply_chain_forecasting_walkthrough.ipynb` for plots and RAG fusion — suitable for **CDV dashboards** or Apps calling the deployed model.

## Cloudera AI touchpoints (short)

- **Workbench / Jobs:** `create_training_job.py`, `submit_experiment_jobs.py`, or `main.py` manually.  
- **Model Registry / Serving:** `create_model.py` + `model_api.py` (`predict`).  
- **Warehouse:** `utils/data_access.py` (read-only); `load_logistics_data.py` for row counts.  
- **Experiments:** `EXPERIMENT_NAME` → `forecasting_metadata.json`.

Synthetic mock data only; safe for public demos.
