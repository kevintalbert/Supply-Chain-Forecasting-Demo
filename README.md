# Supply Chain Forecasting (Cloudera AI)

Small demo modeled after **[CML_AMP_Churn_Prediction](https://github.com/cloudera/CML_AMP_Churn_Prediction)**:

1. **Train** two `sklearn.ensemble.HistGradientBoostingRegressor` models on procurement CSVs.
2. **Deploy** `code/model_api.predict` as a CML Model.

No RAG, no TensorFlow/LSTM, no extra requirements profiles — **`cdsw-build.sh`** is only `pip3 install -r requirements.txt`.

---

## Models

| Artifact | Use case |
|----------|----------|
| `models/dense_gbm.joblib` | Regular monthly-style demand for one NSN — rolling multi-step forecast |
| `models/sparse_gbm.joblib` | Rare buys — predict **next** purchase price from gap + market + supplier features |

---

## Quickstart (Workbench)

```bash
pip install -r requirements.txt
python code/main.py     # writes models/*.joblib from data/raw/*.csv
python create_model.py  # needs CDSW_API_URL, CDSW_APIV2_KEY, CDSW_PROJECT_ID
```

Optional UI:

```bash
streamlit run code/app.py
```

---

## Model API (`predict`)

JSON body is a **single dict**. Actions:

- **`health`** — `{"action": "health"}`
- **`forecast_dense`** — `{"action": "forecast_dense", "nsn": "9150-01-123-4567", "horizon_months": 6}`
- **`forecast_sparse`** — `{"action": "forecast_sparse", "nsn": "4820-00-111-2222"}`

---

## Data

Place the three CSVs under **`data/raw/`** (or set **`LOGISTICS_DATA_DIR`** to a folder that contains them). See `code/load_logistics_data.py` / `code/data_access.py` for schema.

---

## Troubleshooting

- **`ModuleNotFoundError` on the replica:** build installs **`requirements.txt`** via **`cdsw-build.sh`** — ensure that file is synced to the project.
- **Registry push errors:** treat as platform/registry issues; this repo keeps dependencies minimal on purpose.
