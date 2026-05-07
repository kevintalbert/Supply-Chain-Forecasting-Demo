"""
Forecasting with two sklearn models (mirrors the churn AMP pattern: train → joblib → predict).

- **Dense:** monthly lubricant-style series → HistGradientBoostingRegressor (rolling forecast).
- **Sparse:** irregular purchases → HistGradientBoostingRegressor on gap/market/supplier features.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore", category=FutureWarning)

from data_access import load_price_history, load_supplier_shipping

DENSE_DEMO_NSN = "9150-01-123-4567"
SPARSE_DEMO_NSN = "4820-00-111-2222"


@dataclass
class DenseTrainResult:
    nsn: str
    gbm_mae: float
    n_points: int


def train_gradient_boosting_dense(
    df_item: pd.DataFrame, n_lags: int = 6
) -> Tuple[HistGradientBoostingRegressor, List[str], float]:
    """Predict next month price from lags + demand + market + calendar month."""
    g = df_item.sort_values("date").reset_index(drop=True)
    price = g["unit_price"].values.astype(float)
    X_rows = []
    y = []
    feat_names = [f"lag_{k}" for k in range(1, n_lags + 1)] + [
        "demand_quantity",
        "market_index",
        "month",
    ]
    for i in range(n_lags, len(price)):
        lags = price[i - n_lags : i]
        row = np.concatenate(
            [
                lags,
                np.array(
                    [
                        float(g.loc[i, "demand_quantity"]),
                        float(g.loc[i, "market_index"]),
                        float(pd.Timestamp(g.loc[i, "date"]).month),
                    ]
                ),
            ]
        )
        X_rows.append(row)
        y.append(price[i])
    X = np.array(X_rows)
    y = np.array(y)
    if len(y) < 12:
        raise ValueError("Not enough history for gradient boosting.")
    split = int(len(y) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = HistGradientBoostingRegressor(
        max_depth=5,
        learning_rate=0.08,
        max_iter=200,
        random_state=42,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))
    return model, feat_names, mae


def build_sparse_training_frame(
    price_df: pd.DataFrame,
    ship_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    ship_df = ship_df.sort_values(["supplier_id", "report_month"])
    for nsn, grp in price_df.groupby("nsn"):
        g = grp.sort_values("date").reset_index(drop=True)
        if len(g) < 2:
            continue
        density = g["data_density_class"].iloc[0]
        supplier_id = g["supplier_id"].iloc[0]
        sh = ship_df[ship_df["supplier_id"] == supplier_id]
        for i in range(1, len(g)):
            prev = g.iloc[i - 1]
            cur = g.iloc[i]
            gap_days = (cur["date"] - prev["date"]).days
            lead_roll = (
                float(sh["avg_lead_time_days"].tail(3).mean())
                if len(sh)
                else np.nan
            )
            exp_roll = (
                float(sh["expedite_rate"].tail(3).mean()) if len(sh) else np.nan
            )
            rows.append(
                {
                    "nsn": nsn,
                    "density": density,
                    "gap_days": gap_days,
                    "log_gap": np.log1p(gap_days),
                    "prev_price": prev["unit_price"],
                    "market_index": cur["market_index"],
                    "demand_quantity": cur["demand_quantity"],
                    "delta_market": cur["market_index"] - prev["market_index"],
                    "target_price": cur["unit_price"],
                    "supplier_avg_lead_3m": lead_roll,
                    "supplier_expedite_3m": exp_roll,
                }
            )
    return pd.DataFrame(rows)


def train_sparse_gbm(frame: pd.DataFrame) -> Tuple[HistGradientBoostingRegressor, List[str], float]:
    sub = frame[frame["density"] == "sparse_multi_year"].dropna()
    feat_cols = [
        "log_gap",
        "prev_price",
        "market_index",
        "demand_quantity",
        "delta_market",
        "supplier_avg_lead_3m",
        "supplier_expedite_3m",
    ]
    if len(sub) < 8:
        raise ValueError("Not enough sparse rows for GBM demo.")
    X = sub[feat_cols].values
    y = sub["target_price"].values
    model = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.06,
        max_iter=200,
        min_samples_leaf=2,
        l2_regularization=0.5,
        random_state=42,
    )
    split = max(1, int(len(y) * 0.75))
    model.fit(X[:split], y[:split])
    pred = model.predict(X[split:])
    mae = float(mean_absolute_error(y[split:], pred)) if len(pred) else 0.0
    return model, feat_cols, mae


def predict_next_sparse(
    model: HistGradientBoostingRegressor,
    feat_cols: List[str],
    last_row: Dict[str, Any],
) -> float:
    X = np.array([[last_row[c] for c in feat_cols]])
    return float(model.predict(X)[0])


def iterative_gbm_forecast(
    dense_history: pd.DataFrame,
    gbm_bundle: Dict[str, Any],
    horizon: int = 6,
) -> List[Dict[str, Any]]:
    g = dense_history.sort_values("date").reset_index(drop=True)
    model: HistGradientBoostingRegressor = gbm_bundle["model"]
    n_lags: int = gbm_bundle.get("n_lags", 6)
    feat_names: List[str] = gbm_bundle["feature_names"]
    price = g["unit_price"].tolist()
    last_date = pd.Timestamp(g["date"].iloc[-1])
    last_demand = float(g["demand_quantity"].iloc[-1])
    last_market = float(g["market_index"].iloc[-1])
    out = []
    for h in range(1, horizon + 1):
        lags = np.array(price[-n_lags:], dtype=float)
        next_month = (last_date + pd.DateOffset(months=h)).month
        row = np.concatenate(
            [lags, np.array([last_demand, last_market, float(next_month)])]
        )
        pred = float(model.predict(row.reshape(1, -1))[0])
        out.append(
            {
                "step": h,
                "forecast_price": pred,
                "reference_month": str((last_date + pd.DateOffset(months=h)).date()),
            }
        )
        price.append(pred)
    return out


def run_training(
    data_dir: Optional[str],
    models_dir: str,
    dense_nsn: str = DENSE_DEMO_NSN,
) -> Dict[str, Any]:
    os.makedirs(models_dir, exist_ok=True)
    price_df = load_price_history(data_dir)
    ship_df = load_supplier_shipping(data_dir)

    dense = price_df[price_df["nsn"] == dense_nsn].copy()
    if dense.empty:
        raise ValueError(f"No rows for dense NSN {dense_nsn}")

    gbm_model, gbm_features, gbm_mae = train_gradient_boosting_dense(dense)

    sparse_frame = build_sparse_training_frame(price_df, ship_df)
    sparse_model, sparse_feats, sparse_mae = train_sparse_gbm(sparse_frame)

    joblib.dump(
        {
            "model": gbm_model,
            "feature_names": gbm_features,
            "n_lags": 6,
        },
        os.path.join(models_dir, "dense_gbm.joblib"),
    )
    joblib.dump(
        {"model": sparse_model, "feature_names": sparse_feats},
        os.path.join(models_dir, "sparse_gbm.joblib"),
    )
    joblib.dump(sparse_frame, os.path.join(models_dir, "sparse_training_frame.joblib"))

    meta = DenseTrainResult(nsn=dense_nsn, gbm_mae=gbm_mae, n_points=int(len(dense)))

    summary = {
        "dense_nsn": dense_nsn,
        "dense_points": meta.n_points,
        "dense_gbm_mae_holdout": gbm_mae,
        "sparse_gbm_mae_holdout": sparse_mae,
        "models_written": ["dense_gbm.joblib", "sparse_gbm.joblib"],
    }
    exp = os.environ.get("EXPERIMENT_NAME")
    if exp:
        summary["experiment_name"] = exp
    with open(os.path.join(models_dir, "forecasting_metadata.json"), "w") as f:
        json.dump(summary, f, indent=2)

    try:
        from cml_experiments import log_training_run_mlflow

        log_training_run_mlflow(summary, models_dir)
    except Exception as e:
        print(f"MLflow / Experiments logging skipped: {e}")

    return summary
