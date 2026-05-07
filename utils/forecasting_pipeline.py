"""
Dense time series: ARIMA + scikit-learn Gradient Boosting + optional LSTM.

Sparse / intermittent demand: feature-based gradient boosting with calendar gaps.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore", category=FutureWarning)

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_UTILS_DIR)
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from data_access import load_price_history, load_supplier_shipping  # noqa: E402

DENSE_DEMO_NSN = "9150-01-123-4567"
SPARSE_DEMO_NSN = "4820-00-111-2222"


@dataclass
class DenseTrainResult:
    nsn: str
    arima_order: Tuple[int, int, int]
    arima_mae: float
    gbm_mae: float
    lstm_mae: Optional[float]
    n_points: int


def _lags(series: np.ndarray, n_lags: int) -> np.ndarray:
    rows = []
    for i in range(n_lags, len(series)):
        rows.append(series[i - n_lags : i])
    return np.array(rows)


def train_arima(series: pd.Series, order: Tuple[int, int, int] = (2, 1, 1)):
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(series.astype(float), order=order)
    return model.fit()


def forecast_arima(fitted, steps: int) -> np.ndarray:
    return fitted.forecast(steps=steps).values


def train_gradient_boosting_dense(
    df_item: pd.DataFrame, n_lags: int = 6
) -> Tuple[HistGradientBoostingRegressor, List[str], float]:
    """Direct multi-step style: predict next month price from lags + exogenous."""
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
        raise ValueError("Not enough history for gradient boosting benchmark.")
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


def train_lstm_dense(
    df_item: pd.DataFrame, seq_len: int = 12
) -> Tuple[Any, Optional[float]]:
    """Small univariate LSTM on scaled prices; returns (keras_model, holdout_mae)."""
    try:
        import tensorflow as tf
        from tensorflow import keras
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        return None, None

    g = df_item.sort_values("date").reset_index(drop=True)
    price = g["unit_price"].values.reshape(-1, 1).astype(float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(price)
    Xs, ys = [], []
    for i in range(seq_len, len(scaled)):
        Xs.append(scaled[i - seq_len : i, 0])
        ys.append(scaled[i, 0])
    X_arr = np.array(Xs).reshape(-1, seq_len, 1)
    y_arr = np.array(ys)
    if len(y_arr) < 16:
        return None, None
    split = int(len(y_arr) * 0.85)
    X_train, X_val = X_arr[:split], X_arr[split:]
    y_train, y_val = y_arr[:split], y_arr[split:]

    tf.keras.utils.set_random_seed(42)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(seq_len, 1)),
            keras.layers.LSTM(32, return_sequences=False),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=8,
        verbose=0,
    )
    pred_s = model.predict(X_val, verbose=0).flatten()
    true_prices = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    pred_prices = scaler.inverse_transform(pred_s.reshape(-1, 1)).flatten()
    mae = float(mean_absolute_error(true_prices, pred_prices))
    # attach scaler for inference
    model._demo_scaler = scaler  # type: ignore[attr-defined]
    model._demo_seq_len = seq_len  # type: ignore[attr-defined]
    return model, mae


def build_sparse_training_frame(
    price_df: pd.DataFrame,
    ship_df: pd.DataFrame,
) -> pd.DataFrame:
    """Pool irregular series into rows with gap-based features."""
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
    # LOOCV-style small sample: simple holdout last 20%
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

    series = dense.set_index("date")["unit_price"].asfreq("MS").interpolate()
    arima_fit = train_arima(series)
    arima_holdout = max(3, min(6, len(series) // 5))
    train_s = series.iloc[:-arima_holdout]
    test_s = series.iloc[-arima_holdout:]
    arima_fit_eval = train_arima(train_s)
    arima_pred = forecast_arima(arima_fit_eval, steps=len(test_s))
    arima_mae = float(mean_absolute_error(test_s.values, arima_pred))

    gbm_model, gbm_features, gbm_mae = train_gradient_boosting_dense(dense)
    lstm_model, lstm_mae = train_lstm_dense(dense)

    sparse_frame = build_sparse_training_frame(price_df, ship_df)
    sparse_model, sparse_feats, sparse_mae = train_sparse_gbm(sparse_frame)

    joblib.dump(arima_fit, os.path.join(models_dir, "dense_arima_fit.joblib"))
    joblib.dump(
        {
            "model": gbm_model,
            "feature_names": gbm_features,
            "n_lags": 6,
        },
        os.path.join(models_dir, "dense_gbm.joblib"),
    )
    if lstm_model is not None:
        lstm_path = os.path.join(models_dir, "dense_lstm.keras")
        lstm_model.save(lstm_path)
        joblib.dump(
            {
                "scaler": lstm_model._demo_scaler,  # type: ignore[attr-defined]
                "seq_len": lstm_model._demo_seq_len,  # type: ignore[attr-defined]
            },
            os.path.join(models_dir, "dense_lstm_meta.joblib"),
        )

    joblib.dump(
        {"model": sparse_model, "feature_names": sparse_feats},
        os.path.join(models_dir, "sparse_gbm.joblib"),
    )
    joblib.dump(sparse_frame, os.path.join(models_dir, "sparse_training_frame.joblib"))

    meta = DenseTrainResult(
        nsn=dense_nsn,
        arima_order=(2, 1, 1),
        arima_mae=arima_mae,
        gbm_mae=gbm_mae,
        lstm_mae=lstm_mae,
        n_points=int(len(dense)),
    )

    summary = {
        "dense_nsn": dense_nsn,
        "dense_points": meta.n_points,
        "arima_mae_holdout": arima_mae,
        "gbm_mae_holdout": gbm_mae,
        "lstm_mae_holdout": lstm_mae,
        "sparse_gbm_mae_holdout": sparse_mae,
        "sparse_strategy": "gradient_boosting_on_gap_and_market_features",
        "models_written": [
            "dense_arima_fit.joblib",
            "dense_gbm.joblib",
            "sparse_gbm.joblib",
        ]
        + (
            ["dense_lstm.keras", "dense_lstm_meta.joblib"]
            if lstm_model is not None
            else []
        ),
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


def iterative_gbm_forecast(
    dense_history: pd.DataFrame,
    gbm_bundle: Dict[str, Any],
    horizon: int = 6,
) -> List[Dict[str, Any]]:
    """Rolling one-step GBM forecasts using last known demand/market when unknown."""
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


def lstm_forecast_next(
    dense_history: pd.DataFrame,
    keras_model,
    lstm_meta: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    if keras_model is None:
        return None
    scaler = None
    seq_len = 12
    if lstm_meta:
        scaler = lstm_meta.get("scaler")
        seq_len = int(lstm_meta.get("seq_len", 12))
    if scaler is None:
        scaler = getattr(keras_model, "_demo_scaler", None)
    if scaler is None:
        return None
    seq_len = int(getattr(keras_model, "_demo_seq_len", seq_len))
    g = dense_history.sort_values("date").reset_index(drop=True)
    price = g["unit_price"].values.reshape(-1, 1).astype(float)
    scaled = scaler.transform(price)
    if len(scaled) < seq_len:
        return None
    seq = scaled[-seq_len:, 0].reshape(1, seq_len, 1)
    pred_s = keras_model.predict(seq, verbose=0)[0, 0]
    inv = scaler.inverse_transform([[float(pred_s)]])[0, 0]
    return float(inv)
