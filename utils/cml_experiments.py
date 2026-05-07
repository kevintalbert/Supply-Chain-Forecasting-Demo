"""
Cloudera AI Experiments (v2) via the MLflow Tracking API.

See:
https://docs.cloudera.com/machine-learning/1.5.5/experiments/topics/ml-exp-v2-tracking.html

Runs created here appear under **Project → Experiments** in the CML UI when using ML Runtimes.
"""

from __future__ import annotations

import os
from typing import Any, Dict

_DEFAULT_EXPERIMENT = "Supply Chain Forecasting"


def log_training_run_mlflow(summary: Dict[str, Any], models_dir: str) -> bool:
    """
    Log holdout metrics and ``forecasting_metadata.json`` to the active Cloudera experiment store.

    Environment:
      ``MLFLOW_EXPERIMENT_NAME`` — experiment name (created if missing). Default: Supply Chain Forecasting
      ``EXPERIMENT_NAME`` or ``MLFLOW_RUN_NAME`` — run name in the UI (optional)
      ``MLFLOW_DISABLE`` — set to 1 / true to skip (e.g. quick local test)
    """
    if os.environ.get("MLFLOW_DISABLE", "").lower() in ("1", "true", "yes"):
        return False
    try:
        import mlflow
    except ImportError:
        return False

    exp_name = (os.environ.get("MLFLOW_EXPERIMENT_NAME") or "").strip() or _DEFAULT_EXPERIMENT
    run_name = (
        (os.environ.get("EXPERIMENT_NAME") or "").strip()
        or (os.environ.get("MLFLOW_RUN_NAME") or "").strip()
        or None
    )

    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("dense_nsn", str(summary.get("dense_nsn", "")))
        if summary.get("experiment_name"):
            mlflow.log_param("experiment_name_label", str(summary["experiment_name"]))
        mlflow.log_param("sparse_strategy", str(summary.get("sparse_strategy", "")))
        mlflow.log_param("dense_points", int(summary.get("dense_points") or 0))

        for key in (
            "arima_mae_holdout",
            "gbm_mae_holdout",
            "sparse_gbm_mae_holdout",
        ):
            val = summary.get(key)
            if val is not None:
                mlflow.log_metric(key, float(val))

        lstm = summary.get("lstm_mae_holdout")
        if lstm is not None:
            mlflow.log_metric("lstm_mae_holdout", float(lstm))

        meta_path = os.path.join(models_dir, "forecasting_metadata.json")
        if os.path.isfile(meta_path):
            mlflow.log_artifact(meta_path, artifact_path="metadata")

    print(
        f"Experiments: logged MLflow run under experiment {exp_name!r} "
        f"(open Project → Experiments in Cloudera AI to trace)."
    )
    return True
