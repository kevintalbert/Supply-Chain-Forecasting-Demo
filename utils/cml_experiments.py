"""

Cloudera AI Experiments (v2) via the MLflow Tracking API.



Runs appear under **Project → Experiments** when MLflow is installed.

"""



from __future__ import annotations



import os

from typing import Any, Dict



_DEFAULT_EXPERIMENT = "Supply Chain Forecasting"





def log_training_run_mlflow(summary: Dict[str, Any], models_dir: str) -> bool:

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

        mlflow.log_param("dense_points", int(summary.get("dense_points") or 0))



        for key in ("dense_gbm_mae_holdout", "sparse_gbm_mae_holdout"):

            val = summary.get(key)

            if val is not None:

                mlflow.log_metric(key, float(val))



        meta_path = os.path.join(models_dir, "forecasting_metadata.json")

        if os.path.isfile(meta_path):

            mlflow.log_artifact(meta_path, artifact_path="metadata")



    print(

        f"Experiments: logged MLflow run under experiment {exp_name!r} "

        f"(Project → Experiments)."

    )

    return True

