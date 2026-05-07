#!/usr/bin/env python3
"""
Create several **training jobs** that differ by ``EXPERIMENT_NAME`` (and optional overrides).

Each job writes ``models/forecasting_metadata.json`` with ``experiment_name`` when training finishes.
Training also logs an **MLflow** run to **Project → Experiments** (see ``utils/cml_experiments.py``,
``MLFLOW_EXPERIMENT_NAME``, ``EXPERIMENT_NAME``).

Compare runs in the **Experiments** UI, the CML Jobs UI, or by downloading artifacts from each job run.

Edit ``EXPERIMENTS`` below, then run inside Workbench::

    python submit_experiment_jobs.py

Requires ``CDSW_API_URL``, ``CDSW_APIV2_KEY``, ``CDSW_PROJECT_ID`` (same as ``create_training_job.py``).

**Note:** All runs target the same ``models/`` directory in the project workspace unless you use
separate branches/projects — typical pattern is sequential runs or separate projects per experiment.
"""

from __future__ import annotations

import sys

try:
    import cmlapi  # noqa: F401 - ensure same env as create_training_job
except ImportError:
    print("cmlapi not installed — run inside Cloudera AI Workbench")
    sys.exit(1)

from create_training_job import create_job_definition, start_job_run

# (job_name_suffix, environment dict embedded in the Job definition)
EXPERIMENTS: list[tuple[str, dict[str, str]]] = [
    ("baseline", {"EXPERIMENT_NAME": "baseline"}),
    ("scenario-b", {"EXPERIMENT_NAME": "scenario_b"}),
]


def main():
    base = "supply-chain-train"
    for suffix, env in EXPERIMENTS:
        name = f"{base}-{suffix}"
        jid = create_job_definition(
            name=name,
            arguments="--all",
            environment=env,
        )
        start_job_run(jid)


if __name__ == "__main__":
    main()
