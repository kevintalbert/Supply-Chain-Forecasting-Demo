#!/usr/bin/env python3
"""
Create (and optionally run) a Cloudera AI **Job** via API v2 (`cmlapi`) that executes ``code/main.py``.

Uses the same credentials as ``create_model.py``: ``CDSW_API_URL``, ``CDSW_APIV2_KEY``, ``CDSW_PROJECT_ID``.

ML Runtime projects **require** a runtime image on the Job: set ``CML_RUNTIME_ID`` (same as ``create_model.py``), ``CML_JOB_RUNTIME_ID``, pass ``--runtime``, or rely on the built-in default Workbench Python 3.13 runtime string.

Examples (Workbench session):

  python create_training_job.py --run
  python create_training_job.py --name my-retrain --arguments "--train" --run

Environment variables placed on the **job definition** (not just your shell) so scheduled reruns keep them:

  EXPERIMENT_NAME=baseline python create_training_job.py --run

See ``submit_experiment_jobs.py`` for multiple labeled experiment jobs.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

try:
    import cmlapi
except ImportError:
    print("cmlapi not installed — run inside Cloudera AI Workbench")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ML Runtime projects require a kernel image on Job creation (same as model builds).
_DEFAULT_ML_RUNTIME = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.13-standard:2026.04.1-b7"

def _resolve_runtime_id(explicit: str | None) -> str:
    return (
        (explicit or "").strip()
        or (os.getenv("CML_RUNTIME_ID") or "").strip()
        or (os.getenv("CML_JOB_RUNTIME_ID") or "").strip()
        or _DEFAULT_ML_RUNTIME
    )


def _client_and_project():
    host = os.getenv("CDSW_API_URL", "").replace("/api/v1", "").rstrip("/")
    api_key = os.getenv("CDSW_APIV2_KEY")
    project_id = os.getenv("CDSW_PROJECT_ID")
    if not all([host, api_key, project_id]):
        raise ValueError("Need CDSW_API_URL, CDSW_APIV2_KEY, CDSW_PROJECT_ID")
    return cmlapi.default_client(url=host, cml_api_key=api_key), project_id


def build_job_request(
    *,
    name: str,
    script: str,
    arguments: str,
    cpu: float,
    memory: float,
    timeout: int,
    environment: dict[str, str] | None,
    runtime_identifier: str | None,
) -> "cmlapi.CreateJobRequest":
    req = cmlapi.CreateJobRequest()
    req.name = name
    req.script = script
    req.arguments = arguments
    req.kernel = "python3"
    req.cpu = cpu
    req.memory = memory
    req.timeout = timeout
    if environment:
        req.environment = environment
    if not runtime_identifier:
        raise ValueError("runtime_identifier is required for ML Runtime projects")
    req.runtime_identifier = runtime_identifier
    return req


def create_job_definition(
    *,
    name: str,
    script: str = "code/main.py",
    arguments: str = "--all",
    cpu: float | None = None,
    memory: float | None = None,
    timeout: int | None = None,
    environment: dict[str, str] | None = None,
    runtime_identifier: str | None = None,
) -> str:
    """Create a Job in the project; return ``job_id``."""
    client, project_id = _client_and_project()
    cpu = cpu if cpu is not None else float(os.getenv("CML_JOB_CPU", "2"))
    memory = memory if memory is not None else float(os.getenv("CML_JOB_MEMORY", "8"))
    timeout = timeout if timeout is not None else int(os.getenv("CML_JOB_TIMEOUT_SEC", "7200"))
    rt = _resolve_runtime_id(runtime_identifier)
    logger.info("Job ML Runtime: %s", rt)

    body = build_job_request(
        name=name,
        script=script,
        arguments=arguments,
        cpu=cpu,
        memory=memory,
        timeout=timeout,
        environment=environment,
        runtime_identifier=rt,
    )
    job = client.create_job(body, project_id)
    jid = getattr(job, "id", None) or str(job)
    logger.info("Created job %r id=%s", name, jid)
    return str(jid)


def start_job_run(job_id: str) -> str:
    """Start a one-off run for an existing job; return run id."""
    client, project_id = _client_and_project()
    body = cmlapi.CreateJobRunRequest()
    run = client.create_job_run(body, project_id, job_id)
    rid = getattr(run, "id", None) or str(run)
    logger.info("Started job run id=%s", rid)
    return str(rid)


def _collect_env_from_shell(keys: tuple[str, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k in keys:
        v = os.environ.get(k)
        if v:
            out[k] = v
    return out


def main():
    p = argparse.ArgumentParser(description="CML Jobs API: create training job for code/main.py")
    p.add_argument("--name", default="supply-chain-forecast-train", help="Job display name")
    p.add_argument("--script", default="code/main.py", help="Project file to execute")
    p.add_argument("--arguments", default="--all", help="CLI args passed to the script")
    p.add_argument("--cpu", type=float, default=None)
    p.add_argument("--memory", type=float, default=None)
    p.add_argument("--timeout", type=int, default=None, help="Seconds")
    p.add_argument("--run", action="store_true", help="Submit a run immediately after create")
    p.add_argument("--run-only", metavar="JOB_ID", help="Only start a run; do not create")
    p.add_argument(
        "--runtime",
        default=None,
        help="ML Runtime image id (defaults to CML_RUNTIME_ID, then CML_JOB_RUNTIME_ID, then Workbench Python 3.11 runtime)",
    )
    args = p.parse_args()

    if args.run_only:
        start_job_run(args.run_only)
        return

    env_keys = (
        "EXPERIMENT_NAME",
        "MLFLOW_EXPERIMENT_NAME",
        "MLFLOW_RUN_NAME",
        "MLFLOW_DISABLE",
        "DENSE_DEMO_NSN",
        "LOGISTICS_DATA_SOURCE",
        "LOGISTICS_DATA_DIR",
        "LOGISTICS_DATABASE",
        "LOGISTICS_IMPALA_CONN",
    )
    env = _collect_env_from_shell(env_keys)

    jid = create_job_definition(
        name=args.name,
        script=args.script,
        arguments=args.arguments,
        cpu=args.cpu,
        memory=args.memory,
        timeout=args.timeout,
        environment=env if env else None,
        runtime_identifier=args.runtime,
    )
    if args.run:
        start_job_run(jid)


if __name__ == "__main__":
    main()
