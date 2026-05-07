#!/usr/bin/env python3
"""Deploy supply-chain forecasting model to Cloudera AI (cmlapi Models API). See create_training_job.py for Jobs API."""

import logging
import os
import sys
import time
from datetime import datetime

try:
    import cmlapi
except ImportError:
    print("cmlapi not installed — run inside Cloudera AI")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _model_build_request(**fields):
    """CreateModelBuildRequest with slim-deps env when the installed cmlapi supports it."""
    try:
        return cmlapi.CreateModelBuildRequest(
            **fields,
            environment={"CDSW_REQUIREMENTS_PROFILE": "model"},
        )
    except TypeError:
        pass
    try:
        return cmlapi.CreateModelBuildRequest(
            **fields,
            build_environment={"CDSW_REQUIREMENTS_PROFILE": "model"},
        )
    except TypeError:
        pass
    logger.warning(
        "cmlapi.CreateModelBuildRequest has no environment field — set "
        "CDSW_REQUIREMENTS_PROFILE=model under Project environment variables "
        "(not your personal user profile; user env is not passed to model builds)."
    )
    return cmlapi.CreateModelBuildRequest(**fields)


class Deployer:
    def __init__(self):
        self.host = os.getenv("CDSW_API_URL", "").replace("/api/v1", "").rstrip("/")
        self.api_key = os.getenv("CDSW_APIV2_KEY")
        self.project_id = os.getenv("CDSW_PROJECT_ID")
        if not all([self.host, self.api_key, self.project_id]):
            raise ValueError("Need CDSW_API_URL, CDSW_APIV2_KEY, CDSW_PROJECT_ID")
        self.client = cmlapi.default_client(url=self.host, cml_api_key=self.api_key)
        # Keep in sync with create_training_job._DEFAULT_ML_RUNTIME (ML Runtime image for model build + replicas).
        self.runtime_id = os.getenv(
            "CML_RUNTIME_ID",
            "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.13-standard:2026.04.1-b7",
        )
        self.body = {
            "name": "supply-chain-price-forecast-api",
            "description": "Dense ARIMA+GBM+LSTM forecasts, sparse GBM, contract RAG spike explanations",
            "file_path": "model_api.py",
            "function_name": "predict",
            "kernel": "python3",
            "cpu": 2.0,
            "memory": 8,
            "nvidia_gpu": 0,
            "replicas": {"min": 1, "max": 2},
        }

    def files_ok(self) -> bool:
        req = [
            "model_api.py",
            "requirements.txt",
            "utils/forecasting_pipeline.py",
            "utils/data_access.py",
            "utils/contract_rag.py",
        ]
        bad = [f for f in req if not os.path.exists(f)]
        if bad:
            logger.error("Missing: %s", bad)
            return False
        return True

    def go(self) -> bool:
        if not self.files_ok():
            return False
        logger.info("Creating model...")
        m = self.client.create_model(
            cmlapi.CreateModelRequest(
                name=self.body["name"],
                description=self.body["description"],
                project_id=self.project_id,
                visibility="private",
                disable_authentication=True,
            ),
            self.project_id,
        )
        b = self.client.create_model_build(
            _model_build_request(
                project_id=self.project_id,
                model_id=m.id,
                file_path=self.body["file_path"],
                function_name=self.body["function_name"],
                kernel=self.body["kernel"],
                runtime_identifier=self.runtime_id,
                runtime_addon_identifiers=[],
                comment=f"auto {datetime.now().isoformat()}",
            ),
            self.project_id,
            m.id,
        )
        logger.info("Build %s — waiting...", b.id)
        while True:
            bb = self.client.get_model_build(self.project_id, m.id, b.id)
            st = bb.status.lower()
            if st == "built":
                break
            if "fail" in st:
                logger.error("Build failed: %s", getattr(bb, "failure_reason", ""))
                return False
            time.sleep(15)
        d = self.client.create_model_deployment(
            cmlapi.CreateModelDeploymentRequest(
                project_id=self.project_id,
                model_id=m.id,
                build_id=b.id,
                cpu=self.body["cpu"],
                memory=self.body["memory"],
                nvidia_gpus=self.body["nvidia_gpu"],
                replicas=self.body["replicas"]["min"],
            ),
            self.project_id,
            m.id,
            b.id,
        )
        logger.info("Deployment started: %s", d.id)
        return True


def main():
    Deployer().go()


if __name__ == "__main__":
    main()
