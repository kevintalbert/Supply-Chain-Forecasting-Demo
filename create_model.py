#!/usr/bin/env python3
"""Deploy the forecasting model to Cloudera AI (cmlapi). Same credentials as Jobs: CDSW_API_URL, CDSW_APIV2_KEY, CDSW_PROJECT_ID."""

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


class Deployer:
    def __init__(self):
        self.host = os.getenv("CDSW_API_URL", "").replace("/api/v1", "").rstrip("/")
        self.api_key = os.getenv("CDSW_APIV2_KEY")
        self.project_id = os.getenv("CDSW_PROJECT_ID")
        if not all([self.host, self.api_key, self.project_id]):
            raise ValueError("Need CDSW_API_URL, CDSW_APIV2_KEY, CDSW_PROJECT_ID")
        self.client = cmlapi.default_client(url=self.host, cml_api_key=self.api_key)
        self.runtime_id = os.getenv(
            "CML_RUNTIME_ID",
            "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.13-standard:2026.04.1-b7",
        )
        self.body = {
            "name": "supply-chain-forecast-api",
            "description": "Dense + sparse price forecasts (sklearn HistGradientBoosting)",
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
            "cdsw-build.sh",
            "utils/forecasting_pipeline.py",
            "utils/data_access.py",
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
            cmlapi.CreateModelBuildRequest(
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
