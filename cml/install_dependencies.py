#!/usr/bin/env python3
# ###########################################################################
# CML “Install Dependencies” job (same role as Cloudera AMP MLflow Tracking
# `cml/install_dependencies.py`). Runs a full `pip install -r requirements.txt`
# for Workbench / Jobs. Model *image* builds use `cdsw-build.sh` (often with
# `CDSW_REQUIREMENTS_PROFILE=model` at project level from `.project-metadata.yaml`).
# ###########################################################################
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "requirements.txt"


def main() -> None:
    if not REQ.is_file():
        print(f"Missing {REQ}", file=sys.stderr)
        sys.exit(1)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        cwd=str(ROOT),
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", str(REQ)],
        cwd=str(ROOT),
    )
    print("cml/install_dependencies.py: installed", REQ)


if __name__ == "__main__":
    main()
