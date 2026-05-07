#!/usr/bin/env python3
# ###########################################################################
# CML “Install Dependencies” job — `pip install -r requirements.txt` for Workbench sessions.
# Model builds use the same `requirements.txt` via `cdsw-build.sh`.
# ###########################################################################
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    for key in ("SUPPLY_CHAIN_PROJECT_ROOT", "CDSW_PROJECT_ROOT"):
        override = (os.environ.get(key) or "").strip()
        if override and os.path.isdir(override):
            return Path(override).resolve()
    try:
        here = __file__
    except NameError:
        here = None
    if here:
        return Path(here).resolve().parents[1]
    # Notebooks / interactive: no __file__; CML Workbench project lives under /home/cdsw.
    return Path("/home/cdsw").resolve()


ROOT = _project_root()
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
