#!/usr/bin/env python3
"""Train two sklearn models and save artifacts under ``models/`` (see ``utils/forecasting_pipeline.py``)."""

from __future__ import annotations

import argparse
import os
import sys


def _project_root() -> str:
    for key in ("SUPPLY_CHAIN_PROJECT_ROOT", "CDSW_PROJECT_ROOT"):
        override = (os.environ.get(key) or "").strip()
        if override and os.path.isdir(override):
            return os.path.abspath(override)
    try:
        here = __file__
    except NameError:
        here = None
    if here:
        return os.path.dirname(os.path.abspath(here))
    # Notebooks / interactive: no __file__; CML Workbench cwd is /home/cdsw.
    return "/home/cdsw"


PROJECT_ROOT = _project_root()
UTILS_PATH = os.path.join(PROJECT_ROOT, "utils")
sys.path.insert(0, UTILS_PATH)

DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def ensure_sample_data():
    os.makedirs(DATA_RAW, exist_ok=True)
    names = [
        "supplier_shipping_performance.csv",
        "item_price_history_forecasting.csv",
        "procurement_transactions.csv",
    ]
    have_all = all(os.path.exists(os.path.join(DATA_RAW, n)) for n in names)
    src_base = os.environ.get("LOGISTICS_DATA_DIR")
    if not src_base:
        if not have_all:
            print(
                "LOGISTICS_DATA_DIR not set — place the three procurement CSV files in data/raw/"
            )
        return
    for n in names:
        dst = os.path.join(DATA_RAW, n)
        if os.path.exists(dst):
            continue
        src = os.path.join(src_base, n)
        if os.path.exists(src):
            import shutil

            shutil.copy2(src, dst)
            print(f"Copied {n} -> data/raw/")
        else:
            print(f"Source missing: {src}")


def train():
    from forecasting_pipeline import DENSE_DEMO_NSN, run_training

    ensure_sample_data()
    os.makedirs(MODELS_DIR, exist_ok=True)
    dense_nsn = os.environ.get("DENSE_DEMO_NSN", DENSE_DEMO_NSN)
    summary = run_training(data_dir=DATA_RAW, models_dir=MODELS_DIR, dense_nsn=dense_nsn)
    print("Training summary:", summary)


def main():
    parser = argparse.ArgumentParser(description="Train dense + sparse forecasting models.")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.parse_known_args()
    train()


if __name__ == "__main__":
    main()
