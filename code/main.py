#!/usr/bin/env python3
"""Train two sklearn models and save artifacts under ``models/`` (see ``forecasting_pipeline.py``)."""

from __future__ import annotations

import argparse
import os

# Same as CML churn AMP: cwd is project root in sessions/jobs; modules live next to this file.
try:
    os.chdir("code")
except Exception:
    pass

from forecasting_pipeline import DENSE_DEMO_NSN, run_training

DATA_RAW = os.path.join("..", "data", "raw")
MODELS_DIR = os.path.join("..", "models")


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
