#!/usr/bin/env python3
"""
Supply Chain Forecasting Demo — train forecasting models + contract RAG index.

Usage:
  python main.py --train
  python main.py --build-contract-pdf
  python main.py --build-rag-index
  python main.py --all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UTILS_PATH = os.path.join(PROJECT_ROOT, "utils")
sys.path.insert(0, UTILS_PATH)

DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def ensure_sample_data():
    """Copy CSVs into data/raw when using local CSV mode (see utils/data_access.py)."""
    from data_access import using_csv_files

    if not using_csv_files():
        return
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
    from contract_rag import build_index
    from forecasting_pipeline import run_training

    ensure_sample_data()
    os.makedirs(MODELS_DIR, exist_ok=True)
    summary = run_training(data_dir=DATA_RAW, models_dir=MODELS_DIR)
    print("Training summary:", summary)

    pdf = os.path.join(
        PROJECT_ROOT, "contracts", "CON-7781_Turbine_Oil_Supply_Agreement.pdf"
    )
    if os.path.exists(pdf):
        build_index(pdf, MODELS_DIR)
        print("RAG index built from contract PDF.")
    else:
        print("Skipping RAG index — run: python scripts/build_mock_contract_pdf.py")


def build_pdf():
    subprocess.check_call(
        [
            sys.executable,
            os.path.join(PROJECT_ROOT, "scripts", "build_mock_contract_pdf.py"),
        ]
    )


def build_rag_only():
    from contract_rag import build_index

    pdf = os.path.join(
        PROJECT_ROOT, "contracts", "CON-7781_Turbine_Oil_Supply_Agreement.pdf"
    )
    if not os.path.exists(pdf):
        build_pdf()
    os.makedirs(MODELS_DIR, exist_ok=True)
    build_index(pdf, MODELS_DIR)
    print("RAG index ready.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--build-contract-pdf", action="store_true")
    parser.add_argument("--build-rag-index", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    run_default = args.all or not any(
        [args.train, args.build_contract_pdf, args.build_rag_index]
    )

    if run_default:
        build_pdf()
        train()
        return

    if args.build_contract_pdf:
        build_pdf()
    if args.train:
        train()
    if args.build_rag_index:
        build_rag_only()


if __name__ == "__main__":
    main()
