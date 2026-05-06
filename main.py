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


def _project_root() -> str:
    """Resolve repo root; ``__file__`` is missing in some notebook / job runners.

    Override with ``SUPPLY_CHAIN_PROJECT_ROOT`` or ``CDSW_PROJECT_ROOT`` if needed.
    """
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

    argv0 = sys.argv[0] if sys.argv else ""
    if argv0 and argv0 not in ("-", "<stdin>"):
        abs0 = os.path.abspath(argv0)
        if os.path.isfile(abs0):
            parts = abs0.replace("\\", "/").split("/")
            base = os.path.basename(abs0)
            # Jupyter: argv0 is site-packages/ipykernel_launcher.py — not the project
            if base != "ipykernel_launcher.py" and "site-packages" not in parts:
                return os.path.dirname(abs0)

    return os.path.abspath(os.getcwd())


PROJECT_ROOT = _project_root()
UTILS_PATH = os.path.join(PROJECT_ROOT, "utils")
sys.path.insert(0, UTILS_PATH)

DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def ensure_sample_data():
    """Ensure demo CSVs exist under data/raw (used when Impala tables are empty or unreadable)."""
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
    from forecasting_pipeline import DENSE_DEMO_NSN, run_training

    ensure_sample_data()
    os.makedirs(MODELS_DIR, exist_ok=True)
    dense_nsn = os.environ.get("DENSE_DEMO_NSN", DENSE_DEMO_NSN)
    summary = run_training(data_dir=DATA_RAW, models_dir=MODELS_DIR, dense_nsn=dense_nsn)
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
    # Jobs may run main.py inside Jupyter/ipython (ipykernel passes -f kernel.json, etc.)
    args, _unknown = parser.parse_known_args()

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
