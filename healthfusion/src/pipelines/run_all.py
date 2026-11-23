"""Run every phase end-to-end, in order.

    python -m src.pipelines.run_all            # full run (~40 min on 2 CPU cores)
    python -m src.pipelines.run_all --quick    # smoke run: fewer models, 20 permutations

Each step writes its outputs to data/processed, results/ or models/ and can
also be run on its own (see README).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def step(title: str, fn) -> None:
    t = time.time()
    print(f"\n=== {title} ===", flush=True)
    fn()
    print(f"--- done in {time.time() - t:.0f}s", flush=True)


def main(quick: bool = False) -> None:
    from src.pipelines import (make_figures, run_cross_dataset, run_explainability, run_mmash_experiments,
                               run_preprocessing, run_qol_experiments, train_final_models)
    from src.data.inspection import run_inspection

    step("Phase 1  dataset inspection", run_inspection)
    step("Phases 3-4  preprocessing and features", run_preprocessing.main)
    step("Phases 5-7  QoL_Stress experiments",
         lambda: run_qol_experiments.main(quick=quick, n_perm=20 if quick else 200))
    step("MMASH sleep experiments", lambda: run_mmash_experiments.main(n_perm=50 if quick else 1000))
    step("Phase 8  explainability", run_explainability.main)
    step("Phase 9  cross-dataset analysis", run_cross_dataset.main)
    step("Final models + registry", train_final_models.main)
    step("Phase 11  figures", make_figures.main)
    step("Sample API payloads", lambda: subprocess.run([sys.executable, "scripts/make_sample.py"], check=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
