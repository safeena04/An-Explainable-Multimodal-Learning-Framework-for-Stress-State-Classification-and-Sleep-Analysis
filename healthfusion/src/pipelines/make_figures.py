"""Phase 11: regenerate every paper figure from the saved metrics.

Run:  python -m src.pipelines.make_figures
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import METRICS_DIR, PROCESSED_DIR
from src.visualization import plots


def main() -> None:
    summary = pd.read_csv(METRICS_DIR / "qol_cv_summary.csv")
    best = pd.read_csv(METRICS_DIR / "qol_best_per_modality_set.csv")
    baselines = summary[summary.experiment.str.startswith("Baseline")]
    meta = json.load(open(METRICS_DIR / "qol_experiment_meta.json"))
    oof = pd.read_csv(METRICS_DIR / "qol_oof_predictions.csv")
    out = [plots.qol_modality_comparison(best, baselines, meta["n_participants_complete_case"]),
           plots.qol_fusion_strategies(summary)]

    roc = {}
    for mod in ("ECG", "EDA", "HR"):
        e = best[best.modalities == mod]
        if len(e):
            roc[f"{mod} only ({e.iloc[0].model})"] = e.iloc[0].experiment
    roc[f"Fusion {meta['best_overall'].split(' | ')[0]} ({meta['best_overall'].split(' | ', 1)[1]})"] = \
        meta["best_overall"] if "+" in meta["best_overall"].split(" | ")[0] else \
        best[best.modalities.str.contains(r"\+")].iloc[0].experiment
    out.append(plots.qol_roc_curves(oof, roc))

    row = summary.set_index("experiment").loc[meta["best_overall"]]
    cm = np.array([[row.cm_tn, row.cm_fp], [row.cm_fn, row.cm_tp]])
    out.append(plots.confusion_matrix_plot(cm, f"Confusion matrix (mean per CV repeat)\n{meta['best_overall']}",
                                           "qol_confusion_matrix_best"))
    out.append(plots.qol_paired_effects(pd.read_csv(METRICS_DIR / "qol_paired_tests.csv")))
    out.append(plots.qol_stai_manipulation(pd.read_csv(PROCESSED_DIR / "qol_questionnaires.csv")
                                           .dropna(subset=["stai_relax", "stai_stress"])))
    w = pd.read_csv(METRICS_DIR / "qol_stacking_weights.csv", header=[0, 1], index_col=0)
    out.append(plots.qol_stacking_weights(w))

    res = pd.read_csv(METRICS_DIR / "mmash_loso_results.csv")
    out.append(plots.mmash_loso(res, "sleep_efficiency"))
    corr = pd.read_csv(METRICS_DIR / "mmash_correlations.csv")
    out.append(plots.mmash_correlation_heatmap(corr[corr.modality != "Exploratory (not a predictor)"]))
    for p in out:
        print(p)


if __name__ == "__main__":
    main()
