"""Train the final (deployable) models and write model cards + registry.

Stress: for every modality subset (ECG, EDA, HR, and their combinations) the
configuration with the best cross-validated ROC-AUC is refitted on all
complete-case participants with the same inner-CV tuning used during
validation. The API picks the model that matches the modalities a request
provides, so no whole modality ever has to be imputed.

Sleep: a model is saved for serving only if it passed its validation criterion.

Run:  python -m src.pipelines.train_final_models
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import CV_REPEAT_SEEDS, GLOBAL_SEED, METRICS_DIR, MMASH_PRIMARY_TARGET, OUTER_FOLDS, PROCESSED_DIR
from src.evaluation.cv import fit_estimator
from src.evaluation.stats import corrected_resampled_ttest, holm
from src.models.registry import save_card_only, save_model, write_registry
from src.models.zoo import make_regressor
from src.pipelines.run_qol_experiments import experiment_specs, load_sets
from src.preprocessing.mmash_preprocess import MMASH_MODALITIES
from src.preprocessing.qol_preprocess import MODALITIES


def stress_models() -> list[dict]:
    folds = pd.read_csv(METRICS_DIR / "qol_cv_folds.csv")
    summary = pd.read_csv(METRICS_DIR / "qol_cv_summary.csv")
    best = pd.read_csv(METRICS_DIR / "qol_best_per_modality_set.csv")
    perm = pd.read_csv(METRICS_DIR / "qol_permutation_tests.csv").set_index("experiment")
    specs = {s.name: s for s in experiment_specs()}
    df = load_sets()["complete_case"]
    y, g = df["label"].to_numpy(), df["pid"].to_numpy()
    entries = []
    for _, row in best.iterrows():
        spec = specs[row.experiment]
        f = folds[folds.experiment == row.experiment]
        tt = corrected_resampled_ttest(f["roc_auc"].to_numpy() - 0.5, int(f.n_train.mean()), int(f.n_test.mean()))
        validated = bool(tt["mean_diff"] > 0 and tt["p_value"] < 0.05)
        est, params = fit_estimator(spec, df, y, g, GLOBAL_SEED)
        cols = spec.feature_list()
        key = "stress_" + "_".join(m.lower() for m in spec.modalities)
        s = summary.set_index("experiment").loc[row.experiment]
        card = {
            "key": key, "task": "stress_classification", "dataset": "QoL_Stress",
            "modalities": list(spec.modalities), "strategy": spec.strategy, "algorithm": spec.model,
            "features": cols, "positive_class": "stress (1); relax = 0", "threshold": 0.5,
            "trained_on": {"participants": int(df.pid.nunique()), "rows": int(len(df)),
                           "subset": "complete-case participants (all three modalities in both phases)"},
            "hyperparameters": params, "random_seed": GLOBAL_SEED,
            "validation": {
                "scheme": f"{OUTER_FOLDS}-fold participant-grouped CV x {len(CV_REPEAT_SEEDS)} repeats, nested tuning",
                "seeds": CV_REPEAT_SEEDS,
                "roc_auc_mean": s.roc_auc_mean, "roc_auc_sd": s.roc_auc_sd,
                "balanced_accuracy_mean": s.balanced_accuracy_mean, "macro_f1_mean": s.macro_f1_mean,
                "pairwise_accuracy_mean": s.pairwise_accuracy_mean,
                "corrected_ttest_vs_chance": tt,
                "permutation_p_value": float(perm.loc[row.experiment, "p_value"]) if row.experiment in perm.index else None,
                "criterion": "mean AUC > 0.5 and corrected resampled t-test p < 0.05",
                "validated": validated,
            },
            "intended_use": "Research demonstration only. Not a medical device; not for diagnosis.",
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        artifact = {"estimator": est, "spec": {"modalities": spec.modalities, "strategy": spec.strategy,
                                                "model": spec.model, "features": cols},
                    "background": df[cols].copy(), "card": card}
        save_model("stress", key, artifact, card)
        entries.append({"key": key, "task": "stress", "modalities": list(spec.modalities),
                        "experiment": row.experiment, "validated": validated,
                        "roc_auc_mean": round(float(s.roc_auc_mean), 4), "corrected_p": tt["p_value"]})
        print(f"{key:<26s} {row.experiment:<40s} AUC {s.roc_auc_mean:.3f} p={tt['p_value']:.4f} "
              f"validated={validated}")
    return entries


def sleep_models() -> list[dict]:
    res = pd.read_csv(METRICS_DIR / "mmash_loso_results.csv")
    df = pd.read_csv(PROCESSED_DIR / "mmash_participant_features.csv")
    prim = res[(res.target == MMASH_PRIMARY_TARGET) & res.perm_p_value.notna()].copy()
    # every permutation-tested configuration of the primary target is one test of the same
    # question, so the p-values are Holm-adjusted before the validation criterion is applied
    prim["perm_p_holm"] = holm(prim["perm_p_value"].to_numpy())
    prim = prim.sort_values("mae")
    best = prim.iloc[0]
    validated = bool(best.r2 > 0 and best.perm_p_holm < 0.05)
    short = {"Phys": "Physiological", "Behav": "Behavioural", "Psych": "Psychological"}
    mods = [short[m] for m in best.modalities.split("+")]
    cols = [c for m in mods for c in MMASH_MODALITIES[m]]
    card = {
        "key": "sleep_efficiency", "task": "sleep_efficiency_regression", "dataset": "MMASH",
        "best_configuration": {"modalities": mods, "strategy": best.strategy, "model": best.model},
        "features": cols, "n_participants": int(best.n),
        "validation": {"scheme": "leave-one-subject-out", "mae": best.mae, "rmse": best.rmse, "r2": best.r2,
                       "baseline_mae": float(res[(res.target == MMASH_PRIMARY_TARGET) &
                                                 (res.strategy == "baseline")].mae.iloc[0]),
                       "permutation_p_value": best.perm_p_value,
                       "permutation_p_holm": best.perm_p_holm,
                       "n_configurations_tested": int(len(prim)),
                       "criterion": "LOSO R2 > 0 and Holm-adjusted permutation p < 0.05 (lowest-MAE tested model)",
                       "validated": validated},
        "decision": "served" if validated else
        "NOT served: no MMASH model predicted sleep efficiency better than the participant mean under LOSO.",
    }
    if validated and best.strategy != "late":
        est = make_regressor(best.model).fit(df[cols], df[MMASH_PRIMARY_TARGET])
        save_model("sleep", "sleep_efficiency", {"estimator": est, "features": cols, "background": df[cols]}, card)
    else:
        save_card_only("sleep", "sleep_efficiency", card)
    prim.to_csv(METRICS_DIR / "mmash_permutation_tested_models.csv", index=False)
    print("sleep:", card["decision"], f"(lowest-MAE tested model {best.modalities} {best.model}: LOSO R2 "
          f"{best.r2:.3f}, perm p {best.perm_p_value:.3f}, Holm p {best.perm_p_holm:.3f})")
    return [{"key": "sleep_efficiency", "task": "sleep", "validated": validated, "r2": float(best.r2),
             "permutation_p": float(best.perm_p_value), "permutation_p_holm": float(best.perm_p_holm),
             "experiment": f"{best.modalities} | {best.strategy} | {best.model}"}]


def main() -> None:
    entries = stress_models() + sleep_models()
    write_registry(entries)
    print(json.dumps(entries, indent=2, default=float))


if __name__ == "__main__":
    main()
