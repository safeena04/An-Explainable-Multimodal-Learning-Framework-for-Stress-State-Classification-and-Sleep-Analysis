"""Phases 5-7 for QoL_Stress: baselines, single-modality models, fusion,
validation, statistical tests, sensitivity analyses.

Run:  python -m src.pipelines.run_qol_experiments  [--quick]
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (CV_REPEAT_SEEDS, GLOBAL_SEED, METRICS_DIR, N_PERMUTATIONS_QOL, OUTER_FOLDS,
                        PROCESSED_DIR)
from src.evaluation.cv import (METRICS_REPORTED, ExperimentSpec, repeated_group_splits, run_cv)
from src.evaluation.metrics import summarize_folds
from src.evaluation.stats import (corrected_resampled_ttest, paired_wilcoxon, permute_within_groups,
                                  spearman_with_ci)
from src.preprocessing.qol_preprocess import MODALITIES

ALL_FEATURES = [c for cols in MODALITIES.values() for c in cols]
COMBOS = [c for k in (1, 2, 3) for c in itertools.combinations(MODALITIES, k)]
EARLY_MODELS = ["logreg", "svm_rbf", "random_forest", "xgboost", "gradient_boosting"]
LATE_BASES = ["logreg", "xgboost"]


def load_sets() -> dict[str, pd.DataFrame]:
    f = pd.read_csv(PROCESSED_DIR / "qol_phase_features.csv")
    return {"complete_case": f[f["complete_case"]].reset_index(drop=True), "available_case": f}


def experiment_specs(quick: bool = False) -> list[ExperimentSpec]:
    specs = [
        ExperimentSpec(("ECG", "EDA", "HR"), "early", "dummy", MODALITIES, label="Baseline | chance (prior)"),
        ExperimentSpec(("HR",), "single", "logreg", MODALITIES, feature_override=("fb_hr_mean",),
                       label="Baseline | mean Fitbit HR only | logreg"),
        ExperimentSpec(("ECG",), "single", "logreg", MODALITIES, feature_override=("ecg_hr",),
                       label="Baseline | ECG HR only | logreg"),
    ]
    models = ["logreg", "xgboost"] if quick else EARLY_MODELS
    for combo in COMBOS:
        strat = "single" if len(combo) == 1 else "early"
        specs += [ExperimentSpec(combo, strat, m, MODALITIES) for m in models]
        if len(combo) > 1:
            specs += [ExperimentSpec(combo, s, b, MODALITIES) for s in ("late", "stacked") for b in LATE_BASES]
    return specs


def paired_statistics(feat: pd.DataFrame, q: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    tests = paired_wilcoxon(feat, ALL_FEATURES)
    tests.insert(1, "modality", tests["feature"].map({c: m for m, cols in MODALITIES.items() for c in cols}))
    s = q.dropna(subset=["stai_relax", "stai_stress"])
    w = stats.wilcoxon(s["stai_stress"], s["stai_relax"])
    d = s["stai_stress"] - s["stai_relax"]
    manip = {"n": int(len(s)), "stai_relax_mean": float(s["stai_relax"].mean()),
             "stai_stress_mean": float(s["stai_stress"].mean()), "mean_increase": float(d.mean()),
             "cohen_dz": float(d.mean() / d.std(ddof=1)), "wilcoxon_p": float(w.pvalue),
             "n_increase": int((d > 0).sum()), "n_equal": int((d == 0).sum()), "n_decrease": int((d < 0).sum())}
    return tests, manip


def permutation_test(spec: ExperimentSpec, df: pd.DataFrame, n_perm: int, observed_auc: float) -> dict:
    """Group-preserving permutation test on single-repeat 5-fold CV AUC."""
    X, y, g = df, df["label"].to_numpy(), df["pid"].to_numpy()
    splits = repeated_group_splits(g, OUTER_FOLDS, seeds=[GLOBAL_SEED])
    rng = np.random.default_rng(GLOBAL_SEED)
    null = []
    for _ in range(n_perm):
        yp = permute_within_groups(y, g, rng)
        folds, _ = run_cv(spec, X, yp, g, splits)
        null.append(folds["roc_auc"].mean())
    null = np.asarray(null)
    return {"experiment": spec.name, "observed_auc_single_repeat": observed_auc, "n_permutations": n_perm,
            "null_mean": float(null.mean()), "null_95th": float(np.percentile(null, 95)),
            "p_value": float((1 + np.sum(null >= observed_auc)) / (1 + n_perm))}


def main(quick: bool = False, n_perm: int = N_PERMUTATIONS_QOL) -> None:
    t0 = time.time()
    sets = load_sets()
    q = pd.read_csv(PROCESSED_DIR / "qol_questionnaires.csv")

    # ---------------- statistics (all available pairs per feature) ----------------
    tests, manip = paired_statistics(sets["available_case"], q)
    tests.to_csv(METRICS_DIR / "qol_paired_tests.csv", index=False)
    json.dump(manip, open(METRICS_DIR / "qol_manipulation_check.json", "w"), indent=2)
    print("manipulation check:", manip)

    # ---------------- main experiments: complete-case participants ----------------
    cc = sets["complete_case"]
    y, g = cc["label"].to_numpy(), cc["pid"].to_numpy()
    splits = repeated_group_splits(g)
    all_folds, all_oof = [], []
    for spec in experiment_specs(quick):
        t = time.time()
        folds, oof = run_cv(spec, cc, y, g, splits)
        all_folds.append(folds)
        all_oof.append(oof)
        print(f"{spec.name:<45s} AUC {folds.roc_auc.mean():.3f} ± {folds.roc_auc.std():.3f} "
              f"BA {folds.balanced_accuracy.mean():.3f} pair {folds.pairwise_accuracy.mean():.3f} "
              f"({time.time() - t:.0f}s)", flush=True)
    folds = pd.concat(all_folds, ignore_index=True)
    oof = pd.concat(all_oof, ignore_index=True)
    folds.to_csv(METRICS_DIR / "qol_cv_folds.csv", index=False)
    oof.to_csv(METRICS_DIR / "qol_oof_predictions.csv", index=False)

    summary = []
    for exp, gdf in folds.groupby("experiment", sort=False):
        s = summarize_folds(gdf, METRICS_REPORTED)
        row = {"experiment": exp, "modalities": gdf["modalities"].iloc[0], "strategy": gdf["strategy"].iloc[0],
               "model": gdf["model"].iloc[0], "n_participants": len(np.unique(g)), "n_rows": len(cc)}
        for m in METRICS_REPORTED:
            row[f"{m}_mean"], row[f"{m}_sd"] = s.loc[m, "mean"], s.loc[m, "sd"]
            row[f"{m}_ci_low"], row[f"{m}_ci_high"] = s.loc[m, "ci95_low"], s.loc[m, "ci95_high"]
        cm = gdf[["tn", "fp", "fn", "tp"]].sum() / len(CV_REPEAT_SEEDS)
        row.update({f"cm_{k}": v for k, v in cm.items()})
        summary.append(row)
    summary = pd.DataFrame(summary).sort_values("roc_auc_mean", ascending=False)
    summary.to_csv(METRICS_DIR / "qol_cv_summary.csv", index=False)

    # ---------------- model comparisons (corrected resampled t-test) ----------------
    n_tr = int(folds["n_train"].mean())
    n_te = int(folds["n_test"].mean())
    best_by_combo = (summary[~summary.experiment.str.startswith("Baseline")]
                     .sort_values("roc_auc_mean", ascending=False).groupby("modalities").head(1))
    best_by_combo.to_csv(METRICS_DIR / "qol_best_per_modality_set.csv", index=False)
    best_overall = best_by_combo.iloc[0]["experiment"]
    singles = best_by_combo[best_by_combo.modalities.isin(MODALITIES)]
    best_single = singles.iloc[0]["experiment"]
    key = ["repeat", "fold"]

    def compare(a, b):
        fa = folds[folds.experiment == a].set_index(key)["roc_auc"]
        fb = folds[folds.experiment == b].set_index(key)["roc_auc"]
        r = corrected_resampled_ttest((fa - fb.loc[fa.index]).to_numpy(), n_tr, n_te)
        return {"model_a": a, "model_b": b, "metric": "roc_auc", **r}

    comps = [compare(best_overall, e) for e in best_by_combo.experiment if e != best_overall]
    comps.append(compare(best_overall, "Baseline | chance (prior)"))
    comps.append(compare(best_overall, "Baseline | mean Fitbit HR only | logreg"))
    for combo in [c for c in COMBOS if len(c) > 1]:
        mods = "+".join(combo)
        early_best = summary[(summary.modalities == mods) & (summary.strategy == "early")].iloc[0]["experiment"]
        for s in ("late", "stacked"):
            for b in LATE_BASES:
                e = f"{mods} | {s} | {b}"
                if e in set(folds.experiment):
                    comps.append(compare(e, early_best))
    comps = pd.DataFrame(comps)
    comps.to_csv(METRICS_DIR / "qol_model_comparisons.csv", index=False)

    # ---------------- fusion weights learned by stacking ----------------
    wcols = [c for c in folds.columns if c.startswith("w_")]
    wts = folds[folds.strategy == "stacked"].groupby("experiment")[wcols].agg(["mean", "std"])
    wts.to_csv(METRICS_DIR / "qol_stacking_weights.csv")

    # ---------------- permutation tests ----------------
    perm_specs = [s for s in experiment_specs(quick) if s.name in
                  {best_overall, best_single, "ECG | single | logreg", "EDA | single | logreg",
                   "HR | single | logreg", "ECG+EDA+HR | early | logreg"}]
    perm_rows = []
    for spec in perm_specs:
        obs = folds[(folds.experiment == spec.name) & (folds.repeat == 0)]["roc_auc"].mean()
        heavy = spec.model in ("xgboost", "random_forest", "gradient_boosting", "svm_rbf")
        n = max(50, n_perm // 4) if heavy else n_perm
        t = time.time()
        perm_rows.append(permutation_test(spec, cc, n, obs))
        print(f"permutation {spec.name}: p={perm_rows[-1]['p_value']:.4f} ({time.time() - t:.0f}s)", flush=True)
    pd.DataFrame(perm_rows).to_csv(METRICS_DIR / "qol_permutation_tests.csv", index=False)

    # ---------------- external validation with STAI (not a model input) ----------------
    o = oof[(oof.experiment == best_overall)].groupby(["pid", "y"])["prob"].mean().unstack()
    o.columns = ["p_relax", "p_stress"]
    o["delta_p"] = o["p_stress"] - o["p_relax"]
    ev = o.join(q.set_index("pid")[["stai_delta", "pss14", "stai_relax", "stai_stress"]])
    ext = {"experiment": best_overall,
           "delta_p_vs_stai_delta": spearman_with_ci(ev["delta_p"], ev["stai_delta"]),
           "delta_p_vs_pss14": spearman_with_ci(ev["delta_p"], ev["pss14"]),
           "p_stress_vs_stai_stress": spearman_with_ci(ev["p_stress"], ev["stai_stress"]),
           "note": "Out-of-fold probabilities averaged over CV repeats. STAI/PSS were never model inputs."}
    json.dump(ext, open(METRICS_DIR / "qol_external_validation.json", "w"), indent=2)
    ev.to_csv(METRICS_DIR / "qol_external_validation_table.csv")

    sensitivity_and_meta(quick, t0)


def sensitivity_and_meta(quick: bool = False, t0: float | None = None) -> None:
    """Sensitivity analyses + experiment metadata (reads the saved CV summary)."""
    t0 = t0 or time.time()
    sets = load_sets()
    cc = sets["complete_case"]
    best = pd.read_csv(METRICS_DIR / "qol_best_per_modality_set.csv")
    best_overall = best.iloc[0]["experiment"]
    best_single = best[best.modalities.isin(MODALITIES)].iloc[0]["experiment"]
    spec_lookup = {s.name: s for s in experiment_specs(quick)}
    names = list(dict.fromkeys([best_overall, "ECG+EDA+HR | early | logreg", "ECG+EDA+HR | late | logreg",
                                "ECG+EDA+HR | stacked | logreg"]))
    insp = json.load(open(METRICS_DIR.parent / "reports" / "phase1_inspection.json"))
    tc = insp["qol_stress"]["ecg_time_consistency"]
    ts_out = set(tc["ecg_relax_outliers_pid"]) | set(tc["ecg_stress_outliers_pid"])
    cc_ts = cc[~cc.pid.isin(ts_out)].reset_index(drop=True)
    sens = []
    for label, df in (("available_case_all_66", sets["available_case"]),
                      ("complete_case_ecg_timestamp_consistent", cc_ts)):
        sp = repeated_group_splits(df["pid"].to_numpy())
        for name in names:
            f2, _ = run_cv(spec_lookup[name], df, df["label"].to_numpy(), df["pid"].to_numpy(), sp)
            sens.append({"analysis": label, "experiment": name, "n_participants": df.pid.nunique(),
                         "n_rows": len(df), "roc_auc_mean": f2.roc_auc.mean(), "roc_auc_sd": f2.roc_auc.std(),
                         "balanced_accuracy_mean": f2.balanced_accuracy.mean(),
                         "pairwise_accuracy_mean": f2.pairwise_accuracy.mean()})
            print(f"sensitivity {label} {name}: AUC {f2.roc_auc.mean():.3f}", flush=True)
    pd.DataFrame(sens).to_csv(METRICS_DIR / "qol_sensitivity.csv", index=False)

    meta = {"best_overall": best_overall, "best_single_modality": best_single,
            "n_participants_complete_case": int(cc.pid.nunique()), "outer_folds": OUTER_FOLDS,
            "repeats": len(CV_REPEAT_SEEDS), "seeds": CV_REPEAT_SEEDS,
            "n_configurations_compared": int(pd.read_csv(METRICS_DIR / "qol_cv_summary.csv").shape[0]),
            "selection_note": "best_overall is the highest mean outer-CV AUC among all configurations; choosing "
                              "the maximum of many configurations is optimistic (winner's curse).",
            "runtime_s": round(time.time() - t0)}
    json.dump(meta, open(METRICS_DIR / "qol_experiment_meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer models (smoke test)")
    ap.add_argument("--n-perm", type=int, default=N_PERMUTATIONS_QOL)
    ap.add_argument("--only-sensitivity", action="store_true", help="rerun only the final stage")
    a = ap.parse_args()
    if a.only_sensitivity:
        sensitivity_and_meta(a.quick)
    else:
        main(a.quick, a.n_perm)
