"""MMASH sleep analysis (n = 21 participants, one night each).

Because n is tiny, the analysis is framed as participant-level statistical
modelling with exploratory prediction:
  1. Spearman correlations (bootstrap 95 % CI, Benjamini-Hochberg FDR per target).
  2. Leave-one-subject-out (LOSO) regression, modality-wise and early/late
     fusion, against a mean-prediction baseline.
  3. Label-permutation tests (1000 permutations) for every Ridge model of the
     primary target, so "better than chance" is tested rather than assumed.
Stacked/weighted fusion is NOT run: learning modality weights from 20 training
participants per fold is not statistically meaningful (recorded in the output).

Run:  python -m src.pipelines.run_mmash_experiments
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from src.config import GLOBAL_SEED, METRICS_DIR, MMASH_PRIMARY_TARGET, N_PERMUTATIONS_MMASH, PROCESSED_DIR

N_PERMUTATIONS_RF = 200   # RF LOSO is ~50x slower than Ridge; 200 permutations resolve p to 0.005
from src.evaluation.metrics import regression_metrics
from src.evaluation.stats import benjamini_hochberg, bootstrap_ci, spearman_with_ci
from src.models.zoo import REGRESSORS, make_regressor
from src.preprocessing.mmash_preprocess import MMASH_MODALITIES, TARGETS

EXTRA = ["age", "bmi", "cortisol_before_sleep", "melatonin_before_sleep"]
COMBOS = [c for k in (1, 2, 3) for c in itertools.combinations(MMASH_MODALITIES, k)]
SHORT = {"Physiological": "Phys", "Behavioural": "Behav", "Psychological": "Psych"}


def cols(combo) -> list[str]:
    return [c for m in combo for c in MMASH_MODALITIES[m]]


def combo_name(combo) -> str:
    return "+".join(SHORT[m] for m in combo)


def loso_predict(model_name: str, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    return cross_val_predict(make_regressor(model_name), X, y, cv=LeaveOneOut())


def late_fusion_predict(combo, df, y) -> np.ndarray:
    """Equal-weight average of modality-specific Ridge LOSO predictions."""
    return np.mean([loso_predict("ridge", df[MMASH_MODALITIES[m]], y) for m in combo], axis=0)


def permutation_p(model_name, X, y, observed_mae, n_perm, seed=GLOBAL_SEED, late_combo=None, df=None):
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        yp = rng.permutation(y)
        pred = late_fusion_predict(late_combo, df, yp) if late_combo else loso_predict(model_name, X, yp)
        null[i] = np.mean(np.abs(yp - pred))
    return float((1 + np.sum(null <= observed_mae)) / (1 + n_perm)), float(np.median(null))


def main(n_perm: int = N_PERMUTATIONS_MMASH) -> None:
    t0 = time.time()
    df = pd.read_csv(PROCESSED_DIR / "mmash_participant_features.csv")
    predictors = cols(tuple(MMASH_MODALITIES))

    # 1) correlations ---------------------------------------------------------------
    rows = []
    for t in TARGETS:
        for f in predictors + EXTRA:
            r = spearman_with_ci(df[f], df[t], seed=GLOBAL_SEED)
            mod = next((m for m, c in MMASH_MODALITIES.items() if f in c), "Exploratory (not a predictor)")
            rows.append({"target": t, "feature": f, "modality": mod, **r})
    corr = pd.DataFrame(rows)
    corr["p_fdr"] = corr.groupby("target")["p_value"].transform(lambda p: benjamini_hochberg(p.to_numpy()))
    corr.to_csv(METRICS_DIR / "mmash_correlations.csv", index=False)
    print("correlations with p_fdr < 0.05:", int((corr.p_fdr < 0.05).sum()))

    # 2) LOSO regression ---------------------------------------------------------------
    res, preds = [], []
    for t in TARGETS:
        y = df[t].to_numpy(float)
        base_pred = loso_predict("dummy_mean", df[predictors], y)
        for combo in COMBOS:
            X = df[cols(combo)]
            variants = [(m, "early" if len(combo) > 1 else "single") for m in REGRESSORS if m != "dummy_mean"]
            for model, strat in variants + ([("ridge", "late")] if len(combo) > 1 else []):
                p = late_fusion_predict(combo, df, y) if strat == "late" else loso_predict(model, X, y)
                m = regression_metrics(y, p)
                diff = np.abs(y - base_pred) - np.abs(y - p)    # >0 means better than baseline
                lo, hi = bootstrap_ci(diff, seed=GLOBAL_SEED)
                res.append({"target": t, "modalities": combo_name(combo), "strategy": strat, "model": model,
                            "n": len(y), "n_features": X.shape[1], **m,
                            "mae_gain_vs_mean": float(diff.mean()), "mae_gain_ci_low": lo, "mae_gain_ci_high": hi})
                preds.append(pd.DataFrame({"target": t, "experiment": f"{combo_name(combo)} | {strat} | {model}",
                                           "user": df["user"], "y": y, "pred": p}))
        bm = regression_metrics(y, base_pred)
        res.append({"target": t, "modalities": "none", "strategy": "baseline", "model": "dummy_mean",
                    "n": len(y), "n_features": 0, **bm, "mae_gain_vs_mean": 0.0})
        preds.append(pd.DataFrame({"target": t, "experiment": "baseline | mean", "user": df["user"], "y": y,
                                   "pred": base_pred}))
    res = pd.DataFrame(res)

    # 3) permutation tests ------------------------------------------------------------
    res["perm_p_value"] = np.nan
    res["perm_null_median_mae"] = np.nan
    for i, r in res.iterrows():
        primary = r.target == MMASH_PRIMARY_TARGET and r.model == "ridge"
        secondary = r.target != MMASH_PRIMARY_TARGET and r.model == "ridge" and r.modalities == "Phys+Behav+Psych" \
            and r.strategy == "early"
        if not (primary or secondary):
            continue
        combo = tuple(k for k, v in SHORT.items() if v in r.modalities.split("+"))
        y = df[r.target].to_numpy(float)
        p, med = permutation_p("ridge", df[cols(combo)], y, r.mae, n_perm,
                               late_combo=combo if r.strategy == "late" else None, df=df)
        res.loc[i, ["perm_p_value", "perm_null_median_mae"]] = [p, med]
        print(f"perm {r.target} {r.modalities} {r.strategy}: MAE {r.mae:.2f} p={p:.3f}", flush=True)
    res.to_csv(METRICS_DIR / "mmash_loso_results.csv", index=False)
    pd.concat(preds).to_csv(METRICS_DIR / "mmash_loso_predictions.csv", index=False)

    res = rf_permutation_tests(df, res)
    res.to_csv(METRICS_DIR / "mmash_loso_results.csv", index=False)
    prim = res[res.target == MMASH_PRIMARY_TARGET].sort_values("mae")
    meta = {
        "n_participants": int(len(df)), "primary_target": MMASH_PRIMARY_TARGET, "validation": "leave-one-subject-out",
        "permutations": n_perm,
        "any_model_beats_baseline_primary": bool(((prim.r2 > 0) & (prim.perm_p_value < 0.05)).any()),
        "weighted_fusion": "not run: fusion weights cannot be estimated reliably from 20 training participants",
        "runtime_s": round(time.time() - t0),
    }
    json.dump(meta, open(METRICS_DIR / "mmash_experiment_meta.json", "w"), indent=2)
    print(prim[["modalities", "strategy", "model", "mae", "rmse", "r2", "perm_p_value"]].to_string(index=False))
    print(json.dumps(meta, indent=2))


def _rf_null_mae(X, y, seed):
    yp = np.random.default_rng(seed).permutation(y)
    return float(np.mean(np.abs(yp - loso_predict("random_forest", X, yp))))


def rf_permutation_tests(df: pd.DataFrame, res: pd.DataFrame, n_perm: int = N_PERMUTATIONS_RF) -> pd.DataFrame:
    """Permutation tests for every random-forest model of the primary target whose
    LOSO MAE is below the mean baseline (the only RF results that could be claimed)."""
    base = res[(res.target == MMASH_PRIMARY_TARGET) & (res.strategy == "baseline")]["mae"].iloc[0]
    cand = res[(res.target == MMASH_PRIMARY_TARGET) & (res.model == "random_forest") & (res.mae < base)]
    y = df[MMASH_PRIMARY_TARGET].to_numpy(float)
    for i, r in cand.iterrows():
        combo = tuple(k for k, v in SHORT.items() if v in r.modalities.split("+"))
        X = df[cols(combo)]
        t = time.time()
        null = np.array(Parallel(n_jobs=2)(delayed(_rf_null_mae)(X, y, GLOBAL_SEED + 1 + k) for k in range(n_perm)))
        p = float((1 + np.sum(null <= r.mae)) / (1 + n_perm))
        res.loc[i, ["perm_p_value", "perm_null_median_mae"]] = [p, float(np.median(null))]
        print(f"perm RF {r.modalities}: MAE {r.mae:.2f} R2 {r.r2:+.3f} p={p:.3f} ({time.time() - t:.0f}s)", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-rf-perm", action="store_true", help="add RF permutation tests to saved results")
    a = ap.parse_args()
    if a.only_rf_perm:
        d = pd.read_csv(PROCESSED_DIR / "mmash_participant_features.csv")
        out = rf_permutation_tests(d, pd.read_csv(METRICS_DIR / "mmash_loso_results.csv"))
        out.to_csv(METRICS_DIR / "mmash_loso_results.csv", index=False)
        m = json.load(open(METRICS_DIR / "mmash_experiment_meta.json"))
        prim = out[out.target == MMASH_PRIMARY_TARGET]
        m["any_model_beats_baseline_primary"] = bool(((prim.r2 > 0) & (prim.perm_p_value < 0.05)).any())
        m["rf_permutations"] = N_PERMUTATIONS_RF
        json.dump(m, open(METRICS_DIR / "mmash_experiment_meta.json", "w"), indent=2)
        print(json.dumps(m, indent=2))
    else:
        main()
