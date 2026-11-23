"""Phase 8: explainability (SHAP) for QoL_Stress and MMASH.

QoL: out-of-fold SHAP. For each fold of the first CV repeat the model is fitted
on the training participants and explains only the held-out participants, so
every explanation describes a prediction the model made on unseen people.

Run:  python -m src.pipelines.run_explainability
"""
from __future__ import annotations

import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.model_selection import LeaveOneOut  # noqa: E402

from src.config import FIGURES_DIR, GLOBAL_SEED, METRICS_DIR, PROCESSED_DIR  # noqa: E402
from src.evaluation.cv import fit_estimator, predict_estimator, repeated_group_splits  # noqa: E402
from src.explainability.shap_analysis import (explain_any, explain_pipeline, global_importance,  # noqa: E402
                                              output_scale, rank_stability)
from src.fusion.fusion import LogitPoolFusion  # noqa: E402
from src.models.zoo import make_regressor  # noqa: E402
from src.pipelines.run_qol_experiments import experiment_specs, load_sets  # noqa: E402
from src.preprocessing.mmash_preprocess import MMASH_MODALITIES  # noqa: E402
from src.preprocessing.qol_preprocess import MODALITIES  # noqa: E402
from src.visualization import plots  # noqa: E402

QOL_MOD_OF = {c: m for m, cols in MODALITIES.items() for c in cols}
MMASH_MOD_OF = {c: m for m, cols in MMASH_MODALITIES.items() for c in cols}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def oof_shap(spec, df: pd.DataFrame):
    y, g = df["label"].to_numpy(), df["pid"].to_numpy()
    splits = repeated_group_splits(g, seeds=[GLOBAL_SEED])
    cols = spec.feature_list()
    vals, bases, folds, probs = [], [], [], []
    scale = "log-odds"
    for _, f, tr, te in splits:
        est, _ = fit_estimator(spec, df.iloc[tr], y[tr], g[tr], GLOBAL_SEED)
        v, b = explain_any(est, df.iloc[tr], df.iloc[te], cols)
        if not isinstance(est, LogitPoolFusion):
            scale = output_scale(est)
        vals.append(v[cols])
        bases.append(pd.Series(b, index=v.index))
        folds.append(pd.Series(f, index=v.index))
        probs.append(pd.Series(predict_estimator(spec, est, df.iloc[te]), index=v.index))
    V = pd.concat(vals).sort_index()
    B = pd.concat(bases).sort_index()
    F = pd.concat(folds).sort_index()
    P = pd.concat(probs).sort_index()
    return V, B, F, P, scale


def beeswarm(V: pd.DataFrame, X: pd.DataFrame, title: str, name: str) -> None:
    plots.apply_style()
    exp = shap.Explanation(values=V.to_numpy(), data=X[V.columns].to_numpy(), feature_names=list(V.columns))
    plt.figure()
    shap.plots.beeswarm(exp, max_display=len(V.columns), show=False, plot_size=(7, 0.32 * len(V.columns) + 1.5))
    plt.title(title, loc="left", fontsize=10, fontweight="bold")
    plots.save(plt.gcf(), name)


def waterfall(v: pd.Series, base: float, x: pd.Series, title: str, name: str) -> None:
    plots.apply_style()
    exp = shap.Explanation(values=v.to_numpy(), base_values=base, data=x[v.index].to_numpy(),
                           feature_names=list(v.index))
    plt.figure()
    shap.plots.waterfall(exp, max_display=10, show=False)
    plt.title(title, loc="left", fontsize=9, fontweight="bold")
    plots.save(plt.gcf(), name)


def qol_explainability() -> dict:
    meta = json.load(open(METRICS_DIR / "qol_experiment_meta.json"))
    specs = {s.name: s for s in experiment_specs()}
    targets = list(dict.fromkeys([meta["best_overall"], meta["best_single_modality"], "ECG+EDA+HR | early | logreg",
                                  "ECG+EDA+HR | early | xgboost"]))
    df = load_sets()["complete_case"]
    out, imps, shares = {}, {}, {}
    for name in targets:
        spec = specs[name]
        V, B, F, P, scale = oof_shap(spec, df)
        s = slug(name)
        V.assign(pid=df.loc[V.index, "pid"], phase=df.loc[V.index, "phase"], base_value=B, prob=P, fold=F) \
            .to_csv(METRICS_DIR / f"shap_oof_values_{s}.csv")
        imp = global_importance(V, F.to_numpy(), QOL_MOD_OF)
        imp.to_csv(METRICS_DIR / f"shap_global_{s}.csv", index=False)
        imps[name] = imp
        share = imp.groupby("modality")["mean_abs_shap"].sum()
        shares[name] = (share / share.sum()).reindex([m for m in MODALITIES if m in share.index])
        out[name] = {"output_scale": scale, "rank_stability_mean_pairwise_spearman": rank_stability(V, F.to_numpy()),
                     "modality_share": shares[name].round(4).to_dict(),
                     "top5": imp.head(5)[["feature", "mean_abs_shap", "mean_shap"]].round(4).to_dict("records")}
        nice = name.replace(" | ", ", ")
        plots.shap_importance_bar(imp, f"Global SHAP importance: {nice}", f"shap_bar_{s}")
        beeswarm(V, df, f"SHAP summary (out-of-fold): {nice}", f"shap_beeswarm_{s}")
        print(name, out[name]["modality_share"], "stability", round(out[name]["rank_stability_mean_pairwise_spearman"], 3))

        if name == meta["best_overall"]:
            d = df.loc[V.index].assign(prob=P)
            picks = {
                "confident_stress": d[d.label == 1].prob.idxmax(),
                "confident_relax": d[d.label == 0].prob.idxmin(),
                "misclassified": (d.prob - d.label).abs().idxmax(),
            }
            for tag, i in picks.items():
                r = d.loc[i]
                waterfall(V.loc[i], float(B.loc[i]), df.loc[i],
                          f"{tag.replace('_', ' ')}: {r.pid} {r.phase}, P(stress) = {r.prob:.2f}",
                          f"shap_waterfall_{tag}")
            out[name]["individual_examples"] = {k: {"pid": d.loc[i, "pid"], "phase": d.loc[i, "phase"],
                                                    "prob_stress": float(d.loc[i, "prob"])} for k, i in picks.items()}

    share_df = pd.DataFrame(shares).T.fillna(0)
    share_df.to_csv(METRICS_DIR / "shap_modality_share.csv")
    plots.modality_share_bar(share_df.loc[[n for n in share_df.index if "+" in n.split(" | ")[0]]],
                             "Modality contribution (share of mean |SHAP|)", "shap_modality_share")
    # agreement between explanation models
    agree = {}
    names = list(imps)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = imps[names[i]].set_index("feature")["mean_abs_shap"]
            b = imps[names[j]].set_index("feature")["mean_abs_shap"]
            common = a.index.intersection(b.index)
            if len(common) >= 5:
                agree[f"{names[i]}  vs  {names[j]}"] = float(spearmanr(a[common], b[common])[0])
    out["ranking_agreement_spearman"] = agree
    return out


def mmash_explainability() -> dict:
    df = pd.read_csv(PROCESSED_DIR / "mmash_participant_features.csv")
    cols = [c for m in MMASH_MODALITIES.values() for c in m]
    y = df["sleep_efficiency"].to_numpy(float)
    vals = []
    for tr, te in LeaveOneOut().split(df):
        pipe = make_regressor("ridge").fit(df.iloc[tr][cols], y[tr])
        v, _ = explain_pipeline(pipe, df.iloc[tr][cols], df.iloc[te][cols])
        vals.append(v)
    V = pd.concat(vals)
    imp = global_importance(V, np.arange(len(V)) % 3, MMASH_MOD_OF)
    imp.to_csv(METRICS_DIR / "shap_global_mmash_sleep_efficiency_ridge.csv", index=False)
    res = pd.read_csv(METRICS_DIR / "mmash_loso_results.csv")
    r = res[(res.target == "sleep_efficiency") & (res.modalities == "Phys+Behav+Psych") &
            (res.strategy == "early") & (res.model == "ridge")].iloc[0]
    validated = bool(r.r2 > 0 and r.perm_p_value < 0.05)
    verdict = (f"LOSO R2 = {r.r2:.2f}, permutation p = {r.perm_p_value:.3f}: "
               + ("model beat the mean baseline" if validated else "model did NOT beat the mean baseline"))
    plots.shap_importance_bar(imp.drop(columns="fold_sd"), "MMASH sleep efficiency, Ridge (all modalities), "
                              f"LOSO SHAP\nexploratory: {verdict}", "shap_bar_mmash_sleep_efficiency")
    share = imp.groupby("modality")["mean_abs_shap"].sum()
    return {"target": "sleep_efficiency", "model": "ridge, all modalities, LOSO",
            "modality_share": (share / share.sum()).round(4).to_dict(),
            "top5": imp.head(5)[["feature", "mean_abs_shap"]].round(4).to_dict("records"),
            "model_validated": validated, "verdict": verdict,
            "caveat": "With n = 21 these attributions are descriptive of one fitted model family only."}


def main() -> None:
    res = {"qol_stress": qol_explainability(), "mmash": mmash_explainability(),
           "interpretation_note": "SHAP values describe model behaviour (how predictions depend on inputs), "
                                  "not causal effects of physiology on stress or sleep."}
    json.dump(res, open(METRICS_DIR / "shap_summary.json", "w"), indent=2, default=float)
    print(json.dumps(res["qol_stress"].get("ranking_agreement_spearman"), indent=2))


if __name__ == "__main__":
    main()
