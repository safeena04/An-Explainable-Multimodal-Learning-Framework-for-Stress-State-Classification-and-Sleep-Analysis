"""SHAP explanations for fitted HealthFusion models.

All attributions are in log-odds (stress vs relax) for classifiers and in target
units for regressors. They describe how a *model* uses its inputs; they are not
causal effects of physiology on stress (see docs/RESEARCH_DESIGN.md, Phase H).

Which explainer:
* logistic regression / ridge     -> shap.LinearExplainer (exact, interventional)
* random forest / XGBoost / GB    -> shap.TreeExplainer (exact for trees)
* anything else (e.g. SVM)        -> shap.PermutationExplainer on the log-odds
* LogitPoolFusion (late/stacked)  -> per-modality explanation x modality weight;
  exact because the fused log-odds are a weighted sum of modality log-odds
  (for rows with every modality present).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from xgboost import XGBClassifier

from src.fusion.fusion import EPS, LogitPoolFusion


def _split_pipeline(pipe):
    pre, model = pipe[:-1], pipe[-1]
    return pre, model


def _logit_fn(pipe):
    def f(X):
        p = np.clip(pipe.predict_proba(pd.DataFrame(X, columns=pipe.feature_names_in_))[:, 1], EPS, 1 - EPS)
        return np.log(p / (1 - p))
    return f


def explain_pipeline(pipe, X_background: pd.DataFrame, X_explain: pd.DataFrame, max_background: int = 100,
                     seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """SHAP values for an impute->scale->model pipeline. Returns (values, base_values)."""
    cols = list(X_explain.columns)
    pre, model = _split_pipeline(pipe)
    Zb = pd.DataFrame(pre.transform(X_background), columns=cols)
    Ze = pd.DataFrame(pre.transform(X_explain), columns=cols)
    if len(Zb) > max_background:
        Zb = Zb.sample(max_background, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(model, (LogisticRegression, RidgeCV)):
            exp = shap.LinearExplainer(model, Zb)
            sv = exp(Ze)
            vals, base = sv.values, np.broadcast_to(sv.base_values, (len(Ze),))
        elif isinstance(model, (XGBClassifier, GradientBoostingClassifier)):
            exp = shap.TreeExplainer(model, Zb, model_output="raw")
            sv = exp(Ze)
            vals, base = sv.values, np.broadcast_to(sv.base_values, (len(Ze),))
        elif isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
            exp = shap.TreeExplainer(model)
            sv = exp(Ze)
            vals = sv.values[..., 1] if sv.values.ndim == 3 else sv.values
            base = sv.base_values[..., 1] if np.ndim(sv.base_values) == 2 else sv.base_values
            base = np.broadcast_to(base, (len(Ze),))
            # RF explains probability, not log-odds: documented in the output column name
        else:  # model-agnostic fallback (SVM etc.), on the raw-feature log-odds
            Xb = X_background.sample(min(len(X_background), 50), random_state=seed)
            exp = shap.PermutationExplainer(_logit_fn(pipe), shap.maskers.Independent(Xb.to_numpy()), seed=seed)
            sv = exp(X_explain.to_numpy())
            vals, base = sv.values, np.broadcast_to(sv.base_values, (len(X_explain),))
    return pd.DataFrame(np.asarray(vals), columns=cols, index=X_explain.index), np.asarray(base, dtype=float)


def output_scale(pipe) -> str:
    model = pipe[-1]
    if isinstance(model, RandomForestClassifier):
        return "probability"
    if isinstance(model, (RidgeCV, RandomForestRegressor)):
        return "target units"
    return "log-odds"


def explain_fusion(est: LogitPoolFusion, X_background: pd.DataFrame, X_explain: pd.DataFrame
                   ) -> tuple[pd.DataFrame, np.ndarray]:
    """Weighted per-modality SHAP values; modalities missing for a row get 0."""
    parts, base = [], np.full(len(X_explain), est.intercept_, dtype=float)
    avail_mat = pd.DataFrame({m: X_explain[c].notna().all(axis=1) for m, c in est.modalities.items()})
    n_avail = avail_mat.sum(axis=1).clip(lower=1)
    for mod, cols in est.modalities.items():
        w = est.weights_[mod]
        avail = avail_mat[mod]
        vals = pd.DataFrame(0.0, index=X_explain.index, columns=cols)
        if avail.any():
            bg = X_background.loc[X_background[cols].notna().all(axis=1), cols]
            v, b = explain_pipeline(est.models_[mod], bg, X_explain.loc[avail, cols])
            if est.weighting == "equal":
                w_row = 1.0 / n_avail.loc[avail]
                vals.loc[avail] = v.mul(w_row, axis=0).to_numpy()
                base[avail.to_numpy()] += b * w_row.to_numpy()
            else:
                vals.loc[avail] = v.to_numpy() * w
                base[avail.to_numpy()] += b * w
        parts.append(vals)
    return pd.concat(parts, axis=1), base


def explain_any(est, X_background: pd.DataFrame, X_explain: pd.DataFrame, feature_cols: list[str]):
    if isinstance(est, LogitPoolFusion):
        return explain_fusion(est, X_background, X_explain)
    return explain_pipeline(est, X_background[feature_cols], X_explain[feature_cols])


def global_importance(shap_df: pd.DataFrame, fold_ids: np.ndarray, modality_of: dict) -> pd.DataFrame:
    """mean |SHAP| overall, plus the SD of fold-wise mean |SHAP| as a stability measure."""
    absdf = shap_df.abs()
    imp = absdf.mean().rename("mean_abs_shap").to_frame()
    imp["fold_sd"] = absdf.groupby(fold_ids).mean().std()
    imp["mean_shap"] = shap_df.mean()
    imp["modality"] = [modality_of.get(f, "?") for f in imp.index]
    imp.index.name = "feature"
    return imp.reset_index().sort_values("mean_abs_shap", ascending=False)


def rank_stability(shap_df: pd.DataFrame, fold_ids: np.ndarray) -> float:
    """Mean pairwise Spearman correlation between fold-wise importance rankings."""
    from scipy.stats import spearmanr
    per_fold = shap_df.abs().groupby(fold_ids).mean()
    rs = [spearmanr(per_fold.iloc[i], per_fold.iloc[j])[0]
          for i in range(len(per_fold)) for j in range(i + 1, len(per_fold))]
    return float(np.nanmean(rs)) if rs else float("nan")
