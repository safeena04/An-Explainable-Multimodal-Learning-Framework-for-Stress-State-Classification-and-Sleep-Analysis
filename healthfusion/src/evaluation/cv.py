"""Participant-independent, nested cross-validation.

Outer loop : repeated GroupKFold (5 folds x 5 repeats). All rows of a
             participant are in the test fold or in the training fold, never both.
Inner loop : GroupKFold (3 folds) on the outer-training participants only, for
             hyper-parameter selection (and, for stacking, fusion weights).
Imputation, scaling, tuning and fusion weights are all fitted inside the outer
training fold. `assert_no_group_overlap` is called on every split.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, GroupKFold

from src.config import CV_REPEAT_SEEDS, INNER_FOLDS, OUTER_FOLDS
from src.evaluation.metrics import CLASSIFICATION_METRICS, classification_metrics, pairwise_accuracy
from src.fusion.fusion import LogitPoolFusion
from src.models.zoo import make_classifier


@dataclass
class ExperimentSpec:
    """One experiment = modality set + fusion strategy + learning algorithm."""
    modalities: tuple
    strategy: str            # 'single' | 'early' | 'late' | 'stacked'
    model: str
    columns: dict = field(default_factory=dict)   # modality -> feature list
    feature_override: tuple | None = None         # e.g. a one-feature baseline
    label: str | None = None

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        return f"{'+'.join(self.modalities)} | {self.strategy} | {self.model}"

    def feature_list(self) -> list[str]:
        if self.feature_override:
            return list(self.feature_override)
        return [c for m in self.modalities for c in self.columns[m]]


def repeated_group_splits(groups: np.ndarray, n_splits: int = OUTER_FOLDS, seeds=CV_REPEAT_SEEDS):
    groups = np.asarray(groups)
    out = []
    for r, seed in enumerate(seeds):
        gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for f, (tr, te) in enumerate(gkf.split(np.zeros(len(groups)), groups=groups)):
            assert_no_group_overlap(groups, tr, te)
            out.append((r, f, tr, te))
    return out


def assert_no_group_overlap(groups: np.ndarray, train_idx, test_idx) -> None:
    overlap = set(np.asarray(groups)[train_idx]) & set(np.asarray(groups)[test_idx])
    if overlap:
        raise AssertionError(f"participant leakage between train and test: {sorted(overlap)[:5]}")


def build_estimator(spec: ExperimentSpec, seed: int):
    """Return an unfitted estimator for the spec (tuning happens in fit_estimator)."""
    if spec.strategy in ("late", "stacked"):
        mods = {m: spec.columns[m] for m in spec.modalities}
        return LogitPoolFusion(modalities=mods, base=spec.model, weighting="equal" if spec.strategy == "late"
                               else "stacked", random_state=seed)
    return make_classifier(spec.model, seed)


def fit_estimator(spec: ExperimentSpec, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, seed: int):
    est = build_estimator(spec, seed)
    if spec.strategy in ("late", "stacked"):
        est.fit(X, y, groups=groups)
        return est, {m: v for m, v in est.best_params_.items()}
    pipe, grid = est
    cols = spec.feature_list()
    if grid:
        gs = GridSearchCV(pipe, grid, cv=GroupKFold(INNER_FOLDS, shuffle=True, random_state=seed),
                          scoring="roc_auc", n_jobs=1)
        gs.fit(X[cols], y, groups=groups)
        return gs.best_estimator_, gs.best_params_
    pipe.fit(X[cols], y)
    return pipe, {}


def predict_estimator(spec: ExperimentSpec, est, X: pd.DataFrame) -> np.ndarray:
    if spec.strategy in ("late", "stacked"):
        return est.predict_proba(X)[:, 1]
    return est.predict_proba(X[spec.feature_list()])[:, 1]


def _run_fold(spec, X, y, groups, r, f, tr, te, seed):
    est, params = fit_estimator(spec, X.iloc[tr], y[tr], groups[tr], seed)
    prob = predict_estimator(spec, est, X.iloc[te])
    m = classification_metrics(y[te], prob)
    m["pairwise_accuracy"] = pairwise_accuracy(groups[te], y[te], prob)
    weights = getattr(est, "weights_", None)
    row = {"experiment": spec.name, "modalities": "+".join(spec.modalities), "strategy": spec.strategy,
           "model": spec.model, "repeat": r, "fold": f, "n_train": len(tr), "n_test": len(te),
           "best_params": str(params), **m}
    if weights:
        row.update({f"w_{k}": v for k, v in weights.items()})
    oof = pd.DataFrame({"experiment": spec.name, "repeat": r, "fold": f, "pid": groups[te],
                        "y": y[te], "prob": prob})
    return row, oof


def run_cv(spec: ExperimentSpec, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, splits,
           n_jobs: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = CV_REPEAT_SEEDS
    res = Parallel(n_jobs=n_jobs)(
        delayed(_run_fold)(spec, X, y, groups, r, f, tr, te, seeds[r]) for r, f, tr, te in splits)
    folds = pd.DataFrame([a for a, _ in res])
    oof = pd.concat([b for _, b in res], ignore_index=True)
    return folds, oof


def pooled_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    """Metrics on pooled out-of-fold predictions, one row per repeat."""
    rows = []
    for (exp, r), g in oof.groupby(["experiment", "repeat"]):
        m = classification_metrics(g["y"], g["prob"])
        m["pairwise_accuracy"] = pairwise_accuracy(g["pid"], g["y"], g["prob"])
        rows.append({"experiment": exp, "repeat": r, **m})
    return pd.DataFrame(rows)


METRICS_REPORTED = CLASSIFICATION_METRICS + ["pairwise_accuracy"]
