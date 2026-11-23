"""Multimodal fusion strategies.

Early fusion  : concatenate the (imputed, scaled) features of all modalities and
                train one model. Implemented by simply passing all columns to a
                standard pipeline from src.models.zoo.

Late fusion   : train one model per modality and pool their predictions in
                log-odds space (a log-linear opinion pool):
                    logit p = sum_m w_m * logit p_m
                Equal weights (w_m = 1 / #available modalities) give the
                unweighted pool. Pooling log-odds (rather than probabilities)
                makes the fused score additive across modalities, so feature
                attributions of linear base models remain exact.

Weighted fusion (stacking, Wolpert 1992): the weights are learned. Out-of-fold
                modality logits are produced *inside the training fold* by a
                participant-grouped inner CV; a logistic meta-model with
                non-negative weights is fitted on them by minimising the
                L2-penalised log-loss. Test data are never used for weights.

Missing modalities: a modality absent for a row is simply left out of the pool
(equal) or contributes 0 log-odds, i.e. no evidence (stacking). This lets the
late-fusion models score any subset of sensors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict

from src.config import GLOBAL_SEED, INNER_FOLDS
from src.models.zoo import make_classifier

EPS = 1e-3


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_nonneg_logistic(Z: np.ndarray, y: np.ndarray, l2: float = 1e-2) -> tuple[float, np.ndarray]:
    """Logistic regression y ~ sigmoid(b0 + Z w) with w >= 0 (L-BFGS-B)."""
    n, m = Z.shape

    def loss(theta):
        b0, w = theta[0], theta[1:]
        z = b0 + Z @ w
        ll = np.mean(np.logaddexp(0, z) - y * z)
        return ll + l2 * np.sum(w ** 2)

    def grad(theta):
        b0, w = theta[0], theta[1:]
        r = _sigmoid(b0 + Z @ w) - y
        return np.concatenate([[r.mean()], Z.T @ r / n + 2 * l2 * w])

    x0 = np.concatenate([[0.0], np.full(m, 1.0 / m)])
    bounds = [(None, None)] + [(0.0, None)] * m
    res = minimize(loss, x0, jac=grad, method="L-BFGS-B", bounds=bounds)
    return float(res.x[0]), res.x[1:]


class LogitPoolFusion(BaseEstimator, ClassifierMixin):
    """Late (equal) or stacked (learned non-negative weights) fusion."""

    def __init__(self, modalities: dict | None = None, base: str = "logreg", weighting: str = "equal",
                 tune: bool = True, random_state: int = GLOBAL_SEED):
        self.modalities = modalities
        self.base = base
        self.weighting = weighting
        self.tune = tune
        self.random_state = random_state

    def _inner_cv(self):
        return GroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=self.random_state)

    def fit(self, X: pd.DataFrame, y, groups=None):
        y = np.asarray(y).astype(int)
        groups = np.arange(len(y)) if groups is None else np.asarray(groups)
        self.classes_ = np.array([0, 1])
        self.models_, self.best_params_ = {}, {}
        oof = {}
        for mod, cols in self.modalities.items():
            avail = X[cols].notna().all(axis=1).to_numpy()
            Xm, ym, gm = X.loc[avail, cols], y[avail], groups[avail]
            pipe, grid = make_classifier(self.base, self.random_state)
            if self.tune and grid and len(np.unique(gm)) >= INNER_FOLDS:
                gs = GridSearchCV(pipe, grid, cv=self._inner_cv(), scoring="roc_auc", n_jobs=1)
                gs.fit(Xm, ym, groups=gm)
                pipe = gs.best_estimator_
                self.best_params_[mod] = gs.best_params_
            else:
                pipe.fit(Xm, ym)
            self.models_[mod] = pipe
            if self.weighting == "stacked":
                col = np.full(len(y), np.nan)
                col[avail] = cross_val_predict(clone(pipe), Xm, ym, groups=gm, cv=self._inner_cv(),
                                               method="predict_proba")[:, 1]
                oof[mod] = col
        if self.weighting == "stacked":
            Z = np.column_stack([np.nan_to_num(_logit(oof[m]), nan=0.0) for m in self.modalities])
            self.intercept_, w = fit_nonneg_logistic(Z, y)
            self.weights_ = dict(zip(self.modalities, w))
        else:
            self.intercept_ = 0.0
            self.weights_ = {m: 1.0 / len(self.modalities) for m in self.modalities}
        return self

    def modality_logits(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for mod, cols in self.modalities.items():
            col = np.full(len(X), np.nan)
            avail = X[cols].notna().all(axis=1).to_numpy()
            if avail.any():
                col[avail] = _logit(self.models_[mod].predict_proba(X.loc[avail, cols])[:, 1])
            out[mod] = col
        return pd.DataFrame(out, index=X.index)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        L = self.modality_logits(X)
        if self.weighting == "stacked":
            w = np.array([self.weights_[m] for m in self.modalities])
            return self.intercept_ + np.nan_to_num(L.to_numpy(), nan=0.0) @ w
        n_avail = L.notna().sum(axis=1).to_numpy()
        return np.where(n_avail > 0, L.sum(axis=1, skipna=True).to_numpy() / np.maximum(n_avail, 1), 0.0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
