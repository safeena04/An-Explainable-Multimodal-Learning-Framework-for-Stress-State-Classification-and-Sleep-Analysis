"""Fusion estimators on synthetic data (code tests only)."""
import numpy as np
import pandas as pd

from src.fusion.fusion import LogitPoolFusion, fit_nonneg_logistic


def _data(n_groups=40, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), 2)
    y = np.tile([0, 1], n_groups)
    X = pd.DataFrame({"a1": y + rng.normal(0, 1, y.size), "a2": rng.normal(0, 1, y.size),
                      "b1": rng.normal(0, 1, y.size), "b2": rng.normal(0, 1, y.size)})
    return X, y, g


def test_stacking_learns_informative_modality():
    X, y, g = _data()
    m = LogitPoolFusion({"A": ["a1", "a2"], "B": ["b1", "b2"]}, base="logreg", weighting="stacked").fit(X, y, groups=g)
    assert m.weights_["A"] > m.weights_["B"] >= 0
    p = m.predict_proba(X)[:, 1]
    assert p.shape == (len(y),) and ((p >= 0) & (p <= 1)).all()


def test_missing_modality_rows_still_scored():
    X, y, g = _data()
    m = LogitPoolFusion({"A": ["a1", "a2"], "B": ["b1", "b2"]}, base="logreg", weighting="equal").fit(X, y, groups=g)
    Xm = X.copy()
    Xm.loc[:4, ["b1", "b2"]] = np.nan
    Xm.loc[5:9, ["a1", "a2", "b1", "b2"]] = np.nan
    p = m.predict_proba(Xm)[:, 1]
    assert np.isfinite(p).all()
    assert np.allclose(p[5:10], 0.5)  # no evidence at all -> 0.5


def test_nonneg_logistic_bounds():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(200, 2))
    y = (Z[:, 0] - Z[:, 1] + rng.normal(0, 0.5, 200) > 0).astype(int)
    _, w = fit_nonneg_logistic(Z, y)
    assert (w >= 0).all() and w[0] > w[1]
