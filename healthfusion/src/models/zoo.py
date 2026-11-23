"""Model definitions and (small) hyper-parameter grids.

Grids are deliberately small: with 45-66 participants, large grids overfit the
inner validation folds. Every pipeline starts with median imputation and
standard scaling *inside* the pipeline, so they are re-fitted on each training
fold and never see test data.
"""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import GLOBAL_SEED

CLASSIFIERS = ["dummy", "logreg", "svm_rbf", "random_forest", "xgboost", "gradient_boosting"]


def _pipe(est) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", est)])


def make_classifier(name: str, seed: int = GLOBAL_SEED) -> tuple[Pipeline, dict]:
    if name == "dummy":
        return _pipe(DummyClassifier(strategy="prior")), {}
    if name == "logreg":
        return _pipe(LogisticRegression(max_iter=5000, solver="lbfgs")), {"model__C": [0.01, 0.1, 1.0, 10.0]}
    if name == "svm_rbf":
        # Platt-scaled probabilities via CalibratedClassifierCV (replacement for the
        # deprecated SVC(probability=True)); calibration CV runs on training data only.
        svc = CalibratedClassifierCV(SVC(kernel="rbf", random_state=seed), method="sigmoid", cv=3, ensemble=False)
        return (_pipe(svc), {"model__estimator__C": [0.1, 1.0, 10.0], "model__estimator__gamma": ["scale", 0.01]})
    if name == "random_forest":
        return (_pipe(RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=1)),
                {"model__max_depth": [3, None]})
    if name == "xgboost":
        return (_pipe(XGBClassifier(n_estimators=150, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                    random_state=seed, n_jobs=1, eval_metric="logloss", verbosity=0)),
                {"model__max_depth": [2, 3], "model__min_child_weight": [1, 3]})
    if name == "gradient_boosting":
        return (_pipe(GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, subsample=0.8,
                                                 random_state=seed)),
                {"model__max_depth": [2, 3]})
    raise ValueError(f"unknown classifier {name}")


REGRESSORS = ["dummy_mean", "ridge", "random_forest"]


def make_regressor(name: str, seed: int = GLOBAL_SEED) -> Pipeline:
    """Regressors for n=21. RidgeCV selects alpha by efficient leave-one-out on the
    training fold only (nested by construction). RF is fixed and shallow (no
    tuning: there is not enough data to tune it honestly)."""
    if name == "dummy_mean":
        return _pipe(DummyRegressor(strategy="mean"))
    if name == "ridge":
        return _pipe(RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0]))
    if name == "random_forest":
        return _pipe(RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=3,
                                           random_state=seed, n_jobs=1))
    raise ValueError(f"unknown regressor {name}")
