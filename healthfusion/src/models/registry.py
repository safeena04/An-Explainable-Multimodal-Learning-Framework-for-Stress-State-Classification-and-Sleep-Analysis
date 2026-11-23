"""Model registry: saving/loading final models together with their model cards.

A model card records what the model is, how it was validated, and whether it
passed the pre-specified validation criterion. The API only serves predictions
from models whose card says `validated: true`.

Validation criteria (fixed before looking at the final results):
* stress classifiers: mean outer-CV ROC-AUC > 0.5 AND Nadeau-Bengio corrected
  resampled t-test against 0.5 gives p < 0.05 (25 participant-grouped folds).
* sleep regressors: LOSO R^2 > 0 AND label-permutation p < 0.05 after Holm
  adjustment over all permutation-tested configurations of the primary target
  (1000 permutations for Ridge, 200 for random forest).
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import sklearn

from src.config import MODELS_DIR

REGISTRY_PATH = MODELS_DIR / "registry.json"


def environment() -> dict:
    import numpy
    import pandas
    import shap
    import xgboost
    import neurokit2
    return {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
            "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "shap": shap.__version__,
            "neurokit2": neurokit2.__version__}


def save_model(task: str, key: str, artifact: dict, card: dict) -> Path:
    d = MODELS_DIR / task
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{key}.joblib"
    joblib.dump(artifact, path)
    (d / f"{key}.json").write_text(json.dumps(card, indent=2, default=str))
    return path


def save_card_only(task: str, key: str, card: dict) -> Path:
    d = MODELS_DIR / task
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{key}.json"
    p.write_text(json.dumps(card, indent=2, default=str))
    return p


def write_registry(entries: list[dict]) -> None:
    REGISTRY_PATH.write_text(json.dumps({"models": entries, "environment": environment()}, indent=2, default=str))


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {"models": [], "environment": {}}
    return json.loads(REGISTRY_PATH.read_text())


def load_artifact(task: str, key: str) -> dict:
    return joblib.load(MODELS_DIR / task / f"{key}.joblib")
