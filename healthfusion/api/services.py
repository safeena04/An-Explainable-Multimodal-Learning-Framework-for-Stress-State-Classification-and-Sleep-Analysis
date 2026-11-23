"""Artifact access layer for the API.

Everything the API returns is read from files produced by the pipelines
(results/, models/, data/processed/). The API never recomputes or alters a
research result; it serves them and runs inference with registered models.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import METRICS_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from src.explainability.shap_analysis import explain_any
from src.models.registry import load_artifact, load_registry
from src.preprocessing.qol_preprocess import MODALITIES as QOL_MODALITIES

QOL_FEATURE_TO_MODALITY = {c: m for m, cols in QOL_MODALITIES.items() for c in cols}
DISCLAIMER = ("Research prototype trained on a small sample of healthy adults in one laboratory protocol. "
              "Not a medical device; outputs must not be used for diagnosis or clinical decisions.")


class ArtifactMissing(RuntimeError):
    """Raised when a pipeline output the endpoint needs has not been generated yet."""


def clean(obj):
    """Make numpy/pandas values JSON-safe (NaN/inf -> None)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _require(path: Path) -> Path:
    if not path.exists():
        raise ArtifactMissing(f"{path.name} not found; run the pipelines first (python -m src.pipelines.run_all)")
    return path


def read_csv(name: str, folder: Path = METRICS_DIR) -> pd.DataFrame:
    return pd.read_csv(_require(folder / name))


def read_json(name: str, folder: Path = METRICS_DIR) -> dict:
    return json.loads(_require(folder / name).read_text())


def records(df: pd.DataFrame) -> list[dict]:
    return clean(df.to_dict("records"))


# --------------------------------------------------------------------- datasets
def inspection() -> dict:
    return read_json("phase1_inspection.json", REPORTS_DIR)


def qol_qc() -> pd.DataFrame:
    return read_csv("qol_qc_log.csv", PROCESSED_DIR)


def mmash_qc() -> pd.DataFrame:
    return read_csv("mmash_qc_log.csv", PROCESSED_DIR)


def stacking_weights() -> list[dict]:
    """Tidy view of the learned stacking weights (mean and SD over outer folds)."""
    w = pd.read_csv(_require(METRICS_DIR / "qol_stacking_weights.csv"), header=[0, 1], index_col=0)
    out = []
    for exp, row in w.iterrows():
        for (col, stat), v in row.items():
            if stat == "mean":
                out.append({"experiment": exp, "modality": col.replace("w_", ""), "weight_mean": v,
                            "weight_sd": row[(col, "std")]})
    return clean([r for r in out if r["weight_mean"] == r["weight_mean"]])


# --------------------------------------------------------------------- models
@lru_cache(maxsize=1)
def registry() -> dict:
    reg = load_registry()
    if not reg["models"]:
        raise ArtifactMissing("models/registry.json is empty; run python -m src.pipelines.train_final_models")
    return reg


def stress_entries() -> list[dict]:
    return [m for m in registry()["models"] if m["task"] == "stress"]


def sleep_entry() -> dict | None:
    return next((m for m in registry()["models"] if m["task"] == "sleep"), None)


@lru_cache(maxsize=16)
def stress_artifact(key: str) -> dict:
    return load_artifact("stress", key)


def card(task: str, key: str) -> dict:
    return json.loads(_require(MODELS_DIR / task / f"{key}.json").read_text())


def stress_model_for(modalities: set[str]) -> dict | None:
    for e in stress_entries():
        if set(e["modalities"]) == modalities:
            return e
    return None


def predict_stress(features: dict, explain: bool = True) -> dict:
    """Pick the registered model for exactly the modalities supplied, predict and explain."""
    present = {m for m, cols in QOL_MODALITIES.items()
               if all(features.get(c) is not None and np.isfinite(features.get(c)) for c in cols)}
    partial = {m: [c for c in cols if features.get(c) is None]
               for m, cols in QOL_MODALITIES.items()
               if m not in present and any(features.get(c) is not None for c in cols)}
    if not present:
        return {"error": "no_complete_modality", "partial_modalities": partial,
                "required_features": QOL_MODALITIES}
    entry = stress_model_for(present)
    if entry is None:
        return {"error": "no_model_for_modalities", "modalities": sorted(present)}
    if not entry["validated"]:
        c = card("stress", entry["key"])
        return {"error": "model_not_validated", "modalities": sorted(present), "model_key": entry["key"],
                "validation": c["validation"],
                "validated_modality_sets": [e["modalities"] for e in stress_entries() if e["validated"]]}
    art = stress_artifact(entry["key"])
    cols = art["spec"]["features"]
    X = pd.DataFrame([{c: features[c] for c in cols}])
    est = art["estimator"]
    prob = float(est.predict_proba(X)[:, 1][0]) if art["spec"]["strategy"] in ("late", "stacked") \
        else float(est.predict_proba(X[cols])[:, 1][0])
    out = {"model_key": entry["key"], "modalities_used": sorted(present), "ignored_partial_modalities": partial,
           "strategy": art["spec"]["strategy"], "algorithm": art["spec"]["model"],
           "probability_stress": prob, "threshold": 0.5,
           "predicted_label": "stress" if prob >= 0.5 else "relax",
           "validation": art["card"]["validation"], "disclaimer": DISCLAIMER}
    if explain:
        vals, base = explain_any(est, art["background"], X, cols)
        contrib = [{"feature": c, "modality": QOL_FEATURE_TO_MODALITY[c], "value": features[c],
                    "shap": float(vals[c].iloc[0])} for c in cols]
        contrib.sort(key=lambda d: abs(d["shap"]), reverse=True)
        scale = "probability" if art["spec"]["model"] == "random_forest" and art["spec"]["strategy"] in (
            "single", "early") else "log-odds"
        out["explanation"] = {
            "method": "SHAP", "output_scale": scale, "base_value": float(base[0]),
            "contributions": contrib,
            "by_modality": {m: float(sum(d["shap"] for d in contrib if d["modality"] == m)) for m in sorted(present)},
            "note": "Attributions describe the model's behaviour, not physiological causes.",
        }
    return clean(out)
