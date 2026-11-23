"""Stress-state classification endpoints (QoL_Stress)."""
import numpy as np
from fastapi import APIRouter, HTTPException

from api import services as S
from api.schemas import StressFeatureRequest, StressRawRequest
from src.config import (QOL_ECG_MAX_ARTEFACT_PCT, QOL_ECG_MIN_BEATS, QOL_EDA_MIN_VALID,
                        QOL_HR_MIN_SAMPLES)
from src.preprocessing.qol_signals import process_ecg, process_eda, process_hr_window

router = APIRouter(prefix="/stress", tags=["stress classification"])


def _raise_for(result: dict):
    err = result.get("error")
    if err == "no_complete_modality":
        raise HTTPException(422, detail={"message": "No modality has all of its features. Supply every feature "
                                                    "of at least one modality.", **result})
    if err == "no_model_for_modalities":
        raise HTTPException(422, detail={"message": "No registered model for this modality set.", **result})
    if err == "model_not_validated":
        raise HTTPException(422, detail={"message": "The model for this modality set did not pass validation "
                                                    "(its cross-validated AUC was not reliably above chance), "
                                                    "so it is not served. Add another modality.", **result})


@router.get("/models")
def models():
    """Every stress model in the registry, with its validation verdict."""
    out = []
    for e in S.stress_entries():
        c = S.card("stress", e["key"])
        out.append({"key": e["key"], "modalities": e["modalities"], "strategy": c["strategy"],
                    "algorithm": c["algorithm"], "features": c["features"], "validation": c["validation"],
                    "served": c["validation"]["validated"]})
    return S.clean({"models": out})


@router.get("/models/{key}")
def model_card(key: str):
    try:
        return S.clean(S.card("stress", key))
    except S.ArtifactMissing:
        raise HTTPException(404, f"unknown model '{key}'")


@router.post("/predict")
def predict(req: StressFeatureRequest):
    """Classify one phase-level observation from pre-computed features."""
    result = S.predict_stress(req.features.model_dump(), explain=req.explain)
    _raise_for(result)
    return result


@router.post("/predict/raw")
def predict_raw(req: StressRawRequest):
    """Extract features from raw signals with the training pipeline's code, then classify."""
    feats, extraction = {}, {}
    if req.ecg:
        e = process_ecg(np.asarray(req.ecg.samples), req.ecg.sampling_rate)
        extraction["ecg"] = {k: e[k] for k in ("ecg_n_beats", "ecg_quality", "ecg_duration_s", "ecg_pct_artefact",
                                               "ecg_error")}
        # identical QC rules to the training pipeline (src/preprocessing/qol_preprocess.py)
        if (e["ecg_error"] is None and e["ecg_n_beats"] >= QOL_ECG_MIN_BEATS
                and e["ecg_pct_artefact"] <= QOL_ECG_MAX_ARTEFACT_PCT):
            feats.update({k: e[k] for k in ("ecg_hr", "ecg_rmssd", "ecg_sdnn", "ecg_pnn50", "ecg_hr_range")})
        else:
            extraction["ecg"]["warning"] = "ECG failed quality control (too few beats or > 20 % artefacts): not used"
        if req.ecg.sampling_rate != 250:
            extraction["ecg"]["warning"] = "training ECG was 250 Hz; HRV features are in ms and fs-independent"
    if req.eda:
        d = process_eda(np.asarray(req.eda.scl), np.asarray(req.eda.activation),
                        None if req.eda.valid is None else np.asarray(req.eda.valid), req.eda.sampling_rate)
        extraction["eda"] = {"n_valid": d["eda_n_valid"], "n_rows": d["eda_n_rows"]}
        if d["eda_n_valid"] >= QOL_EDA_MIN_VALID:
            feats.update({k: d[k] for k in ("eda_scl_mean", "eda_scl_std", "eda_scl_slope", "eda_scl_range",
                                            "eda_resp_rate")})
        else:
            extraction["eda"]["warning"] = "fewer than 60 valid samples: EDA not used"
    if req.heart_rate:
        h = process_hr_window(np.asarray(req.heart_rate.timestamps_s), np.asarray(req.heart_rate.bpm))
        extraction["heart_rate"] = {"n_samples": h["fb_n_samples"], "span_min": h["fb_span_min"]}
        if h["fb_n_samples"] >= QOL_HR_MIN_SAMPLES:
            feats.update({k: h[k] for k in ("fb_hr_mean", "fb_hr_std", "fb_hr_p05", "fb_hr_p95", "fb_hr_slope")})
        else:
            extraction["heart_rate"]["warning"] = "fewer than 60 samples: heart rate not used"
    result = S.predict_stress(feats, explain=req.explain)
    if "error" in result:
        result["signal_quality"] = extraction
        _raise_for(result)
    result["extracted_features"] = S.clean(feats)
    result["signal_quality"] = S.clean(extraction)
    return result


@router.get("/statistics")
def statistics():
    """Paired relax-vs-stress tests per feature and the STAI manipulation check."""
    return {"manipulation_check": S.read_json("qol_manipulation_check.json"),
            "paired_tests": S.records(S.read_csv("qol_paired_tests.csv")),
            "external_validation": S.read_json("qol_external_validation.json")}
