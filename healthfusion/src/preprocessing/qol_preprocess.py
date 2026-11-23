"""QoL_Stress preprocessing: decode, quality-control, align and aggregate to
one row per participant x phase (132 possible rows).

Outputs (data/processed/):
  qol_phase_features.csv   features + label + modality availability flags
  qol_qc_log.csv           every QC decision per participant x phase x modality
  qol_questionnaires.csv   STAI per phase, PSS-14, demographics (NOT model inputs)
  qol_ecg_beats.json       nothing; R-peaks are not stored (re-derivable)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (PROCESSED_DIR, QOL_ECG_EXCLUDE_USERNAMES, QOL_ECG_FS, QOL_ECG_MAX_ARTEFACT_PCT,
                        QOL_ECG_MAX_HR_DEVIATION,
                        QOL_ECG_MIN_BEATS, QOL_EDA_MIN_VALID, QOL_HR_MIN_SAMPLES, QOL_HR_WINDOW_MIN)
from src.data.qol_loader import load_qol, parse_waveform, questionnaire_item_scores
from src.preprocessing.qol_signals import (ECG_FEATURES, EDA_FEATURES, HR_FEATURES, process_ecg,
                                           process_eda, process_hr_window)

MODALITIES = {"ECG": ECG_FEATURES, "EDA": EDA_FEATURES, "HR": HR_FEATURES}


def _ecg_table(d: dict, qc: list) -> pd.DataFrame:
    rows = []
    for _, r in d["ecg"].iterrows():
        wp = parse_waveform(r["waveform_samples"])
        feats = process_ecg(wp.samples, QOL_ECG_FS)
        reason = None
        if r["username"] in QOL_ECG_EXCLUDE_USERNAMES:
            reason = "duplicated recording belongs to another participant (see config)"
        elif feats["ecg_error"]:
            reason = feats["ecg_error"]
        elif feats["ecg_n_beats"] < QOL_ECG_MIN_BEATS:
            reason = f"only {feats['ecg_n_beats']} beats detected"
        elif feats["ecg_pct_artefact"] > QOL_ECG_MAX_ARTEFACT_PCT:
            reason = f"{feats['ecg_pct_artefact']:.0f} % of RR intervals rejected as artefacts"
        elif abs(feats["ecg_hr"] - r["heart_rate"]) > QOL_ECG_MAX_HR_DEVIATION:
            reason = f"detected HR {feats['ecg_hr']:.1f} vs device {r['heart_rate']} bpm"
        row = {"pid": r["pid"], "phase": r["phase"], "ecg_device_hr": r["heart_rate"],
               "ecg_device_class": r["result_classification"], "ecg_repaired_samples": wp.n_repaired,
               "ecg_quality": feats["ecg_quality"], "ecg_n_beats": feats["ecg_n_beats"],
               "ecg_pct_artefact": feats["ecg_pct_artefact"]}
        row.update({k: (np.nan if reason else feats[k]) for k in ECG_FEATURES})
        rows.append(row)
        qc.append({"pid": r["pid"], "phase": r["phase"], "modality": "ECG", "included": reason is None,
                   "reason": reason or "passed", "detected_hr": feats["ecg_hr"], "device_hr": r["heart_rate"]})
    return pd.DataFrame(rows)


def _eda_table(d: dict, qc: list, all_keys: pd.MultiIndex) -> pd.DataFrame:
    rows = []
    for (pid, phase), g in d["eda"].sort_values(["pid", "phase", "row_in_session"]).groupby(["pid", "phase"]):
        feats = process_eda(g["scl_avg"].to_numpy(), g["activation"].to_numpy(), g["valid_data"].to_numpy())
        reason = None if feats["eda_n_valid"] >= QOL_EDA_MIN_VALID else f"only {feats['eda_n_valid']} valid samples"
        row = {"pid": pid, "phase": phase, "eda_n_valid": feats["eda_n_valid"]}
        row.update({k: (np.nan if reason else feats[k]) for k in EDA_FEATURES})
        rows.append(row)
        qc.append({"pid": pid, "phase": phase, "modality": "EDA", "included": reason is None, "reason": reason or "passed"})
    present = {(r["pid"], r["phase"]) for r in rows}
    for pid, phase in all_keys:
        if (pid, phase) not in present:
            qc.append({"pid": pid, "phase": phase, "modality": "EDA", "included": False, "reason": "no EDA session in release"})
    return pd.DataFrame(rows)


def _hr_table(d: dict, qc: list) -> pd.DataFrame:
    stai = d["stai"].pivot(index="pid", columns="phase", values="t_s")
    hs = d["hr_second"]
    w = QOL_HR_WINDOW_MIN * 60
    rows = []
    for pid, g in hs.groupby("pid"):
        for phase in ("relax", "stress"):
            end = stai.loc[pid, phase]
            feats = process_hr_window(g["t_s"].to_numpy(), g["value"].to_numpy(), end - w, end)
            reason = None if feats["fb_n_samples"] >= QOL_HR_MIN_SAMPLES else \
                f"only {feats['fb_n_samples']} Fitbit samples in the {QOL_HR_WINDOW_MIN:.0f}-min window"
            row = {"pid": pid, "phase": phase, "fb_n_samples": feats["fb_n_samples"], "fb_span_min": feats["fb_span_min"]}
            row.update({k: (np.nan if reason else feats[k]) for k in HR_FEATURES})
            rows.append(row)
            qc.append({"pid": pid, "phase": phase, "modality": "HR", "included": reason is None, "reason": reason or "passed"})
    return pd.DataFrame(rows)


def questionnaires(d: dict) -> pd.DataFrame:
    stai = d["stai"].pivot(index="pid", columns="phase", values="score").add_prefix("stai_")
    pss = d["pss"].set_index("pid")["score"].rename("pss14")
    demo = d["participants"].set_index("pid")[["sex", "age", "education_level", "job_category"]]
    q = demo.join(pss).join(stai)
    q["stai_delta"] = q["stai_stress"] - q["stai_relax"]
    return q.reset_index()


def build_qol(save: bool = True) -> dict[str, pd.DataFrame]:
    d = load_qol()
    pids = sorted(d["participants"]["pid"])
    keys = pd.MultiIndex.from_product([pids, ["relax", "stress"]], names=["pid", "phase"])
    qc: list = []
    ecg = _ecg_table(d, qc)
    eda = _eda_table(d, qc, keys)
    hr = _hr_table(d, qc)

    feat = pd.DataFrame(index=keys).reset_index()
    for t in (ecg, eda, hr):
        feat = feat.merge(t, on=["pid", "phase"], how="left")
    feat["label"] = (feat["phase"] == "stress").astype(int)
    for mod, cols in MODALITIES.items():
        feat[f"has_{mod}"] = feat[cols].notna().all(axis=1)
    # a modality is usable for paired analysis only if present in BOTH phases
    for mod in MODALITIES:
        both = feat.groupby("pid")[f"has_{mod}"].transform("all")
        feat[f"pair_{mod}"] = both
    feat["complete_case"] = feat[[f"pair_{m}" for m in MODALITIES]].all(axis=1)

    qcdf = pd.DataFrame(qc).sort_values(["pid", "phase", "modality"])
    qn = questionnaires(d)
    out = {"features": feat, "qc": qcdf, "questionnaires": qn}
    if save:
        feat.to_csv(PROCESSED_DIR / "qol_phase_features.csv", index=False)
        qcdf.to_csv(PROCESSED_DIR / "qol_qc_log.csv", index=False)
        qn.to_csv(PROCESSED_DIR / "qol_questionnaires.csv", index=False)
    return out


def stai_item_matrix() -> pd.DataFrame:
    """Item-level STAI answers (for reliability checks only)."""
    d = load_qol()
    items = [c for c in d["stai"].columns if c not in ("username", "time", "phase", "score", "pid", "t_s")]
    m = questionnaire_item_scores(d["stai"], items)
    m.insert(0, "phase", d["stai"]["phase"].values)
    m.insert(0, "pid", d["stai"]["pid"].values)
    return m
