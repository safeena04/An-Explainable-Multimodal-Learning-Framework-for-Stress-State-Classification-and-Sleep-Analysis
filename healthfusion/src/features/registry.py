"""Feature dictionary: one documented entry per model input feature.

`write_feature_dictionary()` exports data/processed/feature_dictionary.csv and
docs/FEATURE_DICTIONARY.md so that every feature in every result table can be
traced to its raw source and transformation.
"""
from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DIR, PROJECT_ROOT

F = []


def _add(name, dataset, modality, unit, source, transform, note=""):
    F.append(dict(feature=name, dataset=dataset, modality=modality, unit=unit, source=source,
                  transformation=transform, note=note))


# ---------------- QoL_Stress (one value per participant x phase) ----------------
_ecg_src = "ecg.csv: waveform_samples (30 s, 250 Hz verified)"
_ecg_tf = ("NeuroKit2 ecg_clean -> ecg_peaks (artefact-corrected) -> RR 300-2000 ms and within 20 % of local "
           "median (strip dropped if > 20 % rejected) -> ")
_add("ecg_hr", "QoL_Stress", "ECG", "bpm", _ecg_src, _ecg_tf + "60000 / mean NN")
_add("ecg_rmssd", "QoL_Stress", "ECG", "ms", _ecg_src, _ecg_tf + "RMSSD", "ultra-short (30 s) HRV")
_add("ecg_sdnn", "QoL_Stress", "ECG", "ms", _ecg_src, _ecg_tf + "SDNN", "30-s SDNN has low reliability")
_add("ecg_pnn50", "QoL_Stress", "ECG", "%", _ecg_src, _ecg_tf + "pNN50", "~30 beats: coarse resolution")
_add("ecg_hr_range", "QoL_Stress", "ECG", "bpm", _ecg_src, _ecg_tf + "max - min instantaneous HR",
     "reflects respiratory sinus arrhythmia")
_eda_src = "eda.csv: scl_avg, activation, valid_data (~160 s per phase, 1 row/s verified)"
_add("eda_scl_mean", "QoL_Stress", "EDA", "device units", _eda_src, "mean SCL over valid rows", "unit undocumented")
_add("eda_scl_std", "QoL_Stress", "EDA", "device units", _eda_src, "SD of SCL over valid rows")
_add("eda_scl_slope", "QoL_Stress", "EDA", "units/min", _eda_src, "OLS slope of SCL vs time (row index / 1 Hz)")
_add("eda_scl_range", "QoL_Stress", "EDA", "device units", _eda_src, "95th - 5th percentile of SCL")
_add("eda_resp_rate", "QoL_Stress", "EDA", "responses/min", _eda_src, "sum(activation) / valid minutes")
_hr_src = "fitbit_heart_rate_per_second.csv (irregular, median 2 s); window = 8 min ending at STAI submission"
_add("fb_hr_mean", "QoL_Stress", "HR", "bpm", _hr_src, "mean HR in phase window")
_add("fb_hr_std", "QoL_Stress", "HR", "bpm", _hr_src, "SD of HR in window", "not HRV (optical, smoothed)")
_add("fb_hr_p05", "QoL_Stress", "HR", "bpm", _hr_src, "5th percentile HR")
_add("fb_hr_p95", "QoL_Stress", "HR", "bpm", _hr_src, "95th percentile HR")
_add("fb_hr_slope", "QoL_Stress", "HR", "bpm/min", _hr_src, "OLS slope of HR vs time in window")

# ---------------- MMASH (one value per participant, pre-sleep window) ----------------
_rr_src = "RR.csv ibi_s (Polar H7), recording start -> first In-Bed time"
_rr_tf = "IBI 0.3-2.0 s and within 20 % of local median; differences only between adjacent valid beats"
_add("phys_hr_mean", "MMASH", "Physiological", "bpm", _rr_src, "60 / mean valid IBI")
_add("phys_rmssd", "MMASH", "Physiological", "ms", _rr_src, _rr_tf + "; median of 5-min segment RMSSD")
_add("phys_sdnn", "MMASH", "Physiological", "ms", _rr_src, _rr_tf + "; median of 5-min segment SDNN")
_add("phys_pnn50", "MMASH", "Physiological", "%", _rr_src, _rr_tf + "; median of 5-min segment pNN50")
_add("phys_hr_evening", "MMASH", "Physiological", "bpm", _rr_src, "mean HR in the 2 h before In-Bed")
_ag_src = "Actigraph.csv (ActiGraph wGT3X-BT, 1 row/s), worn seconds only (Inclinometer Off = 0)"
_add("behav_vm_mean", "MMASH", "Behavioural", "counts", _ag_src, "mean Vector Magnitude, pre-sleep")
_add("behav_steps_per_hour", "MMASH", "Behavioural", "steps/h", _ag_src, "total steps / worn hours, pre-sleep")
_add("behav_frac_standing", "MMASH", "Behavioural", "fraction", _ag_src, "mean Inclinometer Standing")
_add("behav_frac_lying", "MMASH", "Behavioural", "fraction", _ag_src, "mean Inclinometer Lying")
_dia_src = "Activity.csv self-reported diary (codes verified on PhysioNet); code 0 excluded"
_add("behav_mvpa_min", "MMASH", "Behavioural", "min", _dia_src, "minutes in codes 5-6 before In-Bed")
_add("behav_screen_min", "MMASH", "Behavioural", "min", _dia_src, "minutes in codes 8-9 before In-Bed")
_add("behav_caffeine_n", "MMASH", "Behavioural", "count", _dia_src, "number of code-10 entries before In-Bed")
_q_src = "questionnaire.csv (single administration)"
_add("psych_meq", "MMASH", "Psychological", "score 16-86", _q_src, "MEQ chronotype")
_add("psych_stai_state", "MMASH", "Psychological", "score 20-80", _q_src, "STAI-Y1 state anxiety")
_add("psych_stai_trait", "MMASH", "Psychological", "score 20-80", _q_src, "STAI-Y2 trait anxiety; 0 -> missing")
_add("psych_daily_stress", "MMASH", "Psychological", "score 0-406", _q_src, "Daily Stress Inventory")
_add("psych_bis", "MMASH", "Psychological", "score", _q_src, "BIS subscale")
_add("psych_panas_pos_day", "MMASH", "Psychological", "score 5-50", _q_src, "mean PANAS positive at 10/14/18/22 h")
_add("psych_panas_neg_day", "MMASH", "Psychological", "score 5-50", _q_src, "mean PANAS negative at 10/14/18/22 h")

FEATURE_TABLE = pd.DataFrame(F)


def write_feature_dictionary() -> pd.DataFrame:
    FEATURE_TABLE.to_csv(PROCESSED_DIR / "feature_dictionary.csv", index=False)
    lines = ["# Feature dictionary", "",
             "Every model input, its modality, unit, raw source and exact transformation.",
             "Generated by `src/features/registry.py`; do not edit by hand.", ""]
    for ds, g in FEATURE_TABLE.groupby("dataset", sort=False):
        lines += [f"## {ds}", "", "| Feature | Modality | Unit | Source | Transformation | Note |",
                  "|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            lines.append(f"| `{r.feature}` | {r.modality} | {r.unit} | {r.source} | {r.transformation} | {r.note} |")
        lines.append("")
    lines += ["## Excluded on purpose (leakage or timing)", "",
              "| Variable | Dataset | Why excluded from model inputs |", "|---|---|---|",
              "| STAI (per phase) | QoL_Stress | completed at the end of each phase: a self-report of the label |",
              "| PSS-14 | QoL_Stress | one value per participant (identical in both phases): carries no phase information |",
              "| Fitbit HR zone summary | QoL_Stress | whole-day summary, not phase-specific |",
              "| ECG device `heart_rate`, `result_classification` | QoL_Stress | used for QC only |",
              "| Absolute clock times | QoL_Stress | relax always precedes stress: time would encode the label |",
              "| sleep.csv columns | MMASH | they are (or directly compute) the targets |",
              "| Night-time actigraphy / RR | MMASH | recorded during the outcome night (concurrent, not predictive) |",
              "| Pittsburgh (PSQI) | MMASH | contains a self-reported habitual sleep-efficiency component |",
              "| PANAS 9+1, wake-up saliva | MMASH | measured after the outcome night |", ""]
    (PROJECT_ROOT / "docs" / "FEATURE_DICTIONARY.md").write_text("\n".join(lines))
    return FEATURE_TABLE
