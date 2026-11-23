"""Central configuration for HealthFusion.

Every constant used by more than one module lives here so that experiments are
reproducible and every "magic number" is documented in one place.

Values marked VERIFIED were checked against the data itself (see
results/reports/01_phase1_feasibility_report.md). Values marked DESIGN are
methodological choices; they are justified in docs/RESEARCH_DESIGN.md.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("HEALTHFUSION_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MMASH_RAW = RAW_DIR / "mmash" / "DataPaper"          # contains user_1 ... user_22
QOL_RAW = RAW_DIR / "qol_stress"                     # contains ecg.csv, eda.csv, ...
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
REPORTS_DIR = RESULTS_DIR / "reports"

for _d in (PROCESSED_DIR, SAMPLE_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
GLOBAL_SEED = 42
CV_REPEAT_SEEDS = [42, 43, 44, 45, 46]   # DESIGN: 5 repeats of participant-grouped 5-fold CV
OUTER_FOLDS = 5
INNER_FOLDS = 3
N_PERMUTATIONS_QOL = 200                 # label permutations (within participant)
N_PERMUTATIONS_MMASH = 1000

# ---------------------------------------------------------------------------
# QoL_Stress signal constants
# ---------------------------------------------------------------------------
# VERIFIED: every ECG record has ~7500 samples; detected HR at 250 Hz matches the
# device-reported HR within 5 bpm for 99.2 % of records (125/300/500 Hz fail).
QOL_ECG_FS = 250
QOL_ECG_EXPECTED_SAMPLES = 7500          # 30 s x 250 Hz
# VERIFIED: EDA rows arrive at ~1 row per second (163 rows over 0-162 s).
QOL_EDA_FS = 1.0
# DESIGN: Fitbit HR has no phase label. Phase windows are anchored on the STAI
# submission timestamps (same clock as Fitbit, verified). Equal-length windows
# avoid duration confounds.
QOL_HR_WINDOW_MIN = 8.0
QOL_HR_MIN_SAMPLES = 60                  # QC: minimum Fitbit HR samples per window
QOL_ECG_MAX_HR_DEVIATION = 10.0          # QC: |detected HR - device HR| (bpm)
QOL_ECG_MIN_BEATS = 15
QOL_ECG_MAX_ARTEFACT_PCT = 20.0          # QC: max % of RR intervals rejected as artefacts
QOL_EDA_MIN_VALID = 60                   # QC: minimum valid EDA samples per phase

# Two ECG recordings (reading_ids below) appear, byte-identical, under two
# participants (Q15 and Q48). Their timestamps fit Q15's own session timeline
# (STAI/Fitbit) but are shifted by exactly 4 h for Q48, whose Fitbit data also
# end before its relax phase. Evidence therefore assigns them to Q15; Q48's ECG
# is set to missing (its true ECG is not in the release).
QOL_DUPLICATED_ECG_READING_IDS = [
    "eb7e45f0-b707-11ef-8080-808080808080",
    "770c2cd0-b709-11ef-8080-808080808080",
]
QOL_ECG_EXCLUDE_USERNAMES = [
    "b8bfd374386b0793f2abb9d233a5385e336bc9ada86ebd540b471ade6d768f34",  # Q48
]

# ---------------------------------------------------------------------------
# MMASH constants
# ---------------------------------------------------------------------------
MMASH_N_USERS = 22
# Physiologically plausible inter-beat interval range (seconds) = 30-200 bpm.
RR_MIN_S, RR_MAX_S = 0.30, 2.00
# Successive-difference artefact rule: |RR_i - local median| > 20 % of local median.
RR_LOCAL_MEDIAN_WINDOW = 11
RR_MAX_REL_DEVIATION = 0.20
HRV_SEGMENT_S = 300           # 5-min segments (Task Force 1996 short-term HRV)
HRV_ULTRA_SHORT_S = 30        # 30-s segments, comparable with QoL_Stress ECG
# Valid beats must cover >= 60 % of a segment. VERIFIED: MMASH RR streams have a
# recurring ~8-s dropout that caps coverage near 0.80 even in clean recordings,
# so the conventional 80 % rule would reject almost every segment.
HRV_MIN_COVERAGE = 0.60
EVENING_WINDOW_H = 2.0        # pre-bed window for evening heart rate

# Activity diary codes, verified against the PhysioNet MMASH page.
MMASH_ACTIVITY_CODES = {
    1: "sleeping", 2: "laying down", 3: "sitting", 4: "light movement",
    5: "medium activity", 6: "heavy activity", 7: "eating",
    8: "small screen usage", 9: "large screen usage",
    10: "caffeinated drink", 11: "smoking", 12: "alcohol",
}
MMASH_MVPA_CODES = (5, 6)
MMASH_SCREEN_CODES = (8, 9)
MMASH_CAFFEINE_CODES = (10,)

# Questionnaire validity ranges (published instrument ranges, as listed on the
# PhysioNet MMASH page). Values outside the range are set to missing and reported,
# never silently dropped. BIS/BAS is deliberately NOT range-checked: 8 of 22
# participants exceed the standard Carver & White Reward maximum (20), which
# indicates a different scoring variant rather than individual entry errors.
MMASH_QUESTIONNAIRE_RANGES = {
    "MEQ": (16, 86), "STAI1": (20, 80), "STAI2": (20, 80), "Pittsburgh": (0, 21),
    "Daily_stress": (0, 406),
}

MMASH_PRIMARY_TARGET = "sleep_efficiency"
MMASH_SECONDARY_TARGETS = ["waso_min", "tst_min", "sleep_fragmentation_index", "psqi"]
