"""Signal-level processing for QoL_Stress (ECG, EDA, Fitbit HR).

These functions are pure (array in -> dict out) so that exactly the same code
is used by the offline pipeline and by the FastAPI raw-signal endpoint.
"""
from __future__ import annotations

import warnings

import neurokit2 as nk
import numpy as np

from src.config import QOL_ECG_FS, QOL_EDA_FS
from src.features.hrv import rr_validity_mask, time_domain_hrv

ECG_FEATURES = ["ecg_hr", "ecg_rmssd", "ecg_sdnn", "ecg_pnn50", "ecg_hr_range"]
EDA_FEATURES = ["eda_scl_mean", "eda_scl_std", "eda_scl_slope", "eda_scl_range", "eda_resp_rate"]
HR_FEATURES = ["fb_hr_mean", "fb_hr_std", "fb_hr_p05", "fb_hr_p95", "fb_hr_slope"]


# ---------------------------------------------------------------------------
# ECG (single-lead, 30 s, 250 Hz, Fitbit Charge 6 ECG app)
# ---------------------------------------------------------------------------
def process_ecg(samples: np.ndarray, fs: float = QOL_ECG_FS) -> dict:
    """Clean an ECG strip, detect R-peaks and return ultra-short HRV features + QC.

    Steps: NeuroKit2 'neurokit' cleaning (0.5 Hz high-pass Butterworth + 50 Hz
    powerline filter) -> R-peak detection -> Kubios-style artefact correction ->
    RR intervals -> the SAME beat-level artefact rule used for MMASH (0.3-2.0 s
    and within 20 % of the local median; differences only between adjacent valid
    beats) -> time-domain HRV. The extra rule is needed: without it 18 strips had
    physiologically implausible values (SDNN up to 390 ms in 30 s) from missed
    or spurious beats that NeuroKit2's correction left in place.
    30-s HRV is "ultra-short-term"; RMSSD is the most reliable index at this
    length, SDNN/pNN50 are reported with caution.
    """
    x = np.asarray(samples, dtype=float)
    out = {k: np.nan for k in ECG_FEATURES}
    out.update({"ecg_n_beats": 0, "ecg_quality": np.nan, "ecg_duration_s": x.size / fs, "ecg_error": None,
                "ecg_pct_artefact": np.nan})
    if x.size < fs * 10:
        out["ecg_error"] = "signal shorter than 10 s"
        return out
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean = nk.ecg_clean(x, sampling_rate=fs, method="neurokit")
            _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="neurokit", correct_artifacts=True)
            peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
            if peaks.size >= 4:
                q = nk.ecg_quality(clean, rpeaks=peaks, sampling_rate=fs, method="averageQRS")
                out["ecg_quality"] = float(np.nanmean(q))
    except Exception as exc:  # NeuroKit raises on flat/degenerate signals
        out["ecg_error"] = f"neurokit failure: {exc}"
        return out
    rr_ms = np.diff(peaks) / fs * 1000.0
    valid = rr_validity_mask(rr_ms / 1000.0) if rr_ms.size else np.zeros(0, dtype=bool)
    hrv = time_domain_hrv(rr_ms, valid)
    inst_hr = 60000.0 / rr_ms[valid] if valid.any() else np.array([np.nan])
    out.update({
        "ecg_hr": hrv["hr"], "ecg_rmssd": hrv["rmssd"], "ecg_sdnn": hrv["sdnn"], "ecg_pnn50": hrv["pnn50"],
        "ecg_hr_range": float(np.nanmax(inst_hr) - np.nanmin(inst_hr)) if valid.sum() >= 2 else np.nan,
        "ecg_n_beats": int(peaks.size),
        "ecg_pct_artefact": float(100.0 * (1 - valid.mean())) if valid.size else np.nan,
    })
    return out


# ---------------------------------------------------------------------------
# EDA (Fitbit EDA Scan app, ~1 row per second, ~160 s per phase)
# ---------------------------------------------------------------------------
def process_eda(scl: np.ndarray, activation: np.ndarray, valid: np.ndarray | None = None,
                fs: float = QOL_EDA_FS) -> dict:
    """Skin-conductance-level (SCL) statistics and EDA response rate.

    Time axis = row index / fs. `time_sec` in the release is rounded to ~5-s
    blocks and has offsets in 5 sessions, so row order (verified 1 row/s) is used.
    """
    scl = np.asarray(scl, dtype=float)
    act = np.asarray(activation, dtype=float)
    valid = np.ones_like(scl, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    t_min = np.arange(scl.size) / fs / 60.0
    s, a, tt = scl[valid], act[valid], t_min[valid]
    out = {k: np.nan for k in EDA_FEATURES}
    out.update({"eda_n_valid": int(valid.sum()), "eda_n_rows": int(scl.size)})
    if s.size < 10:
        return out
    duration_min = s.size / fs / 60.0
    out.update({
        "eda_scl_mean": float(s.mean()),
        "eda_scl_std": float(s.std(ddof=1)),
        "eda_scl_slope": float(np.polyfit(tt, s, 1)[0]),            # units per minute
        "eda_scl_range": float(np.percentile(s, 95) - np.percentile(s, 5)),
        "eda_resp_rate": float(np.nansum(a) / duration_min),          # responses per minute
    })
    return out


# ---------------------------------------------------------------------------
# Fitbit optical heart rate (irregular, median 2 s spacing)
# ---------------------------------------------------------------------------
def process_hr_window(t_s: np.ndarray, hr_bpm: np.ndarray, start_s: float | None = None,
                      end_s: float | None = None) -> dict:
    """Heart-rate statistics inside [start_s, end_s]. Not HRV: Fitbit HR is a
    smoothed optical estimate, so beat-to-beat variability cannot be derived."""
    t = np.asarray(t_s, dtype=float)
    v = np.asarray(hr_bpm, dtype=float)
    m = np.isfinite(t) & np.isfinite(v) & (v >= 30) & (v <= 220)
    if start_s is not None:
        m &= t >= start_s
    if end_s is not None:
        m &= t <= end_s
    t, v = t[m], v[m]
    out = {k: np.nan for k in HR_FEATURES}
    out["fb_n_samples"] = int(v.size)
    out["fb_span_min"] = float((t.max() - t.min()) / 60.0) if v.size else 0.0
    if v.size < 5:
        return out
    out.update({
        "fb_hr_mean": float(v.mean()), "fb_hr_std": float(v.std(ddof=1)),
        "fb_hr_p05": float(np.percentile(v, 5)), "fb_hr_p95": float(np.percentile(v, 95)),
        "fb_hr_slope": float(np.polyfit((t - t.min()) / 60.0, v, 1)[0]),   # bpm per minute
    })
    return out
