"""MMASH preprocessing: one row per participant (participant-level design).

Design decisions (justified in docs/RESEARCH_DESIGN.md):
* PROSPECTIVE window: all predictor features come from the pre-sleep period
  (recording start -> first 'In Bed' time). Nothing recorded during or after the
  night enters the predictive feature set, so actigraphy-derived sleep metrics
  cannot leak into the inputs.
* Sleep targets come from sleep.csv (actigraphy-scored). User 1 has two in-bed
  episodes; they are aggregated (TST and TIB summed, SE = TST / TIB).
* Excluded from predictors: Pittsburgh (PSQI contains a self-reported habitual
  sleep-efficiency component), PANAS 9+1 (completed the morning after sleep),
  'wake up' saliva sample (after sleep).
* Undocumented diary code 0 rows and rows without an end time are excluded from
  diary features and counted in the QC log.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (EVENING_WINDOW_H, HRV_MIN_COVERAGE, HRV_SEGMENT_S, HRV_ULTRA_SHORT_S,
                        MMASH_CAFFEINE_CODES, MMASH_MVPA_CODES, MMASH_QUESTIONNAIRE_RANGES,
                        MMASH_SCREEN_CODES, PROCESSED_DIR)
from src.data.mmash_loader import load_mmash
from src.features.hrv import rr_validity_mask, segment_hrv

PHYS_FEATURES = ["phys_hr_mean", "phys_rmssd", "phys_sdnn", "phys_pnn50", "phys_hr_evening"]
BEHAV_FEATURES = ["behav_vm_mean", "behav_steps_per_hour", "behav_frac_standing", "behav_frac_lying",
                  "behav_mvpa_min", "behav_screen_min", "behav_caffeine_n"]
PSYCH_FEATURES = ["psych_meq", "psych_stai_state", "psych_stai_trait", "psych_daily_stress",
                  "psych_bis", "psych_panas_pos_day", "psych_panas_neg_day"]
MMASH_MODALITIES = {"Physiological": PHYS_FEATURES, "Behavioural": BEHAV_FEATURES, "Psychological": PSYCH_FEATURES}
TARGETS = ["sleep_efficiency", "tst_min", "waso_min", "sleep_fragmentation_index", "psqi"]


def _sleep_targets(s: pd.DataFrame) -> dict:
    tib = s["Total Minutes in Bed"].sum()
    tst = s["Total Sleep Time (TST)"].sum()
    w = s["Total Minutes in Bed"] / tib
    return {
        "n_sleep_episodes": int(len(s)),
        "in_bed_t_abs": float(s["In Bed t_abs"].min()),
        "onset_t_abs": float(s["Onset t_abs"].min()),
        "out_bed_t_abs": float(s["Out Bed t_abs"].max()),
        "tib_min": float(tib), "tst_min": float(tst),
        "sleep_efficiency": float(100.0 * tst / tib),
        "waso_min": float(s["Wake After Sleep Onset (WASO)"].sum()),
        "sleep_fragmentation_index": float((s["Sleep Fragmentation Index"] * w).sum()),
        "n_awakenings": float(s["Number of Awakenings"].sum()),
        "reported_efficiency_first_episode": float(s["Efficiency"].iloc[0]),
    }


def _physiological(rr: pd.DataFrame, t_end: float, qc: dict) -> tuple[dict, pd.DataFrame]:
    pre = rr[rr["t_abs"] < t_end]
    valid = rr_validity_mask(pre["ibi_s"].to_numpy())
    qc["rr_beats_pre_sleep"] = int(len(pre))
    qc["rr_pct_artefact_pre_sleep"] = float(round(100 * (1 - valid.mean()), 2)) if len(pre) else np.nan
    seg5 = segment_hrv(pre["t_abs"].to_numpy(), pre["ibi_s"].to_numpy(), HRV_SEGMENT_S, HRV_MIN_COVERAGE)
    qc["hrv_5min_segments_kept"] = int(len(seg5))
    ev = pre[pre["t_abs"] >= t_end - EVENING_WINDOW_H * 3600]
    ev_valid = rr_validity_mask(ev["ibi_s"].to_numpy())
    feats = {
        "phys_hr_mean": float(60.0 / pre["ibi_s"].to_numpy()[valid].mean()) if valid.any() else np.nan,
        "phys_rmssd": float(seg5["rmssd"].median()) if len(seg5) else np.nan,
        "phys_sdnn": float(seg5["sdnn"].median()) if len(seg5) else np.nan,
        "phys_pnn50": float(seg5["pnn50"].median()) if len(seg5) else np.nan,
        "phys_hr_evening": float(60.0 / ev["ibi_s"].to_numpy()[ev_valid].mean()) if ev_valid.sum() > 100 else np.nan,
    }
    return feats, seg5


def _behavioural(ag: pd.DataFrame, act: pd.DataFrame, t_start: float, t_end: float, qc: dict) -> dict:
    pre = ag[ag["t_abs"] < t_end]
    worn = pre[pre["Inclinometer Off"] == 0]
    worn_h = len(worn) / 3600.0
    qc["actigraph_worn_hours_pre_sleep"] = round(worn_h, 2)
    a = act.copy()
    qc["diary_code0_rows_excluded"] = int((a["Activity"] == 0).sum())
    qc["diary_rows_without_end_excluded"] = int(a["t_end"].isna().sum())
    a = a[(a["Activity"] != 0) & a["t_end"].notna()]
    clip_start = a["t_start"].clip(lower=t_start, upper=t_end)
    clip_end = a["t_end"].clip(lower=t_start, upper=t_end)
    a = a.assign(minutes=(clip_end - clip_start).clip(lower=0) / 60.0)
    return {
        "behav_vm_mean": float(worn["Vector Magnitude"].mean()) if len(worn) else np.nan,
        "behav_steps_per_hour": float(worn["Steps"].sum() / worn_h) if worn_h > 0 else np.nan,
        "behav_frac_standing": float(worn["Inclinometer Standing"].mean()) if len(worn) else np.nan,
        "behav_frac_lying": float(worn["Inclinometer Lying"].mean()) if len(worn) else np.nan,
        "behav_mvpa_min": float(a.loc[a["Activity"].isin(MMASH_MVPA_CODES), "minutes"].sum()),
        "behav_screen_min": float(a.loc[a["Activity"].isin(MMASH_SCREEN_CODES), "minutes"].sum()),
        "behav_caffeine_n": int((a["Activity"].isin(MMASH_CAFFEINE_CODES) & (a["t_start"] < t_end)).sum()),
    }


def _psychological(q: pd.Series, qc: dict) -> dict:
    q = q.copy()
    invalid = []
    for c, (lo, hi) in MMASH_QUESTIONNAIRE_RANGES.items():
        if pd.notna(q[c]) and not (lo <= q[c] <= hi):
            invalid.append(f"{c}={q[c]}")
            q[c] = np.nan
    qc["questionnaire_values_set_missing"] = ";".join(invalid) or "none"
    pos = [q[f"panas_pos_{h}"] for h in (10, 14, 18, 22)]
    neg = [q[f"panas_neg_{h}"] for h in (10, 14, 18, 22)]
    return {
        "psych_meq": q["MEQ"], "psych_stai_state": q["STAI1"], "psych_stai_trait": q["STAI2"],
        "psych_daily_stress": q["Daily_stress"], "psych_bis": q["BISBAS_bis"],
        "psych_panas_pos_day": float(np.nanmean(pos)), "psych_panas_neg_day": float(np.nanmean(neg)),
        # exploratory only (never predictors)
        "psqi": q["Pittsburgh"], "panas_pos_next_morning": q["panas_pos_9+1"],
        "panas_neg_next_morning": q["panas_neg_9+1"],
    }


def build_mmash(save: bool = True) -> dict[str, pd.DataFrame]:
    parts = load_mmash()
    rows, qc_rows, seg_rows = [], [], []
    for uid, p in parts.items():
        qc = {"user": uid, "load_issues": " | ".join(p.issues) or "none"}
        sleep = p.files.get("sleep")
        if sleep is None or sleep.empty:
            qc.update({"included": False, "reason": "no sleep episode recorded: no target and no pre-sleep window"})
            qc_rows.append(qc)
            continue
        tgt = _sleep_targets(sleep)
        t_end = tgt["in_bed_t_abs"]
        rr, ag, act = p.files["RR"], p.files["Actigraph"], p.files["Activity"]
        t_start = float(min(rr["t_abs"].min(), ag["t_abs"].min()))
        qc["pre_sleep_hours"] = round((t_end - t_start) / 3600, 2)

        phys, seg5 = _physiological(rr, t_end, qc)
        behav = _behavioural(ag, act, t_start, t_end, qc)
        psych = _psychological(p.files["questionnaire"].iloc[0], qc)
        info = p.files["user_info"].iloc[0]
        demo = {"age": float(info["Age"]) if info["Age"] > 0 else np.nan,
                "bmi": float(info["Weight"] / (info["Height"] / 100) ** 2)}
        sal = p.files.get("saliva")
        saliva = {"cortisol_before_sleep": np.nan, "melatonin_before_sleep": np.nan}
        if sal is not None:
            b = sal[sal["SAMPLES"].str.strip() == "before sleep"]
            if len(b):
                saliva = {"cortisol_before_sleep": float(b["Cortisol NORM"].iloc[0]),
                          "melatonin_before_sleep": float(b["Melatonin NORM"].iloc[0])}
        rows.append({"user": uid, **phys, **behav, **psych, **demo, **saliva, **tgt})

        # segment-level HRV for the cross-dataset analysis (30-s, comparable with QoL ECG)
        for period, lo, hi in (("day_pre_sleep", t_start, t_end),
                               ("night_sleep", tgt["onset_t_abs"], tgt["out_bed_t_abs"])):
            m = (rr["t_abs"] >= lo) & (rr["t_abs"] < hi)
            for seg_len, name in ((HRV_ULTRA_SHORT_S, "30s"), (HRV_SEGMENT_S, "5min")):
                s = segment_hrv(rr.loc[m, "t_abs"].to_numpy(), rr.loc[m, "ibi_s"].to_numpy(), seg_len, HRV_MIN_COVERAGE)
                if len(s):
                    seg_rows.append(s.assign(user=uid, period=period, segment=name))
        qc.update({"included": True, "reason": "passed"})
        qc_rows.append(qc)

    feat = pd.DataFrame(rows)
    qcdf = pd.DataFrame(qc_rows)
    segs = pd.concat(seg_rows, ignore_index=True)
    out = {"features": feat, "qc": qcdf, "hrv_segments": segs}
    if save:
        feat.to_csv(PROCESSED_DIR / "mmash_participant_features.csv", index=False)
        qcdf.to_csv(PROCESSED_DIR / "mmash_qc_log.csv", index=False)
        segs.to_csv(PROCESSED_DIR / "mmash_hrv_segments.csv", index=False)
    return out
