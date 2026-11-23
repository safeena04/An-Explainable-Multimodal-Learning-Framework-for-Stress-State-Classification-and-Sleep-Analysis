"""Phase 1: automated, reproducible inspection of both datasets.

Run:  python -m src.pipelines.run_phase1
Writes results/reports/phase1_inspection.json and phase1_inspection_tables.md.
Everything in the written feasibility report is traceable to these outputs.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from src.config import MMASH_QUESTIONNAIRE_RANGES, QOL_RAW, REPORTS_DIR
from src.data.mmash_loader import FILES as MMASH_FILES, load_mmash
from src.data.qol_loader import load_qol, parse_waveform


def _file_summary(df: pd.DataFrame, id_col: str | None = None) -> dict:
    out = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "exact_duplicate_rows": int(df.duplicated().sum()),
    }
    if id_col and id_col in df:
        out["n_participants"] = int(df[id_col].nunique())
    return out


def _time_pattern_counts(s: pd.Series) -> dict:
    pat = s.astype(str).str.strip().str.replace(r"\d", "9", regex=True)
    return {k: int(v) for k, v in pat.value_counts().items()}


# ---------------------------------------------------------------------------
# QoL_Stress
# ---------------------------------------------------------------------------
def inspect_qol() -> dict:
    d = load_qol()
    raw_cols = {k: [c for c in v.columns if c not in ("pid", "t_s", "row_in_session")] for k, v in d.items()}
    rep: dict = {"files": {k: _file_summary(v[raw_cols[k]], "username") for k, v in d.items()}}
    rep["n_participants"] = int(d["participants"]["username"].nunique())

    # --- phases and availability -------------------------------------------------
    rep["phase_counts"] = {
        k: {p: int(n) for p, n in d[k]["phase"].value_counts().items()} for k in ("ecg", "eda", "stai")
    }
    eda_sessions = d["eda"].groupby(["pid", "phase"]).size().unstack()
    rep["eda_sessions"] = {
        "participants_with_any_eda": int(eda_sessions.notna().any(axis=1).sum()),
        "both_phases": int(eda_sessions.notna().all(axis=1).sum()),
        "relax_only": int((eda_sessions["relax"].notna() & eda_sessions["stress"].isna()).sum()),
        "stress_only": int((eda_sessions["stress"].notna() & eda_sessions["relax"].isna()).sum()),
        "participants_without_eda": sorted(set(d["participants"]["pid"]) - set(eda_sessions.index)),
        "rows_per_session": eda_sessions.stack().describe().round(1).to_dict(),
    }

    # --- ECG waveform decoding ----------------------------------------------------
    n_samples, repaired = [], 0
    for w in d["ecg"]["waveform_samples"]:
        wp = parse_waveform(w)
        n_samples.append(len(wp.samples))
        repaired += int(wp.n_repaired > 0)
    n_samples = pd.Series(n_samples)
    dup = d["ecg"][d["ecg"]["reading_id"].duplicated(keep=False)]
    rep["ecg"] = {
        "records": int(len(d["ecg"])),
        "records_per_participant": {int(k): int(v) for k, v in d["ecg"].groupby("pid").size().value_counts().items()},
        "samples_per_record": {int(k): int(v) for k, v in n_samples.value_counts().items()},
        "records_with_repaired_token": repaired,
        "device": sorted(d["ecg"]["hardware_version"].unique().tolist()),
        "result_classification_by_phase": pd.crosstab(d["ecg"]["phase"], d["ecg"]["result_classification"]).to_dict(),
        "duplicated_reading_ids": {
            rid: {"pids": g["pid"].tolist(), "phases": g["phase"].tolist(),
                  "identical_waveforms": bool(g["waveform_samples"].nunique() == 1),
                  "reading_times": g["reading_time"].tolist()}
            for rid, g in dup.groupby("reading_id")
        },
        "sampling_rate_note": "Not stated in the files. 7500 samples per record; verified as 250 Hz "
                              "by R-peak heart-rate agreement with device HR (see preprocessing QC).",
    }

    # --- EDA sampling ------------------------------------------------------------
    eda = d["eda"]
    per_sess = eda.groupby(["pid", "phase"]).agg(n=("time_sec", "size"), tmin=("time_sec", "min"), tmax=("time_sec", "max"))
    normal = per_sess[per_sess["tmin"] >= 0]
    rep["eda"] = {
        "time_sec_resolution": "integer seconds, repeated in blocks (~5 rows share one value)",
        "rows_per_second_normal_sessions": float(round((normal["n"] / (normal["tmax"] - normal["tmin"] + 1)).median(), 3)),
        "sessions_with_negative_time_offset": [f"{a}/{b}" for a, b in per_sess[per_sess["tmin"] < 0].index],
        "valid_data_false_rows": int((~eda["valid_data"]).sum()),
        "activation_values": {str(k): int(v) for k, v in eda["activation"].value_counts().items()},
        "scl_avg_range": [float(eda["scl_avg"].min()), float(eda["scl_avg"].max())],
        "unit_note": "Units of scl_avg are not documented in the release.",
    }

    # --- Fitbit HR ---------------------------------------------------------------
    hs = d["hr_second"].sort_values(["pid", "t_s"])
    gaps = hs.groupby("pid")["t_s"].diff().dropna()
    rep["fitbit_hr"] = {
        "per_second_rows": int(len(hs)),
        "per_second_interval_s": {str(int(k)): int(v) for k, v in gaps.value_counts().head(8).items()},
        "per_second_median_interval_s": float(gaps.median()),
        "per_second_rows_per_participant": hs.groupby("pid").size().describe().round(1).to_dict(),
        "per_minute_rows": int(len(d["hr_minute"])),
        "phase_column_present": False,
        "summary_note": "fitbit_heart_rate_summary.csv is a whole-day zone summary (Out of Range minutes ~1440); "
                        "it is not phase-specific and cannot be used for phase classification.",
    }

    # --- timestamps & session timeline -------------------------------------------
    rep["timestamp_formats"] = {
        "stai.time": _time_pattern_counts(d["stai"]["time"]),
        "pss_14.time": _time_pattern_counts(d["pss"]["time"]),
        "ecg.reading_time": _time_pattern_counts(d["ecg"]["reading_time"]),
        "fitbit_per_second.time": _time_pattern_counts(d["hr_second"]["time"]),
        "eda.time_sec": "integer seconds relative to the start of each EDA scan",
    }
    st = d["stai"].pivot(index="pid", columns="phase", values="t_s")
    ecg_t = d["ecg"].pivot(index="pid", columns="phase", values="t_s")
    tl = pd.DataFrame({
        "pss": d["pss"].set_index("pid")["t_s"],
        "stai_relax": st["relax"], "stai_stress": st["stress"],
        "ecg_relax": ecg_t["relax"], "ecg_stress": ecg_t["stress"],
        "fitbit_first": hs.groupby("pid")["t_s"].min(), "fitbit_last": hs.groupby("pid")["t_s"].max(),
    })
    rel = tl.sub(tl["stai_relax"], axis=0) / 60.0
    rep["timeline_minutes_relative_to_stai_relax"] = rel.describe().round(1).to_dict()
    rep["stai_stress_always_after_relax"] = bool((rel["stai_stress"] > 0).all())
    in_relax = (rel["ecg_relax"] <= 0) & (rel["ecg_relax"] >= -15)
    in_stress = (rel["ecg_stress"] > 0) & (rel["ecg_stress"] <= rel["stai_stress"])
    rep["ecg_time_consistency"] = {
        "ecg_relax_within_15min_before_stai_relax": int(in_relax.sum()),
        "ecg_stress_between_stai_relax_and_stai_stress": int(in_stress.sum()),
        "ecg_relax_outliers_pid": sorted(rel.index[~in_relax].tolist()),
        "ecg_stress_outliers_pid": sorted(rel.index[~in_stress].tolist()),
    }

    # --- questionnaires ------------------------------------------------------------
    w = d["stai"].pivot(index="pid", columns="phase", values="score")
    rep["stai"] = {
        "relax": w["relax"].describe().round(2).to_dict(), "stress": w["stress"].describe().round(2).to_dict(),
        "stress_gt_relax": int((w["stress"] > w["relax"]).sum()),
        "equal": int((w["stress"] == w["relax"]).sum()),
        "stress_lt_relax": int((w["stress"] < w["relax"]).sum()),
    }
    rep["pss14"] = d["pss"]["score"].describe().round(2).to_dict()
    p = d["participants"]
    rep["demographics"] = {
        "sex": {k: int(v) for k, v in p["sex"].value_counts().items()},
        "age": p["age"].describe().round(1).to_dict(),
        "education_level": {k: int(v) for k, v in p["education_level"].value_counts().items()},
    }
    tl.round(1).to_csv(REPORTS_DIR / "phase1_qol_timeline_seconds.csv")
    return rep


# ---------------------------------------------------------------------------
# MMASH
# ---------------------------------------------------------------------------
def inspect_mmash() -> dict:
    parts = load_mmash()
    rep: dict = {"n_participant_folders": len(parts)}
    avail = pd.DataFrame({uid: {f: (len(p.files[f]) if f in p.files else 0) for f in MMASH_FILES}
                          for uid, p in parts.items()}).T
    rep["rows_per_file_per_user"] = avail.to_dict(orient="index")
    rep["files_missing"] = {f: [int(u) for u in avail.index[avail[f] == 0]] for f in MMASH_FILES if (avail[f] == 0).any()}
    rep["load_issues"] = [i for p in parts.values() for i in p.issues]

    # schemas from user_1 + missing values across all users
    schema, missing, dups = {}, Counter(), Counter()
    for p in parts.values():
        for f, df in p.files.items():
            raw = df[[c for c in df.columns if c not in ("t_abs", "t_start", "t_end") and not c.endswith("t_abs")]]
            schema.setdefault(f, {c: str(t) for c, t in raw.dtypes.items()})
            for c, v in raw.isna().sum().items():
                if v:
                    missing[f"{f}.{c}"] += int(v)
            dups[f] += int(raw.duplicated().sum())
    rep["schemas"] = schema
    rep["missing_values_total"] = dict(missing)
    rep["exact_duplicate_rows_total"] = dict(dups)

    # sampling
    ag_dt, rr_stats = Counter(), []
    for uid, p in parts.items():
        ag = p.files["Actigraph"].sort_values("t_abs")
        ag_dt.update(ag["t_abs"].diff().dropna().astype(int).tolist())
        rr = p.files["RR"]["ibi_s"]
        rr_stats.append({"user": uid, "n_beats": int(len(rr)), "ibi_min_s": float(rr.min()), "ibi_max_s": float(rr.max()),
                         "pct_outside_0.3_2.0s": float(round(100 * ((rr < 0.3) | (rr > 2.0)).mean(), 3)),
                         "recording_hours": float(round((p.files["RR"]["t_abs"].max() - p.files["RR"]["t_abs"].min()) / 3600, 2))})
    rep["actigraph_interval_s_top"] = {str(k): int(v) for k, v in ag_dt.most_common(6)}
    rep["rr_per_user"] = rr_stats

    # sleep episodes
    sleep_rows = []
    for uid, p in parts.items():
        s = p.files.get("sleep")
        sleep_rows.append({"user": uid, "episodes": 0 if s is None else len(s),
                           "efficiency": None if s is None or s.empty else s["Efficiency"].round(2).tolist()})
    rep["sleep_episodes"] = sleep_rows

    # questionnaire validity
    q = pd.concat([p.files["questionnaire"].assign(user=uid) for uid, p in parts.items()], ignore_index=True)
    invalid = []
    for c, (lo, hi) in MMASH_QUESTIONNAIRE_RANGES.items():
        bad = q[(q[c] < lo) | (q[c] > hi)]
        invalid += [{"user": int(r.user), "item": c, "value": float(r[c]), "valid_range": [lo, hi]} for _, r in bad.iterrows()]
    rep["questionnaire_out_of_range"] = invalid
    rep["questionnaire_missing"] = {c: [int(u) for u in q.loc[q[c].isna(), "user"]] for c in q.columns if q[c].isna().any()}
    rep["questionnaire_summary"] = q.drop(columns="user").describe().round(2).to_dict()

    ui = pd.concat([p.files["user_info"].assign(user=uid) for uid, p in parts.items()], ignore_index=True)
    rep["user_info"] = {"gender": {k: int(v) for k, v in ui["Gender"].value_counts().items()},
                        "age_zero_users": ui.loc[ui["Age"] <= 0, "user"].astype(int).tolist(),
                        "age": ui.loc[ui["Age"] > 0, "Age"].describe().round(1).to_dict()}

    act = pd.concat([p.files["Activity"].assign(user=uid) for uid, p in parts.items()], ignore_index=True)
    rep["activity_codes"] = {str(int(k)): int(v) for k, v in act["Activity"].value_counts().sort_index().items()}
    rep["activity_undocumented_code0_rows"] = int((act["Activity"] == 0).sum())
    rep["activity_missing_end"] = int(act["End"].isna().sum())

    sal = pd.concat([p.files["saliva"].assign(user=uid) for uid, p in parts.items() if "saliva" in p.files], ignore_index=True)
    rep["saliva"] = {"users": int(sal["user"].nunique()), "samples": {k: int(v) for k, v in sal["SAMPLES"].value_counts().items()},
                     "melatonin_zero_values": int((sal["Melatonin NORM"] == 0).sum())}
    return rep


def run_inspection() -> dict:
    report = {"qol_stress": inspect_qol(), "mmash": inspect_mmash()}
    with open(REPORTS_DIR / "phase1_inspection.json", "w") as fh:
        json.dump(report, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return report
