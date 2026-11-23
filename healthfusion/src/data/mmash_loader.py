"""Loading and low-level parsing for MMASH (22 participant folders).

Documented repairs (all logged in the returned issue list):
* users 8 and 9 code the second recording day as -29 in RR.csv/Actigraph.csv
  (a month-rollover artefact). Every timestamp in those rows follows day-1
  timestamps chronologically, so -29 is recoded to day 2.
* Absolute time = (day - 1) * 86400 + seconds since midnight, so sessions that
  cross midnight stay monotonic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import MMASH_N_USERS, MMASH_RAW

FILES = ["user_info", "questionnaire", "sleep", "saliva", "Activity", "Actigraph", "RR"]


def _hms_to_s(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float, index=series.index)
    parts = series.astype(str).str.strip().str.split(":", expand=True)
    parts = parts.apply(pd.to_numeric, errors="coerce")
    sec = parts[0] * 3600 + parts[1] * 60
    if parts.shape[1] > 2:
        sec = sec + parts[2].fillna(0)
    return sec.astype(float)


def _fix_day(day: pd.Series, issues: list, uid: int, fname: str) -> pd.Series:
    bad = ~day.isin([1, 2])
    if bad.any():
        vals = sorted(day[bad].unique().tolist())
        issues.append(f"user_{uid}/{fname}: day codes {vals} recoded to 2 ({int(bad.sum())} rows)")
        day = day.where(~bad, 2)
    return day


@dataclass
class MMASHParticipant:
    uid: int
    files: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)


def load_participant(uid: int, raw_dir=MMASH_RAW) -> MMASHParticipant:
    folder = raw_dir / f"user_{uid}"
    p = MMASHParticipant(uid)
    for f in FILES:
        path = folder / f"{f}.csv"
        if not path.exists():
            p.issues.append(f"user_{uid}: {f}.csv not provided")
            continue
        df = pd.read_csv(path)
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
        p.files[f] = df

    for f in ("RR", "Actigraph"):
        if f in p.files:
            df = p.files[f]
            df["day"] = _fix_day(df["day"], p.issues, uid, f)
            df["t_abs"] = (df["day"] - 1) * 86400 + _hms_to_s(df["time"])
    if "Activity" in p.files:
        df = p.files["Activity"]
        start = _hms_to_s(df["Start"])
        end = _hms_to_s(df["End"])
        df["t_start"] = (df["Day"] - 1) * 86400 + start
        # an end earlier than its start crosses midnight
        end_abs = (df["Day"] - 1) * 86400 + end
        end_abs = end_abs.where(end_abs >= df["t_start"], end_abs + 86400)
        df["t_end"] = end_abs
    if "sleep" in p.files:
        df = p.files["sleep"]
        for prefix in ("In Bed", "Out Bed", "Onset"):
            df[f"{prefix} t_abs"] = (df[f"{prefix} Date"] - 1) * 86400 + _hms_to_s(df[f"{prefix} Time"])
        df.loc[df["Out Bed t_abs"] < df["In Bed t_abs"], "Out Bed t_abs"] += 86400
    return p


def load_mmash(raw_dir=MMASH_RAW) -> dict[int, MMASHParticipant]:
    return {uid: load_participant(uid, raw_dir) for uid in range(1, MMASH_N_USERS + 1)}
