"""Loading and low-level parsing for the QoL_Stress dataset.

This module only *reads* and *parses*; it never changes values except for two
documented repairs that every parse issue log reports:

1. Timestamps are time-of-day only (no date). One ECG timestamp is written as
   'Tue Dec 17 13:01:42'; the trailing HH:MM:SS is used.
2. 18 ECG waveforms contain one malformed token, a lone '-' followed by digits
   (e.g. '... -829 - 39 -863 ...'). The pair is merged into ONE missing sample
   (so the time axis keeps its length) and that single sample is linearly
   interpolated. The true value cannot be recovered, so it is not guessed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import QOL_RAW

_TIME_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})\s*$")


def time_of_day_seconds(value) -> float:
    """Convert 'HH:MM:SS', 'H:MM:SS' or 'Tue Dec 17 13:01:42' to seconds since midnight."""
    m = _TIME_RE.search(str(value).strip())
    if not m:
        return float("nan")
    h, mi, s = (int(x) for x in m.groups())
    return float(h * 3600 + mi * 60 + s)


@dataclass
class WaveformParse:
    samples: np.ndarray
    n_tokens: int
    n_repaired: int
    issues: list = field(default_factory=list)


def parse_waveform(text: str) -> WaveformParse:
    """Decode a space-separated ECG waveform string into a float array."""
    tokens = str(text).split()
    values: list[float] = []
    issues: list[str] = []
    k = 0
    while k < len(tokens):
        tok = tokens[k]
        if re.fullmatch(r"-?\d+", tok):
            values.append(float(tok))
            k += 1
        elif tok == "-" and k + 1 < len(tokens) and re.fullmatch(r"\d+", tokens[k + 1]):
            issues.append(f"split token '- {tokens[k + 1]}' at sample {len(values)}")
            values.append(np.nan)
            k += 2
        else:
            issues.append(f"unparseable token '{tok}' at sample {len(values)}")
            values.append(np.nan)
            k += 1
    arr = np.asarray(values, dtype=float)
    n_bad = int(np.isnan(arr).sum())
    if n_bad:
        arr = pd.Series(arr).interpolate(limit_direction="both").to_numpy()
    return WaveformParse(arr, len(tokens), n_bad, issues)


def short_ids(usernames: pd.Series) -> dict[str, str]:
    """Deterministic readable IDs (Q01..Q66) ordered by the hashed username."""
    return {u: f"Q{i + 1:02d}" for i, u in enumerate(sorted(usernames.unique()))}


def load_qol(raw_dir=QOL_RAW) -> dict[str, pd.DataFrame]:
    """Read every QoL_Stress CSV and add parsed time columns + short participant IDs."""
    files = {
        "participants": "participants.csv",
        "ecg": "ecg.csv",
        "eda": "eda.csv",
        "hr_second": "fitbit_heart_rate_per_second.csv",
        "hr_minute": "fitbit_heart_rate_per_minute.csv",
        "hr_summary": "fitbit_heart_rate_summary.csv",
        "pss": "pss_14.csv",
        "stai": "stai.csv",
    }
    d = {k: pd.read_csv(raw_dir / f) for k, f in files.items()}
    ids = short_ids(d["participants"]["username"])
    for k, df in d.items():
        df["pid"] = df["username"].map(ids)
    d["ecg"]["t_s"] = d["ecg"]["reading_time"].map(time_of_day_seconds)
    for k in ("hr_second", "hr_minute", "pss", "stai"):
        d[k]["t_s"] = d[k]["time"].map(time_of_day_seconds)
    d["eda"]["row_in_session"] = d["eda"].groupby(["username", "phase"]).cumcount()
    return d


def questionnaire_item_scores(df: pd.DataFrame, item_cols: list[str]) -> pd.DataFrame:
    """Turn answers like '2. Quite a bit' into the integer 2."""
    out = df[item_cols].apply(lambda c: c.astype(str).str.extract(r"^\s*(\d+)")[0].astype(float))
    return out
