"""Time-domain heart-rate-variability (HRV) features from inter-beat intervals.

Definitions (Task Force of the ESC/NASPE, Circulation 1996):
* mean_rr  : mean normal-to-normal (NN) interval, ms
* hr       : 60000 / mean_rr, beats per minute
* sdnn     : standard deviation of NN intervals, ms (overall variability)
* rmssd    : root mean square of successive NN differences, ms (beat-to-beat,
             vagally mediated variability)
* pnn50    : % of successive NN differences > 50 ms

Artefact handling: an interval is "valid" if it is inside the physiological
range AND within +/-20 % of the local median of its neighbours. Successive
differences are only computed between two *adjacent valid* beats, so an
artefact never creates a fake large difference. "Adjacent" also requires no
recording gap between the two beats: MMASH RR files contain a recurring ~8-s
dropout (verified: 2,337 eight-second gaps for user 1), and a difference taken
across such a gap would compare two beats that are not consecutive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RR_LOCAL_MEDIAN_WINDOW, RR_MAX_REL_DEVIATION, RR_MAX_S, RR_MIN_S


def rr_validity_mask(rr_s: np.ndarray) -> np.ndarray:
    """Boolean mask of valid inter-beat intervals (seconds)."""
    rr = np.asarray(rr_s, dtype=float)
    in_range = (rr >= RR_MIN_S) & (rr <= RR_MAX_S)
    local_med = (
        pd.Series(np.where(in_range, rr, np.nan))
        .rolling(RR_LOCAL_MEDIAN_WINDOW, center=True, min_periods=3)
        .median()
        .to_numpy()
    )
    rel_dev = np.abs(rr - local_med) / local_med
    return in_range & (np.nan_to_num(rel_dev, nan=1.0) <= RR_MAX_REL_DEVIATION)


def time_domain_hrv(rr_ms: np.ndarray, valid: np.ndarray | None = None,
                    adjacent: np.ndarray | None = None) -> dict:
    """HRV from NN intervals in milliseconds.

    valid    : boolean mask of usable beats (default: finite values)
    adjacent : boolean mask of length n-1; adjacent[i] is True when beat i+1
               directly follows beat i with no recording gap (default: all True)
    """
    rr = np.asarray(rr_ms, dtype=float)
    if valid is None:
        valid = np.isfinite(rr)
    if adjacent is None:
        adjacent = np.ones(max(rr.size - 1, 0), dtype=bool)
    nn = rr[valid]
    out = {"n_beats": int(valid.sum()), "mean_rr": np.nan, "hr": np.nan,
           "sdnn": np.nan, "rmssd": np.nan, "pnn50": np.nan}
    if nn.size < 3:
        return out
    pair = valid[1:] & valid[:-1] & adjacent
    diffs = np.diff(rr)[pair]
    out["mean_rr"] = float(nn.mean())
    out["hr"] = float(60000.0 / nn.mean())
    out["sdnn"] = float(nn.std(ddof=1))
    if diffs.size >= 2:
        out["rmssd"] = float(np.sqrt(np.mean(diffs ** 2)))
        out["pnn50"] = float(100.0 * np.mean(np.abs(diffs) > 50.0))
    return out


def adjacency_from_timestamps(t_s: np.ndarray, rr_s: np.ndarray, tolerance_s: float = 1.5) -> np.ndarray:
    """True where consecutive beats are truly consecutive.

    Timestamps have 1-s resolution, so beat i+1 should be stamped within
    rr[i+1] + tolerance seconds of beat i; a larger jump means beats were lost.
    """
    t = np.asarray(t_s, dtype=float)
    rr = np.asarray(rr_s, dtype=float)
    return np.diff(t) <= rr[1:] + tolerance_s


def segment_hrv(t_s: np.ndarray, rr_s: np.ndarray, segment_s: float, min_coverage: float) -> pd.DataFrame:
    """Split a long RR recording into fixed segments and compute HRV per segment.

    A segment is kept only if its valid beats cover >= min_coverage of its length,
    so gaps and artefact bursts do not produce unreliable HRV values.
    """
    t = np.asarray(t_s, dtype=float)
    rr = np.asarray(rr_s, dtype=float)
    valid = rr_validity_mask(rr)
    adj = adjacency_from_timestamps(t, rr)
    if t.size == 0:
        return pd.DataFrame()
    seg_id = np.floor((t - t.min()) / segment_s).astype(int)
    rows = []
    for sid in np.unique(seg_id):
        m = seg_id == sid
        coverage = rr[m & valid].sum() / segment_s
        if coverage < min_coverage:
            continue
        idx = np.flatnonzero(m)
        adj_seg = adj[idx[:-1]] if idx.size > 1 else np.array([], dtype=bool)
        feats = time_domain_hrv(rr[m] * 1000.0, valid[m], adj_seg)
        feats.update({"segment_start_s": float(t.min() + sid * segment_s), "coverage": float(coverage)})
        rows.append(feats)
    return pd.DataFrame(rows)
