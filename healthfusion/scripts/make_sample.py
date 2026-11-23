"""Create small demo payloads in data/sample/ from ONE real QoL_Stress participant.

The payloads are used by the API tests and the README examples. They are real
recordings (CC BY 4.0, Calvo et al., PhysioNet), not synthetic data. Note that
this participant is part of the training data, so a prediction on it is a demo
of the API, not evidence of performance.

Run:  python scripts/make_sample.py [--pid Q03]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import PROCESSED_DIR, QOL_HR_WINDOW_MIN, SAMPLE_DIR  # noqa: E402
from src.data.qol_loader import load_qol, parse_waveform  # noqa: E402


def main(pid: str) -> None:
    d = load_qol()
    feats = pd.read_csv(PROCESSED_DIR / "qol_phase_features.csv")
    stai = d["stai"].pivot(index="pid", columns="phase", values="t_s")
    payloads, feature_rows = {}, {}
    for phase in ("relax", "stress"):
        ecg = d["ecg"][(d["ecg"].pid == pid) & (d["ecg"].phase == phase)].iloc[0]
        eda = d["eda"][(d["eda"].pid == pid) & (d["eda"].phase == phase)].sort_values("row_in_session")
        hs = d["hr_second"][d["hr_second"].pid == pid]
        end = stai.loc[pid, phase]
        w = hs[(hs.t_s >= end - QOL_HR_WINDOW_MIN * 60) & (hs.t_s <= end)]
        payloads[phase] = {
            "ecg": {"samples": parse_waveform(ecg.waveform_samples).samples.tolist(), "sampling_rate": 250},
            "eda": {"scl": eda.scl_avg.tolist(), "activation": eda.activation.tolist(),
                    "valid": eda.valid_data.astype(bool).tolist(), "sampling_rate": 1.0},
            "heart_rate": {"bpm": w.value.astype(float).tolist(), "timestamps_s": (w.t_s - w.t_s.min()).tolist()},
            "explain": True,
        }
        row = feats[(feats.pid == pid) & (feats.phase == phase)].iloc[0]
        cols = [c for c in feats.columns if c.startswith(("ecg_", "eda_", "fb_")) and c in (
            "ecg_hr ecg_rmssd ecg_sdnn ecg_pnn50 ecg_hr_range eda_scl_mean eda_scl_std eda_scl_slope eda_scl_range "
            "eda_resp_rate fb_hr_mean fb_hr_std fb_hr_p05 fb_hr_p95 fb_hr_slope").split()]
        feature_rows[phase] = {"features": {c: float(row[c]) for c in cols}, "explain": True}
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for phase in payloads:
        (SAMPLE_DIR / f"stress_raw_{phase}.json").write_text(json.dumps(payloads[phase]))
        (SAMPLE_DIR / f"stress_features_{phase}.json").write_text(json.dumps(feature_rows[phase], indent=2))
    (SAMPLE_DIR / "README.md").write_text(
        f"# Sample payloads\n\nReal recordings of QoL_Stress participant {pid} (relax and stress phases), "
        "formatted as API request bodies.\n\nSource: Calvo, Cejudo & Martin, *QoL_Stress* (PhysioNet, v1.0.0), "
        "CC BY 4.0.\n\nThis participant is in the training set: predictions on these files demonstrate the API, "
        "they are not evidence of model performance.\n")
    print("written to", SAMPLE_DIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default="Q03")
    main(ap.parse_args().pid)
