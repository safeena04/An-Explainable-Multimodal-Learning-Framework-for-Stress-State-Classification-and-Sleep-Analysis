"""Signal-processing functions on synthetic signals (code tests only, not research data)."""
import neurokit2 as nk
import numpy as np
import pytest

from src.preprocessing.qol_signals import process_ecg, process_eda, process_hr_window


def test_ecg_heart_rate_recovered():
    ecg = nk.ecg_simulate(duration=30, sampling_rate=250, heart_rate=72, noise=0.01, random_state=1)
    out = process_ecg(ecg, 250)
    assert out["ecg_error"] is None
    assert out["ecg_hr"] == pytest.approx(72, abs=3)
    assert out["ecg_n_beats"] >= 30


def test_ecg_too_short():
    assert process_ecg(np.zeros(100), 250)["ecg_error"] is not None


def test_eda_slope_and_rate():
    scl = 40 + 0.5 * np.arange(120) / 60.0      # +0.5 units per minute
    act = np.zeros(120)
    act[::20] = 1                                # 6 responses in 2 minutes
    out = process_eda(scl, act)
    assert out["eda_scl_slope"] == pytest.approx(0.5, rel=1e-6)
    assert out["eda_resp_rate"] == pytest.approx(3.0)


def test_eda_invalid_rows_excluded():
    scl = np.r_[np.full(50, 1000.0), np.full(100, 40.0)]
    valid = np.r_[np.zeros(50, bool), np.ones(100, bool)]
    out = process_eda(scl, np.zeros(150), valid)
    assert out["eda_scl_mean"] == pytest.approx(40.0)
    assert out["eda_n_valid"] == 100


def test_hr_window_bounds_and_plausibility():
    t = np.arange(0, 600, 2.0)
    v = np.full(t.size, 70.0)
    v[5] = 300  # implausible, dropped
    out = process_hr_window(t, v, start_s=100, end_s=400)
    assert out["fb_hr_mean"] == pytest.approx(70.0)
    assert out["fb_n_samples"] == 151


def test_ecg_reports_artefact_share_and_clean_signal_has_few():
    ecg = nk.ecg_simulate(duration=30, sampling_rate=250, heart_rate=65, noise=0.01, random_state=3)
    out = process_ecg(ecg, 250)
    assert "ecg_pct_artefact" in out and out["ecg_pct_artefact"] <= 10
    assert 0 < out["ecg_rmssd"] < 150   # plausible ultra-short HRV on a clean strip
