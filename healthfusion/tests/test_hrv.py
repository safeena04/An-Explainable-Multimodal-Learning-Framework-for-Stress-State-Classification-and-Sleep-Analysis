"""HRV maths on hand-computable inputs (synthetic numbers used only to test code)."""
import numpy as np
import pytest

from src.features.hrv import adjacency_from_timestamps, rr_validity_mask, segment_hrv, time_domain_hrv


def test_constant_rhythm():
    out = time_domain_hrv(np.full(50, 1000.0))
    assert out["hr"] == pytest.approx(60.0)
    assert out["rmssd"] == pytest.approx(0.0)
    assert out["sdnn"] == pytest.approx(0.0)
    assert out["pnn50"] == pytest.approx(0.0)


def test_alternating_rhythm():
    rr = np.array([800.0, 900.0] * 20)
    out = time_domain_hrv(rr)
    assert out["rmssd"] == pytest.approx(100.0)
    assert out["pnn50"] == pytest.approx(100.0)
    assert out["mean_rr"] == pytest.approx(850.0)


def test_invalid_beat_does_not_create_fake_difference():
    rr = np.array([800.0, 800.0, 3000.0, 800.0, 800.0])
    valid = np.array([True, True, False, True, True])
    out = time_domain_hrv(rr, valid)
    assert out["rmssd"] == pytest.approx(0.0)


def test_gap_breaks_successive_differences():
    rr = np.array([800.0, 800.0, 1000.0, 1000.0])
    adjacent = np.array([True, False, True])  # recording gap between beat 2 and 3
    assert time_domain_hrv(rr, adjacent=adjacent)["rmssd"] == pytest.approx(0.0)


def test_validity_mask_flags_out_of_range_and_spikes():
    rr = np.array([0.8] * 10 + [0.2] + [0.8] * 10 + [1.3] + [0.8] * 10)
    m = rr_validity_mask(rr)
    assert not m[10] and not m[21] and m[:10].all()


def test_adjacency_from_timestamps():
    t = np.array([0, 1, 2, 12, 13], dtype=float)
    rr = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
    assert adjacency_from_timestamps(t, rr).tolist() == [True, True, False, True]


def test_segment_hrv_coverage_rule():
    rr = np.full(400, 0.75)
    t = np.cumsum(rr)
    seg = segment_hrv(t, rr, segment_s=60, min_coverage=0.6)
    assert len(seg) >= 4 and (seg["hr"].round(1) == 80.0).all()
