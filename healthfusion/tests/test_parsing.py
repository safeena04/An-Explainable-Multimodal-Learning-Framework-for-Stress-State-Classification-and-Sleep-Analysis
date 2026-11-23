import numpy as np
import pytest

from src.data.qol_loader import parse_waveform, time_of_day_seconds


def test_split_negative_token_is_one_interpolated_sample():
    wp = parse_waveform("10 20 30 - 39 50 60")
    assert len(wp.samples) == 6
    assert wp.n_repaired == 1
    assert wp.samples[3] == pytest.approx(40.0)


def test_clean_waveform_untouched():
    wp = parse_waveform("1 -2 3")
    np.testing.assert_array_equal(wp.samples, [1, -2, 3])
    assert wp.n_repaired == 0


@pytest.mark.parametrize("s,expected", [("12:07:52", 43672), ("9:05:00", 32700), ("Tue Dec 17 13:01:42", 46902),
                                        (" 9:13:58", 33238)])
def test_time_formats(s, expected):
    assert time_of_day_seconds(s) == expected


def test_bad_time_is_nan():
    assert np.isnan(time_of_day_seconds("not a time"))
