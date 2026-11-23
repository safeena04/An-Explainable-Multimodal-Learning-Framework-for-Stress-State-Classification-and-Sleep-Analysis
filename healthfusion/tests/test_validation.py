"""Participant-independence and statistics utilities."""
import numpy as np
import pytest

from src.evaluation.cv import assert_no_group_overlap, repeated_group_splits
from src.evaluation.metrics import classification_metrics, pairwise_accuracy
from src.evaluation.stats import benjamini_hochberg, corrected_resampled_ttest, permute_within_groups


def test_splits_never_share_participants():
    groups = np.repeat([f"P{i}" for i in range(30)], 2)
    for _, _, tr, te in repeated_group_splits(groups):
        assert not set(groups[tr]) & set(groups[te])


def test_overlap_detector():
    groups = np.array(["a", "a", "b", "b"])
    with pytest.raises(AssertionError):
        assert_no_group_overlap(groups, np.array([0, 2]), np.array([1, 3]))


def test_within_group_permutation_keeps_composition():
    groups = np.repeat(np.arange(20), 2)
    y = np.tile([0, 1], 20)
    yp = permute_within_groups(y, groups, np.random.default_rng(0))
    for g in range(20):
        assert sorted(yp[groups == g]) == [0, 1]


def test_pairwise_accuracy():
    assert pairwise_accuracy(["a", "a", "b", "b"], [0, 1, 0, 1], [0.2, 0.8, 0.6, 0.4]) == 0.5


def test_metrics_perfect():
    m = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["roc_auc"] == 1 and m["balanced_accuracy"] == 1 and m["tp"] == 2


def test_bh_matches_manual():
    adj = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.2]))
    # sorted p: 0.01, 0.03, 0.04, 0.2 -> raw 0.04, 0.06, 0.0533, 0.2 -> step-up minimum
    np.testing.assert_allclose(adj, [0.04, 0.16 / 3, 0.16 / 3, 0.2], rtol=1e-6)


def test_corrected_ttest_is_more_conservative():
    d = np.random.default_rng(0).normal(0.05, 0.05, 25)
    naive_t = d.mean() / (d.std(ddof=1) / np.sqrt(25))
    assert abs(corrected_resampled_ttest(d, 80, 20)["t"]) < abs(naive_t)


def test_holm_adjustment():
    from src.evaluation.stats import holm
    np.testing.assert_allclose(holm(np.array([0.01, 0.04, 0.03, np.nan])), [0.03, 0.06, 0.06, np.nan])
