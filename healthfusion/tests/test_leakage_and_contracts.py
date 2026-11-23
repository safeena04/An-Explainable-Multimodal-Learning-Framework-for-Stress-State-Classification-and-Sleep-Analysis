"""Guards against target leakage and checks of processed-data contracts."""
import pandas as pd

from src.features.registry import FEATURE_TABLE
from src.preprocessing.mmash_preprocess import BEHAV_FEATURES, MMASH_MODALITIES, PHYS_FEATURES, PSYCH_FEATURES
from src.preprocessing.qol_preprocess import MODALITIES

from tests.conftest import ROOT, needs_processed

QOL_INPUTS = [c for cols in MODALITIES.values() for c in cols]
MMASH_INPUTS = PHYS_FEATURES + BEHAV_FEATURES + PSYCH_FEATURES


def test_no_questionnaire_or_label_in_qol_inputs():
    banned = ("stai", "pss", "phase", "label", "time", "device", "class")
    assert not [c for c in QOL_INPUTS if any(b in c for b in banned)]


def test_no_sleep_derived_or_post_sleep_variable_in_mmash_inputs():
    banned = ("efficiency", "tst", "waso", "fragment", "psqi", "pittsburgh", "next_morning", "awaken", "onset")
    assert not [c for c in MMASH_INPUTS if any(b in c.lower() for b in banned)]


def test_every_model_input_is_documented():
    documented = set(FEATURE_TABLE.feature)
    assert set(QOL_INPUTS) <= documented and set(MMASH_INPUTS) <= documented


def test_modalities_do_not_overlap():
    for mods in (MODALITIES, MMASH_MODALITIES):
        cols = [c for v in mods.values() for c in v]
        assert len(cols) == len(set(cols))


@needs_processed
def test_qol_processed_contract():
    f = pd.read_csv(ROOT / "data/processed/qol_phase_features.csv")
    assert len(f) == 132 and f.pid.nunique() == 66
    assert (f.label == (f.phase == "stress").astype(int)).all()
    cc = f[f.complete_case]
    assert (cc.groupby("pid").size() == 2).all()
    assert cc[QOL_INPUTS].notna().all().all()


@needs_processed
def test_mmash_processed_contract():
    m = pd.read_csv(ROOT / "data/processed/mmash_participant_features.csv")
    assert len(m) == 21 and 11 not in set(m.user)
    assert m.sleep_efficiency.between(0, 100).all()
