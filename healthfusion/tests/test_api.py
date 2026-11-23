"""End-to-end API tests against the trained artefacts."""
import json

import numpy as np
import pytest

from tests.conftest import ROOT, needs_models

pytestmark = needs_models


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def _sample(name):
    return json.loads((ROOT / "data" / "sample" / name).read_text())


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


@pytest.mark.parametrize("path", ["/", "/datasets", "/datasets/qol_stress", "/datasets/mmash", "/datasets/mmash/qc",
                                  "/stress/models", "/stress/statistics", "/sleep/analysis", "/sleep/model",
                                  "/performance/stress", "/performance/sleep", "/modalities/stress",
                                  "/modalities/sleep", "/explain/summary", "/explain/stress/global",
                                  "/explain/sleep/global", "/cross-dataset"])
def test_get_endpoints(client, path):
    r = client.get(path)
    assert r.status_code == 200, r.text


def test_unknown_dataset_404(client):
    assert client.get("/datasets/nope").status_code == 404


def test_predict_features_all_modalities(client):
    r = client.post("/stress/predict", json=_sample("stress_features_stress.json"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["probability_stress"] <= 1
    assert set(body["modalities_used"]) == {"ECG", "EDA", "HR"}
    ex = body["explanation"]
    if ex["output_scale"] == "log-odds":
        logit = ex["base_value"] + sum(c["shap"] for c in ex["contributions"])
        assert 1 / (1 + np.exp(-logit)) == pytest.approx(body["probability_stress"], abs=0.02)


def test_predict_raw_signals(client):
    r = client.post("/stress/predict/raw", json=_sample("stress_raw_relax.json"))
    assert r.status_code == 200, r.text
    assert set(r.json()["extracted_features"]) >= {"ecg_rmssd", "eda_scl_mean", "fb_hr_mean"}


def test_raw_and_feature_paths_agree(client):
    raw = client.post("/stress/predict/raw", json=_sample("stress_raw_stress.json")).json()
    feat = client.post("/stress/predict", json=_sample("stress_features_stress.json")).json()
    assert raw["probability_stress"] == pytest.approx(feat["probability_stress"], abs=1e-6)


def test_partial_modality_rejected(client):
    r = client.post("/stress/predict", json={"features": {"ecg_hr": 70}})
    assert r.status_code == 422


def test_unvalidated_modality_set_not_served(client):
    from api import services as S
    bad = [e for e in S.stress_entries() if not e["validated"]]
    if not bad:
        pytest.skip("every modality set passed validation")
    mods = set(bad[0]["modalities"])
    feats = {k: v for k, v in _sample("stress_features_relax.json")["features"].items()
             if any(k.startswith(p) for p, m in (("ecg_", "ECG"), ("eda_", "EDA"), ("fb_", "HR")) if m in mods)}
    r = client.post("/stress/predict", json={"features": feats})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "model_not_validated"


def test_sleep_predict_respects_validation(client):
    card = client.get("/sleep/model").json()
    r = client.post("/sleep/predict", json={"features": {}})
    if card["validation"]["validated"]:
        assert r.status_code in (200, 422)
    else:
        assert r.status_code == 423


def test_instance_explanation(client):
    r = client.get("/explain/stress/instance/Q03/stress")
    assert r.status_code == 200 and len(r.json()["contributions"]) > 0
