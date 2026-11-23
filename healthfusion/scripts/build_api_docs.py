"""Write docs/API.md with request/response examples captured from the real API.

Run (after the pipelines and train_final_models):  python scripts/build_api_docs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402


def trim(obj, max_list=4, max_str=160):
    if isinstance(obj, dict):
        return {k: trim(v, max_list, max_str) for k, v in obj.items()}
    if isinstance(obj, list):
        out = [trim(v, max_list, max_str) for v in obj[:max_list]]
        if len(obj) > max_list:
            out.append(f"... {len(obj) - max_list} more")
        return out
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, str) and len(obj) > max_str:
        return obj[:max_str] + "..."
    return obj


def block(title, method, path, body=None, note="", client=None, show_body=True):
    r = client.post(path, json=body) if method == "POST" else client.get(path)
    lines = [f"### `{method} {path}`", ""]
    if note:
        lines += [note, ""]
    if body is not None and show_body:
        lines += ["Request body:", "```json", json.dumps(trim(body, 6), indent=2), "```", ""]
    lines += [f"Response `{r.status_code}` (lists trimmed):", "```json", json.dumps(trim(r.json()), indent=2), "```", ""]
    return lines


def main() -> None:
    c = TestClient(app)
    sample = lambda n: json.loads((ROOT / "data" / "sample" / n).read_text())  # noqa: E731
    doc = ["# HealthFusion API reference", "",
           "Start the server from the project root with `uvicorn api.main:app --reload` and open "
           "`http://127.0.0.1:8000/docs` for the interactive Swagger UI. Every example below was captured "
           "from the running API by `scripts/build_api_docs.py`.", "",
           "Status codes: `200` OK; `404` unknown resource; `422` invalid input, incomplete modality, or a "
           "modality set whose model failed validation; `423` prediction locked because the model was not "
           "validated; `503` pipeline outputs missing.", "", "## Service", ""]
    doc += block("health", "GET", "/health", client=c)
    doc += ["## Datasets", ""]
    doc += block("datasets", "GET", "/datasets", client=c)
    doc += block("qc", "GET", "/datasets/qol_stress/qc?only_excluded=true", client=c)
    doc += ["## Stress classification (QoL_Stress)", ""]
    doc += block("models", "GET", "/stress/models", client=c)
    doc += block("predict", "POST", "/stress/predict", sample("stress_features_stress.json"), client=c,
                 note="Supply all features of each modality you want used. The model trained on exactly that "
                      "modality set is selected.")
    eda_only = {"features": {k: v for k, v in sample("stress_features_relax.json")["features"].items()
                             if k.startswith("eda_")}, "explain": False}
    doc += block("predict eda", "POST", "/stress/predict", eda_only, client=c, note="EDA only:")
    ecg_only = {"features": {k: v for k, v in sample("stress_features_relax.json")["features"].items()
                             if k.startswith("ecg_")}}
    doc += block("predict ecg", "POST", "/stress/predict", ecg_only, client=c,
                 note="ECG only - refused when the ECG-only model failed validation:")
    raw = sample("stress_raw_relax.json")
    doc += ["### `POST /stress/predict/raw`", "",
            "Raw signals are processed by the same code as the training pipeline "
            "(`src/preprocessing/qol_signals.py`). Body shape:", "```json",
            json.dumps({"ecg": {"samples": "[7500 numbers]", "sampling_rate": 250},
                        "eda": {"scl": "[~160 numbers, 1 per second]", "activation": "[0/1 flags]",
                                "valid": "[true/false, optional]", "sampling_rate": 1.0},
                        "heart_rate": {"bpm": "[numbers]", "timestamps_s": "[seconds]"}, "explain": True},
                       indent=2), "```", ""]
    doc += block("raw", "POST", "/stress/predict/raw", raw, client=c, show_body=False)
    doc += block("stats", "GET", "/stress/statistics", client=c)
    doc += ["## Sleep analysis (MMASH)", ""]
    doc += block("analysis", "GET", "/sleep/analysis?target=sleep_efficiency", client=c)
    doc += block("sleep predict", "POST", "/sleep/predict", {"features": {"phys_hr_mean": 80}}, client=c)
    doc += ["## Performance, modalities, explanations, cross-dataset", ""]
    doc += block("perf", "GET", "/performance/stress?top=3", client=c)
    doc += block("mods", "GET", "/modalities/stress", client=c)
    doc += block("global", "GET", "/explain/stress/global", client=c)
    doc += block("instance", "GET", "/explain/stress/instance/Q03/stress", client=c)
    doc += block("cross", "GET", "/cross-dataset", client=c)
    (ROOT / "docs" / "API.md").write_text("\n".join(doc))
    print("docs/API.md written")


if __name__ == "__main__":
    main()
