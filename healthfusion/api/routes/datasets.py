"""Dataset information endpoints (Phase 1 inspection results and QC logs)."""
from fastapi import APIRouter, HTTPException

from api import services as S

router = APIRouter(prefix="/datasets", tags=["datasets"])

_DESCRIPTIONS = {
    "qol_stress": {
        "name": "QoL_Stress", "source": "PhysioNet, version 1.0.0", "license": "CC BY 4.0",
        "design": "66 adults, one session each, relax phase followed by stress phase (fixed order)",
        "modalities": {"ECG": "Fitbit Charge 6 ECG app, 30 s per phase, 250 Hz (verified)",
                       "EDA": "Fitbit EDA scan, ~160 s per phase, 1 row/s (verified)",
                       "HR": "Fitbit optical heart rate, irregular ~2 s spacing, no phase label"},
        "role_in_project": "main quantitative contribution: stress vs relax classification",
    },
    "mmash": {
        "name": "MMASH", "source": "PhysioNet, version 1.0.0", "license": "ODbL",
        "design": "22 healthy young men, ~24 h free-living recording including one night of sleep",
        "modalities": {"Physiological": "Polar H7 inter-beat intervals", "Behavioural":
                       "ActiGraph wGT3X-BT (1 Hz) + activity diary", "Psychological": "questionnaires"},
        "role_in_project": "participant-level sleep analysis (n = 21 with a sleep record), cautious validation",
    },
}


@router.get("")
def list_datasets():
    insp = S.inspection()
    return {
        "datasets": [
            {**_DESCRIPTIONS["qol_stress"], "n_participants": insp["qol_stress"]["n_participants"]},
            {**_DESCRIPTIONS["mmash"], "n_participant_folders": insp["mmash"]["n_participant_folders"]},
        ]
    }


@router.get("/{name}")
def dataset_detail(name: str):
    if name not in _DESCRIPTIONS:
        raise HTTPException(404, f"unknown dataset '{name}'; use one of {list(_DESCRIPTIONS)}")
    return S.clean({**_DESCRIPTIONS[name], "inspection": S.inspection()[name]})


@router.get("/{name}/qc")
def dataset_qc(name: str, only_excluded: bool = False):
    if name == "qol_stress":
        df = S.qol_qc()
        if only_excluded:
            df = df[~df["included"]]
    elif name == "mmash":
        df = S.mmash_qc()
        if only_excluded:
            df = df[~df["included"]]
    else:
        raise HTTPException(404, f"unknown dataset '{name}'")
    return {"dataset": name, "n": len(df), "records": S.records(df)}
