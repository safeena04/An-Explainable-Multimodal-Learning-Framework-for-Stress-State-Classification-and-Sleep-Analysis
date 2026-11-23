"""Explainability endpoints (SHAP). SHAP describes model behaviour, not causation."""
import re

from fastapi import APIRouter, HTTPException

from api import services as S

router = APIRouter(prefix="/explain", tags=["explainability"])


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


@router.get("/summary")
def summary():
    return S.read_json("shap_summary.json")


@router.get("/stress/global")
def stress_global(experiment: str | None = None):
    summ = S.read_json("shap_summary.json")["qol_stress"]
    available = [k for k in summ if " | " in k]
    exp = experiment or S.read_json("qol_experiment_meta.json")["best_overall"]
    if exp not in available:
        raise HTTPException(404, {"message": f"no SHAP results for '{exp}'", "available": available})
    return {"experiment": exp, "importance": S.records(S.read_csv(f"shap_global_{_slug(exp)}.csv")),
            "details": summ[exp], "note": "out-of-fold SHAP: each value explains a held-out participant"}


@router.get("/stress/instance/{pid}/{phase}")
def stress_instance(pid: str, phase: str, experiment: str | None = None):
    """Stored out-of-fold explanation for one participant-phase of the study data."""
    exp = experiment or S.read_json("qol_experiment_meta.json")["best_overall"]
    df = S.read_csv(f"shap_oof_values_{_slug(exp)}.csv")
    row = df[(df.pid == pid) & (df.phase == phase)]
    if row.empty:
        raise HTTPException(404, f"{pid}/{phase} is not in the complete-case set")
    r = row.iloc[0]
    feats = [c for c in df.columns if c.startswith(("ecg_", "eda_", "fb_"))]
    contrib = sorted(({"feature": c, "shap": float(r[c])} for c in feats), key=lambda d: -abs(d["shap"]))
    return S.clean({"experiment": exp, "pid": pid, "phase": phase, "prob_stress_out_of_fold": r["prob"],
                    "base_value": r["base_value"], "contributions": contrib})


@router.get("/sleep/global")
def sleep_global():
    return {"importance": S.records(S.read_csv("shap_global_mmash_sleep_efficiency_ridge.csv")),
            "details": S.read_json("shap_summary.json")["mmash"]}
