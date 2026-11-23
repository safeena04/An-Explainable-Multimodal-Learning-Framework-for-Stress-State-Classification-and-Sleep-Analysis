"""Model performance endpoints (all numbers come from results/metrics)."""
from typing import Optional

from fastapi import APIRouter, HTTPException

from api import services as S

router = APIRouter(prefix="/performance", tags=["model performance"])


@router.get("/stress")
def stress_performance(strategy: Optional[str] = None, model: Optional[str] = None, top: int = 100):
    df = S.read_csv("qol_cv_summary.csv")
    if strategy:
        df = df[df.strategy == strategy]
    if model:
        df = df[df.model == model]
    meta = S.read_json("qol_experiment_meta.json")
    return {"validation": "5-fold participant-grouped CV x 5 repeats (nested tuning), complete-case n = "
                          f"{meta['n_participants_complete_case']}",
            "meta": meta,
            "summary": S.records(df.head(top)),
            "comparisons": S.records(S.read_csv("qol_model_comparisons.csv")),
            "permutation_tests": S.records(S.read_csv("qol_permutation_tests.csv")),
            "sensitivity": S.records(S.read_csv("qol_sensitivity.csv"))}


@router.get("/stress/folds")
def stress_folds(experiment: str):
    df = S.read_csv("qol_cv_folds.csv")
    df = df[df.experiment == experiment]
    if df.empty:
        raise HTTPException(404, f"unknown experiment '{experiment}'")
    return {"experiment": experiment, "folds": S.records(df)}


@router.get("/sleep")
def sleep_performance(target: Optional[str] = None):
    df = S.read_csv("mmash_loso_results.csv")
    if target:
        df = df[df.target == target]
    return {"validation": "leave-one-subject-out, n = 21", "results": S.records(df)}
