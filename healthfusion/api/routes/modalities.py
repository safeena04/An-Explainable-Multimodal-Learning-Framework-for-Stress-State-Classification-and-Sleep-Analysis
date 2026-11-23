"""Modality comparison endpoints (RQ1, RQ3, RQ4)."""
from fastapi import APIRouter, HTTPException

from api import services as S

router = APIRouter(prefix="/modalities", tags=["modality comparison"])


@router.get("/{task}")
def compare(task: str):
    if task == "stress":
        best = S.read_csv("qol_best_per_modality_set.csv")
        cols = ["modalities", "experiment", "strategy", "model", "roc_auc_mean", "roc_auc_ci_low",
                "roc_auc_ci_high", "balanced_accuracy_mean", "macro_f1_mean", "pairwise_accuracy_mean"]
        return {"task": "stress", "best_per_modality_set": S.records(best[cols]),
                "shap_modality_share": S.records(S.read_csv("shap_modality_share.csv").rename(
                    columns={"Unnamed: 0": "experiment"})),
                "stacking_weights": S.stacking_weights()}
    if task == "sleep":
        res = S.read_csv("mmash_loso_results.csv")
        r = res[(res.target == "sleep_efficiency")]
        return {"task": "sleep", "target": "sleep_efficiency", "results": S.records(r.sort_values("mae"))}
    raise HTTPException(404, "task must be 'stress' or 'sleep'")
