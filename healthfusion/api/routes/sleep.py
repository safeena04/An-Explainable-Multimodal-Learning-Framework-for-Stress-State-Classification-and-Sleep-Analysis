"""Sleep analysis endpoints (MMASH)."""
import pandas as pd
from fastapi import APIRouter, HTTPException

from api import services as S
from api.schemas import SleepFeatureRequest
from src.models.registry import load_artifact

router = APIRouter(prefix="/sleep", tags=["sleep analysis"])


@router.get("/analysis")
def analysis(target: str = "sleep_efficiency", fdr_only: bool = False):
    """Participant-level correlations and LOSO regression results for one target."""
    corr = S.read_csv("mmash_correlations.csv")
    corr = corr[corr.target == target]
    if corr.empty:
        raise HTTPException(404, f"unknown target '{target}'")
    if fdr_only:
        corr = corr[corr.p_fdr < 0.05]
    res = S.read_csv("mmash_loso_results.csv")
    return {"target": target, "n_participants": int(res.n.max()),
            "meta": S.read_json("mmash_experiment_meta.json"),
            "correlations": S.records(corr.sort_values("p_value")),
            "loso_regression": S.records(res[res.target == target].sort_values("mae"))}


@router.get("/model")
def model_card():
    return S.clean(S.card("sleep", "sleep_efficiency"))


@router.post("/predict")
def predict(req: SleepFeatureRequest):
    """Served only if the sleep model passed validation; otherwise 423 Locked with the evidence."""
    card = S.card("sleep", "sleep_efficiency")
    if not card["validation"]["validated"]:
        raise HTTPException(423, detail={
            "message": "Sleep prediction is disabled: no model beat the predict-the-mean baseline under "
                       "leave-one-subject-out validation on MMASH (n = 21). See /sleep/analysis.",
            "validation": S.clean(card["validation"])})
    art = load_artifact("sleep", "sleep_efficiency")
    missing = [c for c in art["features"] if req.features.get(c) is None]
    if missing:
        raise HTTPException(422, detail={"missing_features": missing})
    X = pd.DataFrame([{c: req.features[c] for c in art["features"]}])
    return {"sleep_efficiency_pred": float(art["estimator"].predict(X)[0]), "validation": S.clean(card["validation"]),
            "disclaimer": S.DISCLAIMER}
