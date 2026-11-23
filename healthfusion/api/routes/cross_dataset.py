"""Cross-dataset (common indicator) endpoints. No transfer/generalisation claims."""
from fastapi import APIRouter

from api import services as S

router = APIRouter(prefix="/cross-dataset", tags=["cross-dataset"])


@router.get("")
def summary():
    return {"summary": S.read_json("cross_dataset_summary.json"),
            "state_effects": S.records(S.read_csv("cross_dataset_state_effects.csv")),
            "distributions": S.records(S.read_csv("cross_dataset_distributions.csv")),
            "associations": S.records(S.read_csv("cross_dataset_associations.csv"))}
