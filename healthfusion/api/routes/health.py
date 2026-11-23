"""Service health and index."""
from fastapi import APIRouter

from api import services as S

router = APIRouter(tags=["service"])


@router.get("/health")
def health():
    try:
        reg = S.registry()
        models = {"stress_models": len(S.stress_entries()),
                  "stress_models_served": sum(e["validated"] for e in S.stress_entries()),
                  "sleep_model_served": bool((S.sleep_entry() or {}).get("validated", False))}
        status = "ok"
    except S.ArtifactMissing as exc:
        reg, models, status = {"environment": {}}, {"error": str(exc)}, "degraded"
    return {"status": status, **models, "environment": reg.get("environment", {})}
