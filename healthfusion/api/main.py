"""HealthFusion FastAPI backend.

Start:   uvicorn api.main:app --reload            (from the project root)
Docs:    http://127.0.0.1:8000/docs               (interactive Swagger UI)

Design rules
* Sleep and stress endpoints are separate (different datasets, targets, protocols).
* Every research number is read from pipeline outputs; the API never recomputes them.
* Prediction is served only by models whose model card says validated = true.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes import cross_dataset, datasets, explain, health, modalities, performance, sleep, stress
from api.services import ArtifactMissing
from src.config import FIGURES_DIR

app = FastAPI(
    title="HealthFusion API",
    version="1.0.0",
    description="Explainable multimodal analysis of stress (QoL_Stress) and sleep (MMASH). "
                "Research prototype - not a medical device.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@app.exception_handler(ArtifactMissing)
async def _missing(_: Request, exc: ArtifactMissing):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/", tags=["service"])
def index():
    return {"name": "HealthFusion API", "docs": "/docs", "endpoints": {
        "health": "/health",
        "datasets": ["/datasets", "/datasets/{qol_stress|mmash}", "/datasets/{name}/qc"],
        "stress": ["/stress/models", "/stress/models/{key}", "POST /stress/predict", "POST /stress/predict/raw",
                   "/stress/statistics"],
        "sleep": ["/sleep/analysis", "/sleep/model", "POST /sleep/predict (locked unless validated)"],
        "performance": ["/performance/stress", "/performance/stress/folds?experiment=", "/performance/sleep"],
        "modalities": ["/modalities/stress", "/modalities/sleep"],
        "explain": ["/explain/summary", "/explain/stress/global", "/explain/stress/instance/{pid}/{phase}",
                    "/explain/sleep/global"],
        "cross_dataset": "/cross-dataset",
        "figures": "/figures/{name}.png",
    }}


for r in (health, datasets, stress, sleep, performance, modalities, explain, cross_dataset):
    app.include_router(r.router)

app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")
