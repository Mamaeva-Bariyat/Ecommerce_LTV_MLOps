"""
FastAPI service for Customer LTV prediction.
"""
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Make src importable when running from project root
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import predict_ltv
from src.utils.io import load_config, load_model

app = FastAPI(
    title="Customer LTV Prediction API",
    description="Predict probability of return, LTV and marketing segments for e-commerce customers (Olist).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = PROJECT_ROOT / "models" / "ltv_ensemble.joblib"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Lazy load
_artifact = None
_config = None


def get_artifact():
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run training pipeline first: python -m src.pipeline",
            )
        _artifact = load_model(MODEL_PATH)
    return _artifact


def get_config():
    global _config
    if _config is None:
        _config = load_config(str(CONFIG_PATH))
    return _config


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_features: Optional[int] = None


class SegmentStats(BaseModel):
    decile: int
    mean_pred_ltv: float
    mean_proba: float
    count: int
    recommendation: str


RECOMMENDATIONS = {
    range(1, 4): "Минимальные усилия: общие рассылки 1 раз в квартал",
    range(4, 8): "Персонализация + скидка 15%",
    range(8, 11): "Максимум внимания: индивидуальные предложения, VIP",
}


def _get_recommendation(decile: int) -> str:
    for r, text in RECOMMENDATIONS.items():
        if decile in r:
            return text
    return "Стандартная коммуникация"


@app.get("/", response_model=dict)
def home():
    return {
        "message": "Customer LTV API is running",
        "docs": "/docs",
        "endpoints": ["/health", "/predict", "/predict_batch", "/segments", "/model_info"],
    }


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        art = get_artifact()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            n_features=len(art["features"]),
        )
    except Exception:
        return HealthResponse(status="model_missing", model_loaded=False)


@app.get("/model_info")
def model_info():
    art = get_artifact()
    return {
        "features": art["features"],
        "avg_return_amount": art.get("avg_return_amount"),
        "oof_metrics": art.get("oof_metrics"),
        "config_snapshot": art.get("config_snapshot"),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload a CSV with feature columns → receive predictions."""
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read CSV: {e}")

    try:
        result = predict_ltv(df, model_path=MODEL_PATH, config_path=str(CONFIG_PATH))
        # Keep only useful columns for response
        cols = [c for c in result.columns if c in df.columns or c in ("pred_proba", "pred_ltv", "ltv_decile")]
        return JSONResponse(content=result[cols].to_dict(orient="records"))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """Same as /predict but returns a downloadable-friendly structure + summary."""
    try:
        df = pd.read_csv(file.file)
        result = predict_ltv(df, model_path=MODEL_PATH, config_path=str(CONFIG_PATH))
        summary = {
            "n_customers": len(result),
            "mean_pred_ltv": float(result["pred_ltv"].mean()),
            "mean_proba": float(result["pred_proba"].mean()),
            "decile_distribution": result["ltv_decile"].value_counts().sort_index().to_dict(),
        }
        return {
            "summary": summary,
            "predictions": result.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/segments")
def segments():
    """Return marketing recommendations by LTV decile."""
    data = []
    for d in range(1, 11):
        data.append(
            {
                "decile": d,
                "recommendation": _get_recommendation(d),
            }
        )
    return data


if __name__ == "__main__":
    import uvicorn

    cfg = get_config()
    uvicorn.run(
        "backend.main:app",
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        reload=True,
    )
