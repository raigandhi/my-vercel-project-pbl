from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pickle
import numpy as np
import os

app = FastAPI(
    title="Ocean Weather Prediction API",
    description="API for predicting ocean and weather conditions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loader ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_models")

MODEL_FILES = {
    "ocean_current_velocity": "model_ocean_current_velocity.pkl",
    "precipitation":          "model_precipitation.pkl",
    "sea_surface_temperature":"model_sea_surface_temperature.pkl",
    "visibility":             "model_visibility.pkl",
    "wave_height":            "model_wave_height.pkl",
    "wind_speed_10m":         "model_wind_speed_10m.pkl",
}

_models: dict = {}

def load_model(name: str):
    if name not in _models:
        path = os.path.join(MODEL_DIR, MODEL_FILES[name])
        if not os.path.exists(path):
            raise HTTPException(status_code=503, detail=f"Model '{name}' not found on server.")
        with open(path, "rb") as f:
            _models[name] = pickle.load(f)
    return _models[name]


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictionInput(BaseModel):
    """
    Adjust features to match what your models were trained on.
    These are example features — replace with your actual feature columns.
    """
    latitude: float
    longitude: float
    month: int                        # 1-12
    day_of_year: int                  # 1-366
    sea_level_pressure: Optional[float] = None
    air_temperature: Optional[float] = None
    humidity: Optional[float] = None
    cloud_cover: Optional[float] = None

class PredictionResponse(BaseModel):
    target: str
    prediction: float
    unit: str


# ── Unit labels ───────────────────────────────────────────────────────────────
UNITS = {
    "ocean_current_velocity":  "m/s",
    "precipitation":           "mm",
    "sea_surface_temperature": "°C",
    "visibility":              "km",
    "wave_height":             "m",
    "wind_speed_10m":          "m/s",
}


# ── Helper ────────────────────────────────────────────────────────────────────
def build_feature_array(data: PredictionInput) -> np.ndarray:
    """
    Build numpy array from input. Order must match training feature order.
    Adjust this function to match your actual training columns.
    """
    features = [
        data.latitude,
        data.longitude,
        data.month,
        data.day_of_year,
        data.sea_level_pressure or 1013.25,
        data.air_temperature   or 25.0,
        data.humidity          or 75.0,
        data.cloud_cover       or 50.0,
    ]
    return np.array(features).reshape(1, -1)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Ocean Weather Prediction API",
        "docs": "/docs",
        "available_targets": list(MODEL_FILES.keys()),
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models")
def list_models():
    available = []
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        available.append({
            "name": name,
            "unit": UNITS[name],
            "loaded": name in _models,
            "file_exists": os.path.exists(path),
        })
    return {"models": available}


@app.post("/predict/{target}", response_model=PredictionResponse)
def predict(target: str, data: PredictionInput):
    if target not in MODEL_FILES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown target '{target}'. Available: {list(MODEL_FILES.keys())}"
        )
    model = load_model(target)
    X = build_feature_array(data)
    prediction = float(model.predict(X)[0])
    return PredictionResponse(
        target=target,
        prediction=round(prediction, 4),
        unit=UNITS[target],
    )


@app.post("/predict/all")
def predict_all(data: PredictionInput):
    """Run prediction on all 6 targets at once."""
    results = {}
    X = build_feature_array(data)
    for name in MODEL_FILES:
        try:
            model = load_model(name)
            results[name] = {
                "prediction": round(float(model.predict(X)[0]), 4),
                "unit": UNITS[name],
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    return {"input": data.model_dump(), "predictions": results}