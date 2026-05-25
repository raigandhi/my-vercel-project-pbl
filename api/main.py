from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pickle
import numpy as np
import os

app = FastAPI(
    title="Ocean Weather Prediction API",
    description="API prediksi cuaca laut wilayah Jawa Timur",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lokasi Jawa Timur ─────────────────────────────────────────────────────────
LOCATIONS = {
    "pacitan":        {"name": "Pacitan",        "lat": -8.20, "lon": 111.10},
    "prigi":          {"name": "Prigi",          "lat": -8.29, "lon": 111.74},
    "popoh":          {"name": "Popoh",          "lat": -8.17, "lon": 111.89},
    "sendang_biru":   {"name": "Sendang Biru",   "lat": -8.43, "lon": 112.69},
    "puger":          {"name": "Puger",          "lat": -8.38, "lon": 113.47},
    "pancer":         {"name": "Pancer",         "lat": -8.63, "lon": 114.02},
    "muncar":         {"name": "Muncar",         "lat": -8.44, "lon": 114.33},
    "grajagan":       {"name": "Grajagan",       "lat": -8.66, "lon": 114.23},
    "watu_ulo":       {"name": "Watu Ulo",       "lat": -8.45, "lon": 113.72},
    "blitar_selatan": {"name": "Blitar Selatan", "lat": -8.33, "lon": 112.19},
    "tuban":          {"name": "Tuban",          "lat": -6.90, "lon": 112.05},
    "brondong":       {"name": "Brondong",       "lat": -6.89, "lon": 112.27},
    "paciran":        {"name": "Paciran",        "lat": -6.87, "lon": 112.34},
    "gresik":         {"name": "Gresik",         "lat": -7.15, "lon": 112.65},
    "surabaya":       {"name": "Surabaya",       "lat": -7.19, "lon": 112.65},
    "pasuruan":       {"name": "Pasuruan",       "lat": -7.64, "lon": 112.91},
    "probolinggo":    {"name": "Probolinggo",    "lat": -7.74, "lon": 113.23},
    "situbondo":      {"name": "Situbondo",      "lat": -7.70, "lon": 114.00},
    "banyuwangi":     {"name": "Banyuwangi",     "lat": -8.21, "lon": 114.37},
    "bangkalan":      {"name": "Bangkalan",      "lat": -6.95, "lon": 112.73},
    "sampang":        {"name": "Sampang",        "lat": -7.19, "lon": 113.24},
    "pamekasan":      {"name": "Pamekasan",      "lat": -7.16, "lon": 113.48},
    "sumenep":        {"name": "Sumenep",        "lat": -7.02, "lon": 113.86},
    "kangean":        {"name": "Kangean",        "lat": -6.93, "lon": 115.32},
    "selat_madura":   {"name": "Selat Madura",   "lat": -7.10, "lon": 113.00},
    "selat_bali":     {"name": "Selat Bali",     "lat": -8.17, "lon": 114.43},
}

# ── Model loader ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_models")

MODEL_FILES = {
    "ocean_current_velocity":  "model_ocean_current_velocity.pkl",
    "precipitation":           "model_precipitation.pkl",
    "sea_surface_temperature": "model_sea_surface_temperature.pkl",
    "visibility":              "model_visibility.pkl",
    "wave_height":             "model_wave_height.pkl",
    "wind_speed_10m":          "model_wind_speed_10m.pkl",
}

_models: dict = {}

def load_model(name: str):
    if name not in _models:
        path = os.path.join(MODEL_DIR, MODEL_FILES[name])
        if not os.path.exists(path):
            raise HTTPException(status_code=503, detail=f"Model '{name}' tidak ditemukan di server.")
        with open(path, "rb") as f:
            _models[name] = pickle.load(f)
    return _models[name]


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictionByLocation(BaseModel):
    location: str          # key lokasi, contoh: "surabaya"
    month: int             # 1-12
    day_of_year: int       # 1-366
    sea_level_pressure: Optional[float] = None
    air_temperature: Optional[float] = None
    humidity: Optional[float] = None
    cloud_cover: Optional[float] = None

class PredictionResponse(BaseModel):
    location: str
    lat: float
    lon: float
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
def build_feature_array(lat: float, lon: float, data: PredictionByLocation) -> np.ndarray:
    """
    Sesuaikan urutan fitur ini dengan kolom saat training.
    """
    features = [
        lat,
        lon,
        data.month,
        data.day_of_year,
        data.sea_level_pressure or 1013.25,
        data.air_temperature    or 25.0,
        data.humidity           or 75.0,
        data.cloud_cover        or 50.0,
    ]
    return np.array(features).reshape(1, -1)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Ocean Weather Prediction API — Jawa Timur",
        "docs": "/docs",
        "available_targets": list(MODEL_FILES.keys()),
        "total_locations": len(LOCATIONS),
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/locations")
def list_locations():
    """Daftar semua lokasi yang tersedia beserta koordinatnya."""
    return {
        "total": len(LOCATIONS),
        "locations": [
            {"key": key, **val}
            for key, val in LOCATIONS.items()
        ]
    }

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
def predict(target: str, data: PredictionByLocation):
    if target not in MODEL_FILES:
        raise HTTPException(
            status_code=404,
            detail=f"Target '{target}' tidak dikenal. Tersedia: {list(MODEL_FILES.keys())}"
        )
    loc_key = data.location.lower().replace(" ", "_")
    if loc_key not in LOCATIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Lokasi '{data.location}' tidak ditemukan. Cek /locations untuk daftar lengkap."
        )
    loc = LOCATIONS[loc_key]
    model = load_model(target)
    X = build_feature_array(loc["lat"], loc["lon"], data)
    prediction = float(model.predict(X)[0])

    return PredictionResponse(
        location=loc["name"],
        lat=loc["lat"],
        lon=loc["lon"],
        target=target,
        prediction=round(prediction, 4),
        unit=UNITS[target],
    )


@app.post("/predict/all")
def predict_all(data: PredictionByLocation):
    """Prediksi semua 6 target sekaligus untuk satu lokasi."""
    loc_key = data.location.lower().replace(" ", "_")
    if loc_key not in LOCATIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Lokasi '{data.location}' tidak ditemukan. Cek /locations untuk daftar lengkap."
        )
    loc = LOCATIONS[loc_key]
    X = build_feature_array(loc["lat"], loc["lon"], data)

    results = {}
    for name in MODEL_FILES:
        try:
            model = load_model(name)
            results[name] = {
                "prediction": round(float(model.predict(X)[0]), 4),
                "unit": UNITS[name],
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return {
        "location": loc["name"],
        "lat": loc["lat"],
        "lon": loc["lon"],
        "month": data.month,
        "day_of_year": data.day_of_year,
        "predictions": results,
    }