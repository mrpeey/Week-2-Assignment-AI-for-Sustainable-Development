"""
SmartFarm AI FastAPI backend
- Lightweight endpoints; gracefully degrade if heavy ML libs are not installed
- Endpoints:
  - GET /health
  - POST /yield/predict
  - POST /irrigation/recommend
  - POST /market/sentiment
Run:
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from typing import List, Optional, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="SmartFarm AI API", version="0.1.0")

# Optional imports with graceful fallback
try:
    from src.yield_prediction import YieldPredictionSystem
    HAS_YIELD = True
except Exception:
    HAS_YIELD = False
    YieldPredictionSystem = None  # type: ignore

try:
    from src.smart_irrigation import SmartIrrigationController
    HAS_IRRIGATION = True
except Exception:
    HAS_IRRIGATION = False
    SmartIrrigationController = None  # type: ignore

try:
    from src.market_intelligence import MarketIntelligenceSystem
    HAS_MARKET = True
except Exception:
    HAS_MARKET = False
    MarketIntelligenceSystem = None  # type: ignore

# Models
class YieldRequest(BaseModel):
    temperature_avg: float = 26.5
    rainfall_mm: float = 75.0
    soil_ph: float = 6.8
    nitrogen_ppm: float = 180
    ndvi_avg: float = 0.75
    fertilizer_kg_per_ha: float = 220
    irrigation_frequency: int = 12
    crop_variety_encoded: int = 2
    planting_month: int = 4
    growing_season_days: int = 125

class IrrigationRequest(BaseModel):
    soil_moisture: float = Field(45, ge=0, le=100)
    temperature: float = 28
    rainfall_forecast: float = 0
    days_since_planting: int = 45
    growth_stage: float = Field(0.4, ge=0, le=1)
    evapotranspiration: float = 6.5
    cumulative_water: float = 150
    humidity: float = 60

class SentimentRequest(BaseModel):
    texts: List[str]

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "modules": {
            "yield_prediction": HAS_YIELD,
            "irrigation": HAS_IRRIGATION,
            "market": HAS_MARKET,
        },
    }

@app.post("/yield/predict")
async def predict_yield(payload: YieldRequest):
    data = payload.dict()
    if HAS_YIELD:
        try:
            yp = YieldPredictionSystem()
            # tiny dataset to initialize scalers/models quickly
            df = yp.create_synthetic_data(n_samples=800)
            results, _ = yp.train_ensemble(df)
            pred = yp.predict_yield(data)
            return {
                "ok": True,
                "ensemble_prediction": pred["ensemble_prediction"],
                "confidence_interval": pred["confidence_interval"],
                "models": {k: {"rmse": v["rmse"], "r2": v["r2"]} for k, v in results.items()},
            }
        except Exception as e:
            # graceful fallback
            return {
                "ok": True,
                "note": "Fallback deterministic estimate (no heavy deps)",
                "ensemble_prediction": 6.8 + 0.01 * (data.get("fertilizer_kg_per_ha", 200) - 200)
                + 0.5 * (data.get("ndvi_avg", 0.7) - 0.7),
                "confidence_interval": [6.0, 7.6],
                "error": str(e),
            }
    # No heavy libs available -> simple heuristic
    return {
        "ok": True,
        "note": "Heuristic prediction (install full requirements for models)",
        "ensemble_prediction": 6.5 + 0.5 * (data.get("ndvi_avg", 0.7) - 0.7),
        "confidence_interval": [6.0, 7.2],
    }

@app.post("/irrigation/recommend")
async def irrigation_recommend(payload: IrrigationRequest):
    data = payload.dict()
    if HAS_IRRIGATION:
        try:
            ctrl = SmartIrrigationController()
            rec = ctrl.get_irrigation_recommendation(data)
            return {"ok": True, **rec}
        except Exception as e:
            pass
    # Fallback simple rule
    sm = data.get("soil_moisture", 45)
    rain = data.get("rainfall_forecast", 0)
    base = 12 if sm < 40 else 6
    amount = max(0.0, base - 0.5 * rain)
    return {
        "ok": True,
        "recommended_irrigation_mm": round(amount, 1),
        "reasoning": "Rule-based fallback due to missing RL libs",
        "water_efficiency_score": 80 if 5 <= amount <= 15 else 65,
        "urgency": "High" if sm < 30 else ("Medium" if sm < 45 else "Low"),
    }

@app.post("/market/sentiment")
async def market_sentiment(payload: SentimentRequest):
    texts = payload.texts
    if HAS_MARKET:
        try:
            mis = MarketIntelligenceSystem()
            res = mis.analyze_sentiment(texts)
            return {"ok": True, "results": res}
        except Exception as e:
            pass
    # Fallback keyword-based
    POS = {"up", "increase", "demand", "growth", "strong", "surge"}
    NEG = {"down", "decrease", "shortage", "decline", "weak", "drop"}
    results = []
    for t in texts:
        tl = t.lower()
        score = sum(1 for w in POS if w in tl) - sum(1 for w in NEG if w in tl)
        results.append({
            "text": t,
            "label": "POSITIVE" if score > 0 else ("NEGATIVE" if score < 0 else "NEUTRAL"),
            "polarity": float(score) / 5.0,
        })
    return {"ok": True, "results": results}

# Uvicorn entry point (optional)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
