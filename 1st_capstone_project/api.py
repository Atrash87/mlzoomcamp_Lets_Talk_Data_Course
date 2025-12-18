from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try loading model artifacts with error handling
try:
    model = joblib.load('models/best_pm25_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    logger.info(f"✅ Model loaded with {len(feature_names)} features")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    model = scaler = feature_names = None

app = FastAPI(
    title="PM2.5 Prediction API", 
    version="1.0",
    description="Predict next hour's PM2.5 concentration in Beijing"
)

# ==================== SCHEMAS (Pydantic Models) ====================
class PredictionInput(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Feature dictionary. Check /features endpoint for required features",
        example={
            "DEWP": -21.0,
            "TEMP": -11.0,
            "PRES": 1021.0,
            "Iws": 1.79,
            "Is": 0,
            "Ir": 0,
            "hour": 0,
            "month": 1,
            # Add a few example features - user can check /features for full list
        }
    )

class PredictionOutput(BaseModel):
    prediction: float = Field(..., description="Predicted PM2.5 (µg/m³)")
    aqi_category: str = Field(..., description="Air Quality Index category")
    confidence: float = Field(..., description="Prediction confidence score (0-1)")
    missing_features: List[str] = Field(default=[], description="Features not provided")

class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status")
    model_loaded: bool = Field(..., description="Whether ML model is ready")
    features_count: int = Field(..., description="Number of features model expects")

# ==================== HELPER FUNCTIONS ====================
def get_aqi_category(pm25: float) -> str:
    """Convert PM2.5 to AQI category"""
    if pm25 <= 12.0:
        return "Good"
    elif pm25 <= 35.4:
        return "Moderate"
    elif pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups"
    elif pm25 <= 150.4:
        return "Unhealthy"
    elif pm25 <= 250.4:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def check_model_ready():
    """Check if model is loaded and ready"""
    if model is None or scaler is None or feature_names is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )

# ==================== API ENDPOINTS ====================
@app.get("/")
async def root():
    return {
        "message": "PM2.5 Prediction API",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "features": "/features",
            "predict": "/predict (POST)"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model health"""
    is_loaded = all([model is not None, scaler is not None, feature_names is not None])
    return {
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded,
        "features_count": len(feature_names) if feature_names else 0
    }

@app.get("/features")
async def get_required_features():
    """Get list of all required features"""
    check_model_ready()
    return {
        "features": feature_names,
        "count": len(feature_names),
        "note": "All these features must be provided in /predict endpoint"
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Predict next hour's PM2.5 concentration.
    
    Note: Provide ALL features listed in /features endpoint.
    Missing features will be filled with 0.
    """
    check_model_ready()
    
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.features])
        
        # Check for missing features
        missing_features = []
        for feature in feature_names:
            if feature not in input_df.columns:
                missing_features.append(feature)
                input_df[feature] = 0  # Fill missing with 0
        
        # Reorder columns to match training order
        input_df = input_df[feature_names]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = float(model.predict(input_scaled)[0])
        
        # Get AQI category
        category = get_aqi_category(prediction)
        
        # Calculate confidence (simple heuristic)
        confidence = 0.85
        if len(missing_features) > 5:  # If many features missing
            confidence = max(0.5, confidence - (len(missing_features) * 0.05))
        if prediction < 0:  # Negative prediction is suspicious
            confidence = 0.6
        
        return {
            "prediction": round(prediction, 2),
            "aqi_category": category,
            "confidence": round(confidence, 2),
            "missing_features": missing_features
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"Prediction failed: {str(e)}"
        )

# Optional: Batch prediction endpoint
@app.post("/predict/batch")
async def batch_predict(batch_input: List[PredictionInput]):
    """Predict PM2.5 for multiple samples at once"""
    check_model_ready()
    
    results = []
    for item in batch_input:
        try:
            # Reuse the single prediction logic
            result = await predict(item)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "input": item.features})
    
    return {
        "predictions": results,
        "total": len(results),
        "successful": len([r for r in results if "error" not in r])
    }