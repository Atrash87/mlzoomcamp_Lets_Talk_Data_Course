# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model and features
model = joblib.load("air_quality_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

app = FastAPI(title="Air Quality Prediction API")

# Input data format
class AirQualityInput(BaseModel):
    hour: int
    day_of_week: int
    month: int
    CO_roll3: float
    # Add more features if needed (depending on your trained model)
    # For example: T: float, RH: float, AH: float, etc.

@app.get("/")
def home():
    return {"message": "Air Quality Prediction API is running!"}

@app.post("/predict")
def predict(data: AirQualityInput):
    input_df = pd.DataFrame([data.dict()], columns=feature_columns)
    pred = model.predict(input_df)[0]
    return {"predicted_CO": round(pred, 3)}

