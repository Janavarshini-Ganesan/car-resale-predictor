"""
main.py
-------
FastAPI backend for the Car Resale Price Prediction (Lasso Regression) project.

Run with:
    uvicorn main:app --reload --port 8000
"""

import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schema import CarFeatures, PredictionResponse, FeatureImportanceResponse

app = FastAPI(title="Car Resale Price Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_lasso.pkl")
IMPORTANCE_PATH = os.path.join(BASE_DIR, "feature_importance.json")


@app.get("/")
def root():
    return {"message": "Car Resale Price Prediction API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CarFeatures):
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Model not found. Run train.py first to train and save the model.",
        )

    input_df = pd.DataFrame([features.dict()])
    predicted = model.predict(input_df)[0]
    predicted = max(0.0, float(predicted))

    return PredictionResponse(predicted_price=round(predicted, 2))


@app.get("/feature-importance", response_model=FeatureImportanceResponse)
def feature_importance():
    try:
        with open(IMPORTANCE_PATH) as f:
            importance = json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Feature importance file not found. Run train.py first.",
        )
    return FeatureImportanceResponse(importance=importance)
