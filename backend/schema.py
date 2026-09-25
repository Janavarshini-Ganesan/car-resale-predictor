from pydantic import BaseModel
from typing import Dict


class CarFeatures(BaseModel):
    Age_Years: int
    Mileage_km: int
    Engine_CC: int
    Owner_Count: int
    Accident_History: int  # 0 or 1
    Color_Popularity_Score: float
    Local_Fuel_Price_Index: float
    Service_Center_Visits: int


class PredictionResponse(BaseModel):
    predicted_price: float


class FeatureImportanceResponse(BaseModel):
    importance: Dict[str, float]
