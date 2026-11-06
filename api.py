"""FastAPI service for the Travel.pk booking conversion model.

Serves the trained XGBoost model (model_xgboost.pkl). The model was
trained on synthetic data only; no real Travel.pk booking data was
available. See README.md for the data statement and limitations.

Endpoints:
  GET  /health   - liveness check
  POST /predict  - booking conversion probability for one visitor session
"""
import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_xgboost.pkl")

app = FastAPI(
    title="Travel.pk Booking Conversion Prediction",
    description="Predicts the probability that a Travel.pk visitor completes "
    "a booking. Prototype trained on synthetic data only.",
    version="1.0.0",
)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


class BookingRequest(BaseModel):
    trip_type: str = Field(..., examples=["flight"],
                           description="flight, hotel, tour, umrah or holiday_package")
    price_pkr: int = Field(..., ge=0, examples=[45000])
    days_advance: int = Field(..., ge=0, examples=[30])
    party_size: int = Field(..., ge=1, examples=[2])
    trip_duration_days: int = Field(..., ge=1, examples=[5])
    device: str = Field(..., examples=["mobile"],
                        description="mobile or desktop")
    channel: str = Field(..., examples=["organic"],
                         description="organic, paid_search, social or direct")
    pages_viewed: int = Field(..., ge=0, examples=[8])
    time_on_site_min: float = Field(..., ge=0, examples=[6.5])
    returning_visitor: int = Field(..., ge=0, le=1, examples=[1])
    weekend_departure: int = Field(..., ge=0, le=1, examples=[0])


class BookingResponse(BaseModel):
    booking_probability: float
    predicted_label: int
    predicted_text: str


@app.get("/health")
def health():
    return {"status": "ok", "model": "xgboost", "model_file": "model_xgboost.pkl"}


@app.post("/predict", response_model=BookingResponse)
def predict(req: BookingRequest):
    model = get_model()
    row = pd.DataFrame([req.model_dump()])
    proba = float(model.predict_proba(row)[0, 1])
    label = int(proba >= 0.5)
    return BookingResponse(
        booking_probability=round(proba, 4),
        predicted_label=label,
        predicted_text="likely to book" if label == 1 else "unlikely to book",
    )
