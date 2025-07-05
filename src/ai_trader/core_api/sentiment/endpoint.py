import asyncio
from fastapi import APIRouter
from pydantic import BaseModel
from ai_trader.core_api.sentiment.services import predict_sentiment

router = APIRouter()

class PredictRequest(BaseModel):
    headline: str

class Scores(BaseModel):
    negative: float
    neutral: float
    positive: float

class Prediction(BaseModel):
    sentiment: str
    confidence: float
    scores: Scores


@router.post("/sentiment", response_model=Prediction)
def predict_endpoint(req: PredictRequest):
    obj = predict_sentiment(req.headline)
    return Prediction(**obj)