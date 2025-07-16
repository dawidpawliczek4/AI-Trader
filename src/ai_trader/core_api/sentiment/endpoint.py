import asyncio
from fastapi import APIRouter
from pydantic import BaseModel
from ai_trader.core_api.sentiment.services.predict import predict_sentiment

router = APIRouter()

class PredictRequest(BaseModel):
    headline: list[str]

@router.post("/sentiment", response_model=list[float])
def predict_endpoint(req: PredictRequest):
    list_sentiments = predict_sentiment(req.headline)
    return list_sentiments