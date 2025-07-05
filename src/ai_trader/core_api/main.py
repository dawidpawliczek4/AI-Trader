from fastapi import FastAPI
from ai_trader.core_api.sentiment.endpoint import router as predict_router


app = FastAPI(
    title="AI-Trader Core Api",
    version="0.1.0"
)

app.include_router(predict_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
