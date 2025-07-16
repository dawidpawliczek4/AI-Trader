from fastapi import FastAPI, BackgroundTasks
from ai_trader.core_api.sentiment.endpoint import router as predict_router
from ai_trader.core_api.celery_app import celery_app
from ai_trader.core_api.tasks import sentiment_analysis_task, health_check
from pydantic import BaseModel
from typing import Dict, Any


app = FastAPI(
    title="AI-Trader Core Api",
    version="0.1.0",
    description="AI-powered trading platform with sentiment analysis and market prediction"
)

app.include_router(predict_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}



# Example celery tasks
class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


@app.get("/celery/health")
def celery_health() -> Dict[str, Any]:
    """Check Celery worker health"""
    try:
        # Send a health check task
        task = health_check.delay()
        result = task.get(timeout=10)
        return {
            "status": "healthy",
            "celery_status": "connected",
            "task_result": result
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "celery_status": "disconnected",
            "error": str(e)
        }


@app.post("/tasks/sentiment-analysis")
def create_sentiment_task(text: str) -> TaskResponse:
    """Create a sentiment analysis task"""
    task = sentiment_analysis_task.delay(text)
    return TaskResponse(
        task_id=task.id,
        status="pending",
        message="Sentiment analysis task created"
    )


@app.get("/tasks/{task_id}")
def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get task status and result"""
    task = celery_app.AsyncResult(task_id)
    
    if task.state == "PENDING":
        response = {
            "task_id": task_id,
            "state": task.state,
            "status": "Task is waiting to be processed"
        }
    elif task.state == "SUCCESS":
        response = {
            "task_id": task_id,
            "state": task.state,
            "result": task.result
        }
    else:
        response = {
            "task_id": task_id,
            "state": task.state,
            "status": str(task.info)
        }
    
    return response
