from celery import Celery
from ai_trader.core_api.config import get_settings

settings = get_settings()

celery_app = Celery(
    "ai_trader",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "ai_trader.core_api.tasks",
        "ai_trader.core_api.sentiment.services.scrap",
        "ai_trader.core_api.sentiment.services"
    ]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC", 
    enable_utc=True,
    task_track_started=True,
    task_routes={
        "ai_trader.core_api.tasks.*": {"queue": "default"},
    },    
    beat_schedule={
        "predict-bankier-every-30s": {
            "task": "ai_trader.core_api.sentiment.services.scrap.predict_bankier",
            "schedule": 5.0,
        }
    }
)

celery_app.autodiscover_tasks(["ai_trader.core_api"])
