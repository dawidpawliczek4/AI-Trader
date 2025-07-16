from celery import shared_task
import json, redis, hashlib, logging
from ai_trader.core_api.sentiment.services.predict import predict_sentiment

r = redis.Redis(host="redis", decode_responses=True)
logger = logging.getLogger(__name__)

def _make_id(url:str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:24]

def fetch_bankier():
    return [
        {"headline": "hello world", "url": "https://example.com/1", "published": "2025-07-16T10:00:00"},
        {"headline": "hello 2", "url": "https://example.com/2", "published": "2025-07-16T10:05:00"}
    ]

@shared_task(name="ai_trader.core_api.sentiment.services.scrap.predict_bankier")
def predict_bankier():
    try:
        items = list(fetch_bankier())
        if not items:
            return
        
        # scores = predict_sentiment([it["headline"] for it in items])
        scores = [1,2]

        rows = []
        for it, sc in zip(items, scores):
            payload = {
                "id": _make_id(it["url"]),
                "headline": it["headline"],
                "score": sc,
                "published": it["published"],
                "url": it["url"],
            }
            r.publish("sentiment", json.dumps(payload))
            rows.append(payload)
        
        if rows:
            logger.info("heres rows", rows)
    except Exception as e:
        logger.exception("predict bankier fail", e)


    
