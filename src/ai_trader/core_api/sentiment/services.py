from functools import lru_cache
from ai_trader.nlp.sentiment_analyzer import Sentiment_analyzer

@lru_cache(maxsize=1)
def _load_model():
    sentiment_analyzer = Sentiment_analyzer(
        word_transformation="bow",
        model="nn",
        load_path="./src/ai_trader/core_api/sentiment/model.pth"
    )
    sentiment_analyzer.fit()
    return sentiment_analyzer


def predict_sentiment(headline: list[str]):
    model = _load_model()
    return model.predict(headline)