import torch
from functools import lru_cache
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

@lru_cache(maxsize=1)
def _load_model():
    """Load FinBERT model and tokenizer for financial sentiment analysis"""
    model_name = "ProsusAI/finbert"
    model = BertForSequenceClassification.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def predict_sentiment(headline: str):
    """
    Predict sentiment of financial text using FinBERT
    
    Args:
        headline (str): Financial text/headline to analyze
        
    Returns:
        dict: Contains sentiment prediction and confidence scores
    """
    model, tokenizer = _load_model()
    
    inputs = tokenizer(headline, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = F.softmax(outputs.logits, dim=-1)
        
    sentiment_labels = ["negative", "neutral", "positive"]
    predicted_class = torch.argmax(predictions, dim=-1).item()
    confidence = predictions[0][predicted_class].item()
    
    return {
        "sentiment": sentiment_labels[predicted_class],
        "confidence": confidence,
        "scores": {
            "negative": predictions[0][0].item(),
            "neutral": predictions[0][1].item(),
            "positive": predictions[0][2].item()
        }
    }