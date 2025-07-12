import pytest 
from itertools import product

from ai_trader.nlp.utils.tokenizer import Tokenizer
from ai_trader.nlp.models import *
from ai_trader.nlp.sentiment_analyzer import Sentiment_analyzer

@pytest.fixture
def tokenizer():
    return Tokenizer()

WORD_TRANSFORMS = ["ibe", "bow", "word2vec"]
MODELS = ["decision_tree_regression", "knn_regression", "linear_regression", "nn"]

@pytest.fixture(
    params = list(product(WORD_TRANSFORMS, MODELS)), 
    ids=[f"{wt}_{md}" for wt, md in product(WORD_TRANSFORMS, MODELS)]
)
def sentiment_analyzer(request):
    wt, md = request.param
    return Sentiment_analyzer(wt, md)
    
@pytest.fixture
def custom_X():
    return [
        "The company's stock price surged after the announcement of record-breaking quarterly earnings.",
        "Investors are worried about the rising interest rates and slowing economic growth.",
        "The new product launch has generated strong demand and boosted revenue forecasts.",
        "Analysts downgraded the stock citing disappointing guidance for the next fiscal year.",
        "Positive cash flow and a healthy balance sheet indicate financial stability.",
        "The merger talks stalled amid regulatory concerns and created uncertainty in the market.",
        "Share buyback program demonstrates management's confidence in the company's valuation.",
        "The unexpected loss in the derivatives portfolio resulted in significant write-downs.",
        "Strong insider buying by executives signals potential undervaluation of shares.",
        "Weak consumer spending caused revenue to fall short of Wall Street expectations.",
        "The dividend increase reflects consistent profitability and commitment to shareholders.",
        "Ongoing litigation and potential fines weigh heavily on the company's outlook.",
        "Robust retail sales during the holiday season lifted retail sector stocks.",
        "High volatility in commodity prices is raising concerns among investors.",
        "Successful debt refinancing at lower rates reduced interest expenses significantly.",
        "Supply chain disruptions have eroded profit margins and hurt overall profitability.",
        "The bullish technical indicators suggest the market is gearing up for a rally.",
        "Bearish market sentiment is spreading as global tensions escalate.",
        "Strong order backlog and expanding market share point to future revenue growth.",
        "Mounting debt levels and weak free cash flow raise red flags for investors."
    ]

@pytest.fixture
def custom_y():
    return [
        0.95,
        0.20,
        0.90,
        0.15,
        0.88,
        0.25,
        0.85,
        0.10,
        0.80,
        0.30,
        0.92,
        0.18,
        0.87,
        0.35,
        0.89,
        0.22,
        0.83,
        0.28,
        0.91,
        0.12
    ]
    
@pytest.fixture
def sample_sequences():
    return [
        "The stock market is volatile.",
        "Investors are optimistic about the future.",
        "Economic indicators suggest growth.",
        "The company reported strong earnings.",
        "Market sentiment is bearish.",
        "Analysts predict a recession.",
        "The tech sector is booming.",
        "Inflation rates are rising.",
        "Interest rates are expected to increase.",
        "Global markets are interconnected."
    ]

@pytest.fixture
def sample_text():
    return "The stock market is experiencing significant fluctuations due to geopolitical tensions and economic uncertainty. Investors are closely monitoring the situation as it develops, with many analysts predicting a potential downturn in the near future. The tech sector, however, continues to show resilience, with several companies reporting strong quarterly earnings despite the overall market volatility."