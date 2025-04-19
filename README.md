# AI-Trader

**AI-Trader** is a collaborative machine learning project that implements a stock trading bot capable of predicting future price movements and executing trades under two modes:

- **Historical Simulation**: Backtest strategies on past market data to evaluate performance.
- **Live Trading**: Fetch real-time market and news data to drive trading decisions in a simulated or connected environment.

## Key Features

- **Dual-Mode Operation**: Switch between historical backtesting and live data-driven trading via a simple configuration flag.
- **Sentiment-Driven Predictions**: Leverage an NLP pipeline to analyze financial news headlines and derive sentiment scores that inform trade signals.
- **Economic Indicators**: Incorporate fundamental and technical indicators (e.g., moving averages, RSI) alongside sentiment for robust forecasting.
- **Simulation Engine**: A powerful `simulate` module that mimics exchange behavior, including stop‑loss orders and session-aware execution.
- **Modular Strategy Design**: Strategies are implemented as pluggable functions; define a new tactic in the `strategy/` folder, and the simulator will evaluate it.

## Repository Structure

```
AI-Trader/
├── data/                 # Stored market and news datasets
├── modules/              # Custom feature-engineering and indicator modules
├── nlp/                  # NLP sentiment analysis pipeline (see nlp/README.md)
├── strategy/             # Trading tactic implementations
├── test/                 # Unit and integration tests
├── utils/                # Utility scripts (simulate.py, data loaders)
├── main.py               # Entry point for running simulations or live trading
├── requirements.txt      # Python dependencies
└── README.md             # Project overview and instructions
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Hinski2/AI-Trader.git
   cd AI-Trader
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Copy the example environment file and populate your credentials:

```bash
cp .env.exapmle .env
# Then edit .env to add:
#   USERNAME and PASSWORD for TVDatafeed
#   Other API keys if needed
``` 

## NLP Sentiment Pipeline

See [nlp/README.md](nlp/README.md) for details on:

- Text preprocessing (stopword removal, stemming)
- Tokenization and encoding (BOW, embeddings, BERT)
- Model training and evaluation (Decision Trees, KNN, Neural Networks)

## Simulation Engine

The core simulation logic lives in `utils/simulate.py`. It:

- Streams bars or reads historical CSVs.
- Checks exchange hours via `exchange_calendars`.
- Feeds market and portfolio state into user-defined tactic functions.
- Applies buy/sell/hold decisions and stop‑loss logic.
- Returns final account balance and position size.

## Strategy Development

Strategies are simple Python callables receiving a dictionary:

```python
req = {
    "time": datetime,
    "symbol": "AAPL",
    "exchange": "NASDAQ",
    "account_balance": float,
    "shares": float,
    "stop_loss": float (optional)
}
res = {
    "action": "buy" | "sell" | "hold",
    "quantity": float,         # required if action != "hold"
    "set_stop_loss": float | None
}
```

Drop your strategy file in `strategy/`, and reference it when running `main.py`.

## Contributing

We welcome improvements to models, strategies, and the simulation framework. Please:

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/awesome-strategy`)
3. Commit changes (`git commit -m "Add awesome strategy"
4. Push (`git push origin feat/awesome-strategy`)
5. Open a pull request