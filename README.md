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

After you’ve cloned the repo, you need to:

1. **Install Poetry (if you don’t already have it).**

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

   Make sure `poetry` is on your PATH (e.g. `export PATH="$HOME/.local/bin:$PATH"`).

2. **Change into your project directory** (where `pyproject.toml` lives):

   ```bash
   cd ./AI-Trader
   ```

3. **Install all dependencies via Poetry**:

   ```bash
   poetry install
   ```

   This will:

   * Create (or reuse) a virtual environment under Poetry’s control.
   * Read `pyproject.toml` and `poetry.lock` and install exactly the versions you specified (e.g. NumPy, etc.).

4. **Activate the virtualenv shell**

   ```bash
   poetry env activate
   ```

   * That should return the command to activate virtual env.
   * e.g. `source /Users/some_user/Library/Caches/pypoetry/virtualenvs/facexpr-Wp4LdDR0-py3.13/bin/activate`
   * Run that command.

5. **Verify that your entry-point works**. For example:

   ```bash
   poetry run python src/ai_trader/analysis/test.py
   ```

   You should see the the analysis printed in console.

6. **Install exchange_calendars**:

   ```bash
   poetry run pip install --ignore-requires-python exchange_calendars
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

See [sentiment/README.md](nlp/README.md) for details on:

- Text preprocessing (stopword removal, stemming)
- Tokenization and encoding (BOW, embeddings, BERT)
- Model training and evaluation (Decision Trees, KNN, Neural Networks)

## Simulation Engine

The core simulation logic lives in `simulate/simulate.py`. It:

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