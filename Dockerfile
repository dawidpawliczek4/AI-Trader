FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VENV_IN_PROJECT=1
ENV POETRY_CACHE_DIR=/tmp/poetry_cache

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy poetry files and README (required by poetry)
COPY pyproject.toml poetry.lock README.md ./

# Configure poetry and install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --only=main --no-root && \
    rm -rf $POETRY_CACHE_DIR

# Copy application code
COPY . .

# Install the project itself
RUN poetry install --only-root

# Make sure the source code is available
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose port
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["poetry", "run", "uvicorn", "ai_trader.core_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
