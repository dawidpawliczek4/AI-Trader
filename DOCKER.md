# Docker Compose for AI-Trader

This setup includes:
- **Redis**: Message broker and result backend for Celery
- **FastAPI (Uvicorn)**: Main API server
- **Celery Worker**: Background task processing
- **Celery Beat**: Scheduled task execution
- **Flower**: Celery monitoring dashboard

## Quick Start

1. **Build and start all services:**
   ```bash
   docker-compose up --build
   ```

2. **Start services in background:**
   ```bash
   docker-compose up -d --build
   ```

3. **View logs:**
   ```bash
   docker-compose logs -f
   ```

4. **Stop all services:**
   ```bash
   docker-compose down
   ```

## Services

### API Server (Uvicorn)
- **URL**: http://localhost:8000
- **Health Check**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs
- **Celery Health**: http://localhost:8000/celery/health

### Flower (Celery Monitoring)
- **URL**: http://localhost:5555
- Monitor Celery workers, tasks, and queues

### Redis
- **Port**: 6379
- Message broker and result backend

## Available Endpoints

### Task Management
- `POST /tasks/sentiment-analysis` - Create sentiment analysis task
- `GET /tasks/{task_id}` - Get task status and result
- `GET /celery/health` - Check Celery worker health

## Environment Variables

Copy `.env.example` to `.env` for local development:
```bash
cp .env.example .env
```

## Development

### Run individual services:

**API only:**
```bash
docker-compose up api
```

**Celery worker only:**
```bash
docker-compose up celery-worker
```

**Redis only:**
```bash
docker-compose up redis
```

### Access container shells:
```bash
# API container
docker-compose exec api bash

# Celery worker container
docker-compose exec celery-worker bash
```

### View service logs:
```bash
# All services
docker-compose logs

# Specific service
docker-compose logs api
docker-compose logs celery-worker
docker-compose logs redis
```

## Scaling

Scale Celery workers:
```bash
docker-compose up --scale celery-worker=3
```

## Production Considerations

1. **Environment Variables**: Use proper production values
2. **Volumes**: Consider persistent storage for Redis data
3. **Security**: Configure Redis authentication
4. **Monitoring**: Set up proper logging and monitoring
5. **Load Balancing**: Use nginx or similar for API load balancing
