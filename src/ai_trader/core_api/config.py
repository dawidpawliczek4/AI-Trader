from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Redis settings
    redis_url: str = "redis://localhost:6379/0"
    
    # Celery settings
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
