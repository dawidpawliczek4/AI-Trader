import uvicorn
from ai_trader.core_api.config import get_settings

def main():
    """Start the FastAPI server"""
    settings = get_settings()
    uvicorn.run(
        "ai_trader.core_api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )

if __name__ == "__main__":
    main()
