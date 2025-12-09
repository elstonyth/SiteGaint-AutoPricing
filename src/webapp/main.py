"""
FastAPI application entry point for SiteGiant Pricing Automation.

Run with:
    uvicorn src.webapp.main:app --reload

Open: http://127.0.0.1:8000
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.utils.config_loader import load_config, load_env
from src.utils.logging_config import setup_logging
from src.webapp.middleware import RateLimitConfig, RateLimitMiddleware
from src.webapp.routes import cleanup_expired_sessions, router

logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    load_env()
    config = load_config()

    # Setup structured logging
    log_format = getattr(config.logging, "format", "text") if hasattr(config, "logging") else "text"
    log_level = getattr(config.logging, "level", "INFO") if hasattr(config, "logging") else "INFO"
    setup_logging(level=log_level, log_format=log_format)

    logger.info("SiteGiant Pricing Automation - Web App starting...")
    print("\n" + "=" * 60)
    print("SiteGiant Pricing Automation - Web App")
    print("=" * 60)
    print("Server running at: http://127.0.0.1:8000")
    print("=" * 60 + "\n")
    yield
    logger.info("SiteGiant Pricing Automation - Web App shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="SiteGiant Pricing Automation",
    description="Local web application for automating price updates from Pokedata to SiteGiant",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    """Global error handling middleware."""
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.exception(f"Unhandled error processing {request.url.path}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(e) if app.debug else "An unexpected error occurred",
                "path": str(request.url.path),
            },
            headers={"X-Process-Time": str(process_time)},
        )


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Detailed health check endpoint for monitoring."""
    from src.services.health_service import get_health_service

    cleanup_expired_sessions()
    health_service = get_health_service()
    report = health_service.get_full_health()
    return report.to_dict()


@app.get("/health/simple")
async def simple_health_check() -> dict[str, Any]:
    """Simple health check for load balancers."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/health/ready")
async def readiness_check() -> dict[str, Any]:
    """Readiness check - verifies app can serve requests."""
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
    }


# Add rate limiting middleware
rate_limit_config = RateLimitConfig(
    enabled=True,
    process_rpm=10,  # 10 price processing requests per minute
    api_rpm=30,  # 30 API calls per minute
    export_rpm=20,  # 20 export requests per minute
)
app.add_middleware(RateLimitMiddleware, config=rate_limit_config)

# Mount static files
static_dir = PROJECT_ROOT / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include routes
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.webapp.main:app", host="127.0.0.1", port=8000, reload=True)
