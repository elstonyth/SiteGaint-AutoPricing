"""
FastAPI middleware for rate limiting and request handling.

Provides protection against excessive requests and abuse.
"""

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    enabled: bool = True
    # Default limits (requests per minute)
    default_rpm: int = 60
    # Endpoint-specific limits
    process_rpm: int = 10
    api_rpm: int = 30
    export_rpm: int = 20
    # Burst allowance (extra requests allowed in short bursts)
    burst_multiplier: float = 1.5
    # Window size in seconds
    window_seconds: int = 60


class RateLimitState:
    """Tracks request counts per client."""

    def __init__(self):
        # Dict of client_id -> Dict of endpoint -> list of timestamps
        self.requests: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    def record_request(self, client_id: str, endpoint: str) -> None:
        """Record a request from a client."""
        now = time.time()
        self.requests[client_id][endpoint].append(now)

    def get_request_count(self, client_id: str, endpoint: str, window_seconds: int) -> int:
        """Get the number of requests in the time window."""
        now = time.time()
        cutoff = now - window_seconds

        # Clean old entries and count
        timestamps = self.requests[client_id][endpoint]
        valid_timestamps = [t for t in timestamps if t > cutoff]
        self.requests[client_id][endpoint] = valid_timestamps

        return len(valid_timestamps)

    def cleanup(self, max_age_seconds: int = 300) -> None:
        """Remove old entries to prevent memory growth."""
        now = time.time()
        cutoff = now - max_age_seconds

        for client_id in list(self.requests.keys()):
            for endpoint in list(self.requests[client_id].keys()):
                self.requests[client_id][endpoint] = [
                    t for t in self.requests[client_id][endpoint] if t > cutoff
                ]
                if not self.requests[client_id][endpoint]:
                    del self.requests[client_id][endpoint]
            if not self.requests[client_id]:
                del self.requests[client_id]


# Global rate limit state
_rate_limit_state = RateLimitState()


def get_client_id(request: Request) -> str:
    """Extract client identifier from request."""
    # Try X-Forwarded-For first (for proxied requests)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    # Fall back to client host
    if request.client:
        return request.client.host

    return "unknown"


def get_endpoint_limit(path: str, config: RateLimitConfig) -> int:
    """Get the rate limit for a specific endpoint."""
    if "/process" in path:
        return config.process_rpm
    elif "/export" in path:
        return config.export_rpm
    elif "/api/" in path or "/fetch-fx" in path:
        return config.api_rpm
    else:
        return config.default_rpm


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests."""

    def __init__(self, app, config: RateLimitConfig | None = None):
        super().__init__(app)
        self.config = config or RateLimitConfig()
        self.state = _rate_limit_state
        self._last_cleanup = time.time()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        if not self.config.enabled:
            return await call_next(request)

        # Skip rate limiting for health checks and static files
        path = request.url.path
        if path in ("/health", "/health/simple") or path.startswith("/static"):
            return await call_next(request)

        # Periodic cleanup
        if time.time() - self._last_cleanup > 60:
            self.state.cleanup()
            self._last_cleanup = time.time()

        client_id = get_client_id(request)
        limit = get_endpoint_limit(path, self.config)

        # Check rate limit
        current_count = self.state.get_request_count(client_id, path, self.config.window_seconds)

        if current_count >= limit:
            logger.warning(
                f"Rate limit exceeded: client={client_id}, path={path}, "
                f"count={current_count}, limit={limit}"
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too many requests",
                    "message": f"Rate limit exceeded. Maximum {limit} requests per minute.",
                    "retry_after_seconds": self.config.window_seconds,
                },
                headers={
                    "Retry-After": str(self.config.window_seconds),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                },
            )

        # Record this request
        self.state.record_request(client_id, path)

        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, limit - current_count - 1))

        return response


def create_rate_limit_middleware(
    enabled: bool = True,
    process_rpm: int = 10,
    api_rpm: int = 30,
) -> RateLimitMiddleware:
    """Create a configured rate limit middleware."""
    config = RateLimitConfig(
        enabled=enabled,
        process_rpm=process_rpm,
        api_rpm=api_rpm,
    )
    return lambda app: RateLimitMiddleware(app, config)
