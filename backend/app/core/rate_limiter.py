"""Rate limiting middleware using Redis sliding window."""

from __future__ import annotations

import time
import logging
from typing import Optional

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter backed by Redis.
    Falls back to in-memory tracking when Redis is unavailable.
    """

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.rpm = requests_per_minute
        self._memory_store: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks and static files
        if request.url.path in ("/health", "/docs", "/redoc", "/openapi.json"):
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        window_key = f"rate_limit:{client_ip}"

        is_allowed = await self._check_rate_limit(window_key)
        if not is_allowed:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later.",
            )

        return await call_next(request)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, respecting X-Forwarded-For behind proxies."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def _check_rate_limit(self, key: str) -> bool:
        """Check rate limit using Redis or fallback to memory."""
        try:
            from app.core.redis_client import get_redis
            redis = await get_redis()
            return await self._check_redis(redis, key)
        except Exception:
            return self._check_memory(key)

    async def _check_redis(self, redis, key: str) -> bool:
        """Redis sliding window counter."""
        now = time.time()
        window_start = now - 60

        pipe = redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, 120)
        results = await pipe.execute()

        current_count = results[2]
        return current_count <= self.rpm

    def _check_memory(self, key: str) -> bool:
        """In-memory fallback when Redis is unavailable."""
        now = time.time()
        window_start = now - 60

        if key not in self._memory_store:
            self._memory_store[key] = []

        self._memory_store[key] = [
            ts for ts in self._memory_store[key] if ts > window_start
        ]
        self._memory_store[key].append(now)

        return len(self._memory_store[key]) <= self.rpm
