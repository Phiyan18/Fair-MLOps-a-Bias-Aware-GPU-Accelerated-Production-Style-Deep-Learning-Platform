"""Optional API-key authentication and a simple in-memory rate limiter (both read env vars per request)."""
from __future__ import annotations

import os
import secrets
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request


def _keys() -> list[str]:
    raw = os.getenv("API_KEYS") or os.getenv("API_KEY") or ""
    return [k.strip() for k in raw.split(",") if k.strip()]


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """If API_KEY(S) is set, every protected endpoint needs a matching `X-API-Key` header."""
    keys = _keys()
    if not keys:
        return
    if not x_api_key or not any(secrets.compare_digest(x_api_key.encode(), k.encode()) for k in keys):
        raise HTTPException(status_code=401, detail="Invalid or missing API key",
                            headers={"WWW-Authenticate": "ApiKey"})


class _SlidingWindow:
    def __init__(self) -> None:
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str, limit: int, window: float = 60.0) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits[client]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            if len(self._hits) > 10_000:  # bound memory
                for k in [k for k, v in self._hits.items() if not v]:
                    del self._hits[k]
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = _SlidingWindow()


def rate_limit(request: Request) -> None:
    """RATE_LIMIT_PER_MIN=0 (default) disables it. Per client IP and per pod (behind a proxy use the proxy's limiter)."""
    limit = int(os.getenv("RATE_LIMIT_PER_MIN", "0") or 0)
    if limit <= 0:
        return
    client = request.client.host if request.client else "unknown"
    if not limiter.allow(client, limit):
        raise HTTPException(status_code=429, detail="Rate limit exceeded", headers={"Retry-After": "60"})
