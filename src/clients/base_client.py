"""Shared async HTTP client with retry, timeout, rate limiting, and structured logging."""

import asyncio
import time

import httpx
import structlog

from src.common.logging import get_logger

log = get_logger(__name__)


class BaseClient:
    """Async HTTP client with retry + rate limiting."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 15.0,
        max_retries: int = 3,
        rate_limit_delay: float = 0.1,
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._rate_limit_delay = rate_limit_delay
        self._client = httpx.AsyncClient(base_url=base_url)

    async def get(
        self,
        path: str,
        params: dict | None = None,
        headers: dict | None = None,
    ) -> dict:
        """GET with retry on 429/5xx. Returns parsed JSON."""
        return await self._request("GET", path, params=params, headers=headers)

    async def post(
        self,
        path: str,
        json_body: dict,
        headers: dict | None = None,
    ) -> dict:
        """POST with retry."""
        return await self._request("POST", path, json_body=json_body, headers=headers)

    async def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json_body: dict | None = None,
        headers: dict | None = None,
    ) -> dict:
        """Execute request with retry on 429/5xx and structured logging."""
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            t0 = time.monotonic()
            try:
                response = await asyncio.wait_for(
                    self._client.request(
                        method,
                        path,
                        params=params,
                        json=json_body,
                        headers=headers,
                    ),
                    timeout=self._timeout,
                )
                duration_ms = round((time.monotonic() - t0) * 1000, 1)

                log.info(
                    "http_request",
                    method=method,
                    path=path,
                    status=response.status_code,
                    duration_ms=duration_ms,
                    attempt=attempt + 1,
                )

                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < self._max_retries:
                        backoff = 2**attempt  # 1s, 2s, 4s
                        log.warning(
                            "http_retry",
                            method=method,
                            path=path,
                            status=response.status_code,
                            backoff_s=backoff,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    response.raise_for_status()

                response.raise_for_status()
                await asyncio.sleep(self._rate_limit_delay)
                return response.json()

            except asyncio.TimeoutError:
                duration_ms = round((time.monotonic() - t0) * 1000, 1)
                log.warning(
                    "http_timeout",
                    method=method,
                    path=path,
                    duration_ms=duration_ms,
                    attempt=attempt + 1,
                )
                last_exc = asyncio.TimeoutError(
                    f"{method} {path} timed out after {self._timeout}s"
                )
                if attempt < self._max_retries:
                    backoff = 2**attempt
                    await asyncio.sleep(backoff)
                    continue

            except httpx.HTTPStatusError as exc:
                log.error(
                    "http_error",
                    method=method,
                    path=path,
                    status=exc.response.status_code,
                    detail=exc.response.text[:200],
                )
                raise

        raise last_exc  # type: ignore[misc]

    async def close(self) -> None:
        """Close underlying httpx.AsyncClient."""
        await self._client.aclose()
