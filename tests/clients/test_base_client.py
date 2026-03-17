"""Tests for BaseClient — retry, timeout, rate limiting."""

import asyncio

import httpx
import pytest

from src.clients.base_client import BaseClient


@pytest.mark.asyncio
async def test_base_client_get():
    """Test basic GET against a known endpoint."""
    client = BaseClient(base_url="https://httpbin.org")
    result = await client.get("/get", params={"test": "1"})
    assert result["args"]["test"] == "1"
    await client.close()


@pytest.mark.asyncio
async def test_base_client_post():
    """Test basic POST against a known endpoint."""
    client = BaseClient(base_url="https://httpbin.org")
    result = await client.post("/post", json_body={"key": "value"})
    assert result["json"]["key"] == "value"
    await client.close()


@pytest.mark.asyncio
async def test_base_client_retry_on_5xx(monkeypatch):
    """Verify retry logic fires on 5xx responses."""
    call_count = 0
    original_request = httpx.AsyncClient.request

    async def mock_request(self, method, url, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return httpx.Response(status_code=503, request=httpx.Request(method, url))
        return httpx.Response(
            status_code=200,
            json={"ok": True},
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "request", mock_request)

    client = BaseClient(base_url="https://example.com", rate_limit_delay=0.0)
    result = await client.get("/test")
    assert result == {"ok": True}
    assert call_count == 3
    await client.close()


@pytest.mark.asyncio
async def test_base_client_timeout():
    """Verify timeout raises after retries exhausted."""
    client = BaseClient(
        base_url="https://httpbin.org",
        timeout=0.001,
        max_retries=0,
        rate_limit_delay=0.0,
    )
    with pytest.raises(asyncio.TimeoutError):
        await client.get("/delay/10")
    await client.close()
