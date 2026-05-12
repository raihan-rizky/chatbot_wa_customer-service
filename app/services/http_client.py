"""Shared outbound HTTP clients for connection reuse."""

from __future__ import annotations

import httpx

_supabase_client: httpx.AsyncClient | None = None
_waha_client: httpx.AsyncClient | None = None


def get_supabase_client() -> httpx.AsyncClient:
    """Return a shared Supabase HTTP client."""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = httpx.AsyncClient(
            timeout=10.0,
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )
    return _supabase_client


def get_waha_client() -> httpx.AsyncClient:
    """Return a shared WAHA/media HTTP client."""
    global _waha_client
    if _waha_client is None:
        _waha_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=30, max_keepalive_connections=10),
        )
    return _waha_client


async def close_http_clients() -> None:
    """Close shared HTTP clients during app shutdown."""
    global _supabase_client, _waha_client

    if _supabase_client is not None:
        await _supabase_client.aclose()
        _supabase_client = None

    if _waha_client is not None:
        await _waha_client.aclose()
        _waha_client = None
