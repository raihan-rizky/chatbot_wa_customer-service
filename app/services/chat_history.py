"""Supabase chat history service — persistent message storage."""

from __future__ import annotations

import logging

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

TABLE = "chat_messages_teladan"


def _headers() -> dict[str, str]:
    """Build Supabase REST API headers using service_role key."""
    settings = get_settings()
    return {
        "apikey": settings.supabase_service_key,
        "Authorization": f"Bearer {settings.supabase_service_key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def _base_url() -> str:
    settings = get_settings()
    return f"{settings.supabase_url}/rest/v1/{TABLE}"


async def save_message(
    phone: str,
    role: str,
    content: str,
    image_url: str | None = None,
) -> None:
    """Save a single message to Supabase.

    Args:
        phone: User phone number.
        role: 'user' or 'assistant'.
        content: Message text.
        image_url: Optional WhatsApp media ID or description for image messages.
    """
    payload: dict = {"phone": phone, "role": role, "content": content}
    if image_url:
        payload["image_url"] = image_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            _base_url(),
            headers=_headers(),
            json=payload,
        )
        if resp.status_code >= 400:
            logger.error("Supabase save failed: %s %s", resp.status_code, resp.text)
        else:
            logger.info("Saved %s message for %s", role, phone)


async def get_history(phone: str, limit: int = 20) -> list[dict]:
    """Fetch recent chat history for a phone number.

    Args:
        phone: User phone number.
        limit: Max messages to retrieve (newest last).

    Returns:
        List of dicts with keys: role, content, image_url, created_at.
    """
    params = {
        "phone": f"eq.{phone}",
        "select": "role,content,image_url,created_at",
        "order": "created_at.desc",
        "limit": str(limit),
    }
    headers = _headers()
    headers["Prefer"] = "return=representation"

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(_base_url(), headers=headers, params=params)
        if resp.status_code >= 400:
            logger.error("Supabase fetch failed: %s %s", resp.status_code, resp.text)
            return []

        messages = resp.json()
        messages.reverse()
        return messages


async def clear_history(phone: str) -> None:
    """Delete all messages for a phone number."""
    params = {"phone": f"eq.{phone}"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.delete(_base_url(), headers=_headers(), params=params)
        if resp.status_code >= 400:
            logger.error("Supabase delete failed: %s %s", resp.status_code, resp.text)
        else:
            logger.info("Cleared history for %s", phone)
