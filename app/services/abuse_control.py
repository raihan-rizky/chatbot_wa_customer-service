"""Sender abuse control backed by Supabase.

This stores repeated abuse signals and permanent block state so the bot can
ignore or block abusive numbers across restarts.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.config import get_settings
from app.services.http_client import get_supabase_client

logger = logging.getLogger(__name__)

TABLE = "chat_sender_defense_teladan"


def _headers() -> dict[str, str]:
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def get_sender_defense(phone: str) -> dict | None:
    """Fetch the stored abuse-defense row for a sender."""
    params = {
        "phone": f"eq.{phone}",
        "select": "phone,is_blocked,abuse_count,last_abuse_at,last_seen_at,blocked_at,block_reason",
        "limit": "1",
    }

    client = get_supabase_client()
    resp = await client.get(_base_url(), headers=_headers(), params=params)
    if resp.status_code >= 400:
        logger.error("Supabase defense fetch failed: %s %s", resp.status_code, resp.text)
        return None

    rows = resp.json()
    return rows[0] if rows else None


async def update_sender_defense(phone: str, fields: dict) -> None:
    """Update an existing sender-defense row, creating it if needed."""
    payload = {"phone": phone, **fields}
    existing = await get_sender_defense(phone)

    client = get_supabase_client()
    if existing:
        resp = await client.patch(
            _base_url(),
            headers=_headers(),
            params={"phone": f"eq.{phone}"},
            json=fields,
        )
    else:
        resp = await client.post(
            _base_url(),
            headers=_headers(),
            json=payload,
        )

    if resp.status_code >= 400:
        logger.error("Supabase defense update failed: %s %s", resp.status_code, resp.text)


async def record_abuse(phone: str, reason: str) -> dict:
    """Increment abuse counters for a sender and store the latest reason."""
    row = await get_sender_defense(phone) or {}
    abuse_count = int(row.get("abuse_count") or 0) + 1
    now = _now()
    fields = {
        "abuse_count": abuse_count,
        "last_abuse_at": now,
        "last_seen_at": now,
        "block_reason": reason,
    }
    await update_sender_defense(phone, fields)
    return {
        "phone": phone,
        "abuse_count": abuse_count,
        "last_abuse_at": now,
        "block_reason": reason,
        "is_blocked": bool(row.get("is_blocked")),
    }


async def mark_blocked(phone: str, reason: str) -> dict:
    """Persist a permanent ignore/block state for a sender."""
    row = await get_sender_defense(phone) or {}
    now = _now()
    fields = {
        "is_blocked": True,
        "blocked_at": now,
        "last_seen_at": now,
        "block_reason": reason,
    }
    if row.get("abuse_count") is not None:
        fields["abuse_count"] = int(row.get("abuse_count") or 0)
    await update_sender_defense(phone, fields)
    return {
        "phone": phone,
        "is_blocked": True,
        "blocked_at": now,
        "abuse_count": int(row.get("abuse_count") or 0),
        "block_reason": reason,
    }
