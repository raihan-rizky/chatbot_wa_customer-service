"""Web Push notification service backed by Supabase subscriptions."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from pywebpush import WebPushException, webpush

from app.config import get_settings

logger = logging.getLogger(__name__)

PUSH_TABLE = "push_subscriptions"
EVENT_TABLE = "waha_events"
NEGOTIATION_TABLE = "negotiations"

CLOSING_KEYWORDS = (
    "deal",
    "oke jadi",
    "ok jadi",
    "jadi ambil",
    "saya ambil",
    "ambil",
    "lanjut",
    "setuju",
    "gas",
    "order",
    "pesan",
    "booking",
    "bayar",
    "dp",
)


def _headers(prefer: str = "return=representation") -> dict[str, str]:
    settings = get_settings()
    return {
        "apikey": settings.supabase_service_key,
        "Authorization": f"Bearer {settings.supabase_service_key}",
        "Content-Type": "application/json",
        "Prefer": prefer,
    }


def _table_url(table: str) -> str:
    settings = get_settings()
    return f"{settings.supabase_url}/rest/v1/{table}"


def is_closing_text(text: str) -> bool:
    """Return True when a buyer message looks like a deal confirmation."""
    normalized = " ".join(text.lower().split())
    if not normalized:
        return False
    return any(keyword in normalized for keyword in CLOSING_KEYWORDS)


async def save_waha_event(event_name: str, chat_id: str, message_id: str, payload: dict[str, Any]) -> None:
    """Persist the raw WAHA webhook event for audit/replay."""
    if not get_settings().supabase_url or not get_settings().supabase_service_key:
        logger.warning("Supabase is not configured; skipping WAHA event save")
        return

    row = {
        "event_name": event_name,
        "chat_id": chat_id,
        "message_id": message_id,
        "payload": payload,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(_table_url(EVENT_TABLE), headers=_headers("return=minimal"), json=row)
            if resp.status_code >= 400:
                logger.error("Failed to save WAHA event: %s %s", resp.status_code, resp.text)
    except Exception as exc:
        logger.exception("Failed to save WAHA event: %s", exc)


async def mark_negotiation_closed(chat_id: str, customer_name: str | None = None) -> None:
    """Upsert a negotiation row as closed."""
    row: dict[str, Any] = {
        "chat_id": chat_id,
        "status": "closed",
        "closed_at": datetime.now(timezone.utc).isoformat(),
    }
    if customer_name:
        row["customer_name"] = customer_name

    params = {"on_conflict": "chat_id"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                _table_url(NEGOTIATION_TABLE),
                headers=_headers("resolution=merge-duplicates,return=minimal"),
                params=params,
                json=row,
            )
            if resp.status_code >= 400:
                logger.error("Failed to mark negotiation closed: %s %s", resp.status_code, resp.text)
    except Exception as exc:
        logger.exception("Failed to mark negotiation closed: %s", exc)


async def notify_closing(chat_id: str, phone: str, text: str) -> None:
    """Send closing notification to all stored browser subscriptions."""
    settings = get_settings()
    if not settings.vapid_private_key:
        logger.warning("VAPID_PRIVATE_KEY is not configured; skipping Web Push")
        return

    subscriptions = await _fetch_subscriptions()
    if not subscriptions:
        logger.info("No push subscriptions found; closing notification skipped")
        return

    payload = json.dumps(
        {
            "title": "Closing berhasil",
            "body": f"Pembeli {phone} setuju/deal. Cek chat untuk lanjut proses.",
            "url": f"/dashboard/deals?chat_id={chat_id}",
            "chat_id": chat_id,
            "phone": phone,
            "message": text[:200],
        }
    )

    await asyncio.gather(*[_send_push(sub, payload) for sub in subscriptions])


async def notify_closing_deal(
    *,
    chat_id: str,
    customer_name: str,
    amount: float | int | None,
    message: str,
    url: str,
) -> None:
    """Send a POS-originated closing deal notification to browser subscribers."""
    settings = get_settings()
    if not settings.vapid_private_key:
        logger.warning("VAPID_PRIVATE_KEY is not configured; skipping Web Push")
        return

    subscriptions = await _fetch_subscriptions()
    if not subscriptions:
        logger.info("No push subscriptions found; closing deal notification skipped")
        return

    body = message or f"{customer_name} closing deal."
    payload = json.dumps(
        {
            "title": "Closing deal",
            "body": body,
            "url": url,
            "chat_id": chat_id,
            "customerName": customer_name,
            "amount": amount,
            "message": message[:200],
        }
    )

    await asyncio.gather(*[_send_push(sub, payload) for sub in subscriptions])


async def _fetch_subscriptions() -> list[dict[str, Any]]:
    params = {"select": "id,endpoint,p256dh,auth"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(_table_url(PUSH_TABLE), headers=_headers(), params=params)
            if resp.status_code >= 400:
                logger.error("Failed to fetch push subscriptions: %s %s", resp.status_code, resp.text)
                return []
            return resp.json()
    except Exception as exc:
        logger.exception("Failed to fetch push subscriptions: %s", exc)
        return []


async def _send_push(subscription: dict[str, Any], payload: str) -> None:
    settings = get_settings()
    subscription_info = {
        "endpoint": subscription["endpoint"],
        "keys": {
            "p256dh": subscription["p256dh"],
            "auth": subscription["auth"],
        },
    }

    try:
        await asyncio.to_thread(
            webpush,
            subscription_info=subscription_info,
            data=payload,
            vapid_private_key=settings.vapid_private_key,
            vapid_claims={"sub": settings.vapid_claims_subject},
        )
    except WebPushException as exc:
        logger.warning("Web Push failed for subscription %s: %s", subscription.get("id"), exc)
        if getattr(exc.response, "status_code", None) in {404, 410}:
            await _delete_subscription(subscription["endpoint"])
    except Exception as exc:
        logger.exception("Unexpected Web Push error: %s", exc)


async def _delete_subscription(endpoint: str) -> None:
    params = {"endpoint": f"eq.{endpoint}"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.delete(_table_url(PUSH_TABLE), headers=_headers("return=minimal"), params=params)
            if resp.status_code >= 400:
                logger.error("Failed to delete expired push subscription: %s %s", resp.status_code, resp.text)
    except Exception as exc:
        logger.exception("Failed to delete expired push subscription: %s", exc)
