"""WhatsApp API: send messages and contact actions via WAHA."""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings
from app.services.http_client import get_waha_client

logger = logging.getLogger(__name__)


def _get_headers(settings: Any) -> dict[str, str]:
    """Return WAHA request headers."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if settings.waha_api_key:
        headers["X-Api-Key"] = settings.waha_api_key
    return headers


async def send_message(to: str, body: str) -> None:
    """Send a text message to a WhatsApp user via WAHA."""
    settings = get_settings()
    url = f"{settings.waha_base_url}/api/sendText"
    chat_id = to if "@" in to else f"{to}@c.us"
    payload = {
        "session": settings.waha_session,
        "chatId": chat_id,
        "text": body,
    }

    client = get_waha_client()
    response = await client.post(url, headers=_get_headers(settings), json=payload)

    if response.status_code not in (200, 201):
        logger.error(
            "Failed to send WA message to %s - %s %s",
            to,
            response.status_code,
            response.text,
        )
        response.raise_for_status()

    logger.info("Message sent to %s", to)


async def block_contact(contact_id: str) -> None:
    """Block a WhatsApp contact via WAHA."""
    settings = get_settings()
    url = f"{settings.waha_base_url}/api/contacts/block"
    payload = {
        "contactId": contact_id if "@" in contact_id else f"{contact_id}@c.us",
        "session": settings.waha_session,
    }

    client = get_waha_client()
    response = await client.post(url, headers=_get_headers(settings), json=payload)

    if response.status_code not in (200, 201):
        logger.error(
            "Failed to block WA contact %s - %s %s",
            contact_id,
            response.status_code,
            response.text,
        )
        response.raise_for_status()

    logger.info("Blocked contact %s", contact_id)
