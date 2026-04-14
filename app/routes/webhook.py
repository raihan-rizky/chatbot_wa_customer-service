"""WhatsApp webhook routes — incoming messages (WAHA format)."""

from __future__ import annotations

import asyncio
import logging
import traceback

from fastapi import APIRouter, Request

from app.config import get_settings
from app.services.llm_service import get_ai_response
from app.services.chat_history import save_message
from app.services.image_service import analyze_image, download_wa_media
from app.services.whatsapp import send_message

logger = logging.getLogger(__name__)

router = APIRouter()

# Track processed message IDs to avoid duplicates
_processed_ids: set[str] = set()

# ── Incoming messages ────────────────────────────────────────────
@router.post("/webhook")
async def receive_message(request: Request):
    """Receive incoming WhatsApp messages (WAHA format) and process replies."""
    try:
        body = await request.json()
    except Exception:
        return {"status": "ok"}

    event = body.get("event")
    if event != "message":
        return {"status": "ok"}

    payload = body.get("payload", {})
    if not payload:
        return {"status": "ok"}

    msg_id = payload.get("id", "")
    sender_jid = payload.get("from", "")

    # Ignore group messages and status broadcasts
    if "@g.us" in sender_jid or "status@broadcast" in sender_jid:
        return {"status": "ok"}

    # Ignore messages sent by the bot itself
    if payload.get("fromMe", False):
        return {"status": "ok"}

    sender = sender_jid.replace("@c.us", "")

    # Deduplicate
    if msg_id in _processed_ids:
        logger.info("Skipping duplicate message %s", msg_id)
        return {"status": "ok"}
    _processed_ids.add(msg_id)
    if len(_processed_ids) > 1000:
        _processed_ids.clear()

    msg_type = payload.get("type", "chat")
    has_media = payload.get("hasMedia", False)

    logger.info("Webhook from %s type=%s has_media=%s id=%s", sender, msg_type, has_media, msg_id)

    try:
        if has_media or msg_type == "image":
            await _handle_single_image(sender, payload)
        elif msg_type == "chat":
            text = payload.get("body", "")
            if text:
                await _handle_text(sender, text)
        else:
            logger.info("Skipping unsupported message type: %s", msg_type)
    except Exception:
        logger.error("Error processing webhook:\n%s", traceback.format_exc())

    return {"status": "ok"}


async def _handle_text(phone: str, text: str) -> None:
    """Handle a text message — generate AI reply and save to Supabase."""
    logger.info("Text from %s: %s", phone, text[:80])

    try:
        # LLM will handle everything naturally based on its prompt
        reply = await get_ai_response(phone, text)
        logger.info("AI reply ready, sending to %s", phone)
        await send_message(phone, reply)
        logger.info("Reply sent to %s", phone)
    except Exception:
        logger.error("Failed to reply to %s:\n%s", phone, traceback.format_exc())
        try:
            await send_message(phone, "Maaf, terjadi kesalahan. Coba kirim ulang pesan kamu. 🙏")
        except Exception:
            pass


async def _handle_single_image(phone: str, payload: dict) -> None:
    """Handle a single image message (WAHA) — download, analyze design, and reply."""
    msg_id = payload.get("id")
    # In WAHA, caption is often stored in 'body' for media messages.
    caption = payload.get("body", "")

    logger.info("Media from %s (msg_id=%s)", phone, msg_id)

    try:
        # Save user image message to Supabase
        user_content = caption if caption else "[Gambar dikirim]"
        await save_message(phone, "user", user_content, image_url=f"wa_media:{msg_id}")

        # Download image from WAHA API
        image_bytes = await download_wa_media(msg_id)
        if not image_bytes:
            logger.error("Failed to download image %s", msg_id)
            await send_message(phone, "Maaf, gagal mengunduh gambar ini. Coba kirim ulang.")
            return

        logger.info("Downloaded image: %d bytes", len(image_bytes))

        # Analyze with vision model
        result = await analyze_image(image_bytes, caption)
        logger.info("Analysis done for %s", phone)

        # Save AI response to Supabase
        await save_message(phone, "assistant", result)

        # Send text reply
        await send_message(phone, result)
        logger.info("Image analysis sent to %s", phone)

    except Exception:
        logger.error("Failed to process media from %s:\n%s", phone, traceback.format_exc())
        try:
            await send_message(phone, "Maaf, gagal memproses gambar. Coba kirim ulang. 🙏")
        except Exception:
            pass
