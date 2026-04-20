"""WhatsApp webhook routes — incoming messages (WAHA format)."""

from __future__ import annotations

import asyncio
import logging
import traceback
import time

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

# Rate limiting state
RATE_LIMIT_MESSAGES = 5      # Max messages allowed
RATE_LIMIT_WINDOW = 60       # In seconds
_user_requests: dict[str, list[float]] = {}
_warned_users: set[str] = set()

def is_rate_limited(phone: str) -> bool:
    """Check if a phone number exceeds the allowed rate limit."""
    now = time.time()
    reqs = _user_requests.get(phone, [])
    reqs = [t for t in reqs if now - t < RATE_LIMIT_WINDOW]
    
    if len(reqs) >= RATE_LIMIT_MESSAGES:
        _user_requests[phone] = reqs
        return True
        
    reqs.append(now)
    _user_requests[phone] = reqs
    
    # Reset warning status if they drop below the limit natively (by waiting)
    if phone in _warned_users:
        _warned_users.remove(phone)
        
    # Prevent unbounded growth periodically implicitly
    if len(_user_requests) > 5000:
        _user_requests.clear()
        _warned_users.clear()
        
    return False


# ── Incoming messages ────────────────────────────────────────────
@router.post("/webhook")
async def receive_message(request: Request):
    """Receive incoming WhatsApp messages (WAHA format) and process replies."""
    print("🔔 WEBHOOK ENDPOINT HIT!")  # Force print to Vercel logs
    
    try:
        body = await request.json()
        print("WEBHOOK BODY:", body)
    except Exception:
        print("WEBHOOK ERROR: Invalid JSON")
        return {"status": "ok"}

    event = body.get("event")
    if event != "message":
        print("WEBHOOK: Ignored event type:", event)
        return {"status": "ok"}

    payload = body.get("payload", {})
    if not payload:
        return {"status": "ok"}

    msg_id = payload.get("id", "")
    sender_jid = payload.get("from", "")

    # Handle WAHA lid addressing to get real whatsapp number
    keys_data = payload.get("_data", {}).get("key", {})
    if "remoteJidAlt" in keys_data:
        alt_jid = keys_data["remoteJidAlt"]
        if "@s.whatsapp.net" in alt_jid:
            sender_jid = alt_jid.replace("@s.whatsapp.net", "@c.us")
    elif "remoteJid" in keys_data:
        remote_jid = keys_data["remoteJid"]
        if "@s.whatsapp.net" in remote_jid:
            sender_jid = remote_jid.replace("@s.whatsapp.net", "@c.us")

    if "@s.whatsapp.net" in sender_jid:
        sender_jid = sender_jid.replace("@s.whatsapp.net", "@c.us")

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
        
    # Rate Limiter
    if is_rate_limited(sender):
        logger.warning("Rate limit exceeded for %s", sender)
        if sender not in _warned_users:
            _warned_users.add(sender)
            try:
                # Send polite warning once until they wait
                await send_message(sender, "⚠️ Maaf, kamu mengirim pesan terlalu cepat. Sistem AI kami butuh waktu untuk memproses. Mohon tunggu sekitar 1 menit sebelum mengirim pesan lagi ya.")
            except Exception:
                pass
        return {"status": "ok"}

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

        # Download image from WAHA API by looking at recent chat messages
        image_bytes = await download_wa_media(phone, msg_id)
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

