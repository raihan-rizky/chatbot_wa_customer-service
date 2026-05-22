"""WhatsApp webhook routes — incoming messages (WAHA format)."""

from __future__ import annotations

import asyncio
import hmac
import logging
import traceback
import time

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.config import get_settings
from app.logging_config import set_phone
from app.services.abuse_control import get_sender_defense, mark_blocked, record_abuse
from app.services.llm_service import get_ai_response
from app.services.chat_history import count_user_messages, save_message
from app.services.image_service import analyze_image, download_wa_media
from app.services.push_notifications import (
    assistant_response_requests_admin_handoff,
    classify_closing_intent,
    mark_negotiation_closed,
    notify_closing,
    notify_closing_deal,
    save_waha_event,
)
from app.services.whatsapp import block_contact, send_message

logger = logging.getLogger(__name__)

router = APIRouter()

# Track processed message IDs to avoid duplicates
_processed_ids: set[str] = set()

# Rate limiting state
_user_requests: dict[str, list[float]] = {}
_warned_users: set[str] = set()
_blocked_users: set[str] = set()


class ClosingDealPushRequest(BaseModel):
    customerName: str = Field(..., min_length=1, max_length=120)
    chatId: str = Field(..., min_length=1, max_length=120)
    amount: float | int | None = Field(default=None, ge=0)
    message: str = Field(..., min_length=1, max_length=500)
    url: str = Field(default="/wa", min_length=1, max_length=300)


def _require_closing_deal_push_auth(authorization: str | None) -> None:
    settings = get_settings()
    if not settings.closing_deal_push_secret:
        logger.error("CLOSING_DEAL_PUSH_SECRET is not configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Closing deal push endpoint is not configured",
        )

    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token required")

    if not hmac.compare_digest(token, settings.closing_deal_push_secret):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token")

def is_rate_limited(phone: str, *, max_messages: int, window_seconds: int) -> bool:
    """Check if a phone number exceeds the allowed rate limit."""
    now = time.time()
    reqs = _user_requests.get(phone, [])
    reqs = [t for t in reqs if now - t < window_seconds]

    if len(reqs) >= max_messages:
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


async def _enforce_abuse_block(sender: str, reason: str) -> None:
    """Persist and apply a hard block to a sender."""
    if sender in _blocked_users:
        return

    _blocked_users.add(sender)
    try:
        await mark_blocked(sender, reason)
    except Exception:
        logger.exception("Failed to persist block state for %s", sender)

    settings = get_settings()
    if settings.auto_block_abusive_strangers:
        try:
            await block_contact(sender)
        except Exception:
            logger.exception("Failed to block WAHA contact %s", sender)


@router.post("/api/push/closing-deal")
async def push_closing_deal(
    payload: ClosingDealPushRequest,
    authorization: str | None = Header(default=None),
):
    """Receive closed deal notifications from an external WAHA/POS workflow."""
    _require_closing_deal_push_auth(authorization)

    chat_id = payload.chatId.strip()
    customer_name = payload.customerName.strip()
    message = payload.message.strip()
    url = payload.url.strip() or "/wa"

    await mark_negotiation_closed(chat_id, customer_name)
    await notify_closing_deal(
        chat_id=chat_id,
        customer_name=customer_name,
        amount=payload.amount,
        message=message,
        url=url,
    )

    return {"status": "ok"}


# ── Incoming messages ────────────────────────────────────────────
@router.post("/webhook")
async def receive_message(request: Request, background_tasks: BackgroundTasks):
    """Receive incoming WhatsApp messages (WAHA format) and process replies."""
    logger.info("Webhook endpoint hit")
    
    try:
        body = await request.json()
        logger.debug("Webhook body: %s", body)
    except Exception:
        logger.warning("Webhook error: invalid JSON")
        return {"status": "ok"}

    event = body.get("event")
    if event != "message":
        logger.info("Webhook ignored event type: %s", event)
        return {"status": "ok"}

    payload = body.get("payload", {})
    if not payload:
        return {"status": "ok"}

    msg_id = payload.get("id", "")
    sender_jid = payload.get("from", "")

    # Handle WAHA lid addressing to get real whatsapp number
    _data = payload.get("_data") or {}
    keys_data = _data.get("key") or {}
    
    if "remoteJidAlt" in keys_data:
        alt_jid = keys_data["remoteJidAlt"] or ""
        if "@s.whatsapp.net" in alt_jid:
            sender_jid = alt_jid.replace("@s.whatsapp.net", "@c.us")
    elif "remoteJid" in keys_data:
        remote_jid = keys_data["remoteJid"] or ""
        if "@s.whatsapp.net" in remote_jid:
            sender_jid = remote_jid.replace("@s.whatsapp.net", "@c.us")

    sender_jid = sender_jid or ""
    if "@s.whatsapp.net" in sender_jid:
        sender_jid = sender_jid.replace("@s.whatsapp.net", "@c.us")

    to_jid = payload.get("to") or ""
    participant = payload.get("participant") or ""
    remote_jid = keys_data.get("remoteJid") or ""
    
    is_group = (
        "@g.us" in sender_jid or 
        "@g.us" in to_jid or 
        "@g.us" in participant or 
        "@g.us" in remote_jid or
        payload.get("isGroup") is True
    )

    is_broadcast = (
        "status@broadcast" in sender_jid or
        "status@broadcast" in to_jid or
        "status@broadcast" in remote_jid
    )
    
    is_newsletter = (
        "@newsletter" in sender_jid or
        "@newsletter" in to_jid or
        "@newsletter" in remote_jid
    )

    # Strict check: only process if it's a personal message
    if is_group or is_broadcast or is_newsletter or not sender_jid.endswith("@c.us"):
        logger.info("Ignored non-personal message. sender_jid=%s, is_group=%s", sender_jid, is_group)
        return {"status": "ok"}

    # Ignore messages sent by the bot itself
    if payload.get("fromMe", False):
        return {"status": "ok"}

    sender = sender_jid.replace("@c.us", "")

    # Deduplicate before any external writes.
    if msg_id in _processed_ids:
        logger.info("Skipping duplicate message %s", msg_id)
        return {"status": "ok"}
    _processed_ids.add(msg_id)
    if len(_processed_ids) > 1000:
        _processed_ids.clear()

    background_tasks.add_task(_process_incoming_message, event, sender, msg_id, payload)
    return {"status": "ok"}


async def _process_incoming_message(event: str, sender: str, msg_id: str, payload: dict) -> None:
    """Process a WAHA message after the webhook has already been acknowledged."""
    set_phone(sender)
    process_started = time.perf_counter()
    audit_task = asyncio.create_task(save_waha_event(event, sender, msg_id, payload))

    defense_task = asyncio.create_task(get_sender_defense(sender))
    count_task = asyncio.create_task(count_user_messages(sender))

    defense_result, count_result = await asyncio.gather(
        defense_task,
        count_task,
        return_exceptions=True,
    )
    if isinstance(defense_result, Exception):
        logger.exception(
            "Failed to load defense state for %s",
            sender,
            exc_info=(type(defense_result), defense_result, defense_result.__traceback__),
        )
        defense = None
    else:
        defense = defense_result

    if defense and defense.get("is_blocked"):
        logger.warning("Ignoring blocked sender %s", sender)
        _blocked_users.add(sender)
        return

    settings = get_settings()
    if isinstance(count_result, Exception):
        logger.exception(
            "Failed to count messages for %s",
            sender,
            exc_info=(type(count_result), count_result, count_result.__traceback__),
        )
        user_message_count = settings.stranger_trust_message_count
    else:
        user_message_count = count_result
    is_stranger = user_message_count < settings.stranger_trust_message_count
    rate_limit_messages = (
        settings.stranger_rate_limit_messages
        if is_stranger
        else settings.rate_limit_messages
    )
    rate_limit_window = (
        settings.stranger_rate_limit_window_seconds
        if is_stranger
        else settings.rate_limit_window_seconds
    )

    # Rate Limiter
    if is_rate_limited(
        sender,
        max_messages=rate_limit_messages,
        window_seconds=rate_limit_window,
    ):
        logger.warning("Rate limit exceeded for %s", sender)
        try:
            abuse = await record_abuse(
                sender,
                "rate_limit_exceeded_stranger" if is_stranger else "rate_limit_exceeded",
            )
            if abuse["abuse_count"] >= settings.abuse_block_threshold:
                logger.warning(
                    "Abuse threshold reached for %s; blocking sender (count=%d)",
                    sender,
                    abuse["abuse_count"],
                )
                await _enforce_abuse_block(sender, abuse["block_reason"])
                return
        except Exception:
            logger.exception("Failed to record abuse for %s", sender)
        if sender not in _warned_users:
            _warned_users.add(sender)
            try:
                # Send polite warning once until they wait
                await send_message(sender, "⚠️ Maaf, kamu mengirim pesan terlalu cepat. Sistem AI kami butuh waktu untuk memproses. Mohon tunggu sekitar 1 menit sebelum mengirim pesan lagi ya.")
            except Exception:
                pass
        return

    msg_type = payload.get("type", "chat")
    has_media = payload.get("hasMedia", False)

    logger.info("Webhook from %s type=%s has_media=%s id=%s", sender, msg_type, has_media, msg_id)

    try:
        if has_media or msg_type == "image":
            await _handle_single_image(sender, payload)
        elif msg_type == "chat":
            text = payload.get("body", "")
            if text:
                # Pre-classify closing intent BEFORE generating the AI reply
                # so we can inject upselling instructions when it's a deal.
                pre_closing_result = await classify_closing_intent(
                    sender,
                    text,
                    latest_message_saved=False,
                    stored_user_message_count=user_message_count + 1,
                )
                is_closing = pre_closing_result.is_closing
                logger.info(
                    "Pre-classification for %s message=%s is_closing=%s trigger=%s confidence=%.2f",
                    sender,
                    msg_id,
                    is_closing,
                    pre_closing_result.trigger,
                    pre_closing_result.confidence,
                )

                reply_started = time.perf_counter()
                reply = await _handle_text(sender, text, is_closing=is_closing)
                logger.info(
                    "Customer reply path completed for %s message=%s elapsed_ms=%d total_elapsed_ms=%d",
                    sender,
                    msg_id,
                    int((time.perf_counter() - reply_started) * 1000),
                    int((time.perf_counter() - process_started) * 1000),
                )

                # If pre-classification already detected closing, skip re-classifying
                # and directly mark the negotiation + notify admin.
                if is_closing:
                    logger.info(
                        "Closing pre-detected for %s from message %s trigger=%s confidence=%.2f fallback=%s reason=%s",
                        sender,
                        msg_id,
                        pre_closing_result.trigger,
                        pre_closing_result.confidence,
                        pre_closing_result.fallback_used,
                        pre_closing_result.reason,
                    )
                    await mark_negotiation_closed(sender)
                    await notify_closing(sender, sender, text)
                else:
                    # Fallback: check if the assistant's reply indicates admin handoff
                    # (the LLM may have detected closing intent that keywords missed).
                    if reply and assistant_response_requests_admin_handoff(reply):
                        logger.info(
                            "Assistant handoff phrase detected for %s from message %s; forcing closing classifier",
                            sender,
                            msg_id,
                        )
                        handoff_result = await classify_closing_intent(
                            sender,
                            text,
                            force_trigger="assistant_admin_handoff_phrase",
                            latest_message_saved=True,
                            stored_user_message_count=user_message_count + 1,
                        )
                        if handoff_result.is_closing:
                            logger.info(
                                "Closing detected after assistant handoff for %s from message %s trigger=%s confidence=%.2f fallback=%s reason=%s",
                                sender,
                                msg_id,
                                handoff_result.trigger,
                                handoff_result.confidence,
                                handoff_result.fallback_used,
                                handoff_result.reason,
                            )
                            await mark_negotiation_closed(sender)
                            await notify_closing(sender, sender, text)
                        else:
                            logger.info(
                                "Assistant handoff classifier did not detect closing for %s from message %s confidence=%.2f reason=%s",
                                sender,
                                msg_id,
                                handoff_result.confidence,
                                handoff_result.reason,
                            )
                    else:
                        logger.info(
                            "Closing not detected for %s from message %s trigger=%s confidence=%.2f reason=%s",
                            sender,
                            msg_id,
                            pre_closing_result.trigger,
                            pre_closing_result.confidence,
                            pre_closing_result.reason,
                        )
        else:
            logger.info("Skipping unsupported message type: %s", msg_type)
    except Exception:
        logger.error("Error processing webhook:\n%s", traceback.format_exc())
    finally:
        await audit_task
        logger.info(
            "Webhook processing finished for %s message=%s elapsed_ms=%d",
            sender,
            msg_id,
            int((time.perf_counter() - process_started) * 1000),
        )


async def _handle_text(phone: str, text: str, *, is_closing: bool = False) -> str | None:
    """Handle a text message — generate AI reply and save to Supabase.

    When is_closing=True, the LLM will include an upselling suggestion
    for a relevant complementary product in its reply.
    """
    logger.info("Text from %s: %s", phone, text[:80])
    started = time.perf_counter()

    try:
        # LLM will handle everything naturally based on its prompt
        reply = await get_ai_response(phone, text, is_closing=is_closing)
        logger.info(
            "AI reply ready for %s elapsed_ms=%d; sending to WAHA",
            phone,
            int((time.perf_counter() - started) * 1000),
        )
        await send_message(phone, reply)
        logger.info(
            "Reply sent to %s total_elapsed_ms=%d",
            phone,
            int((time.perf_counter() - started) * 1000),
        )
        return reply
    except Exception:
        logger.error("Failed to reply to %s:\n%s", phone, traceback.format_exc())
        try:
            await send_message(phone, "Maaf, terjadi kesalahan. Coba kirim ulang pesan kamu. 🙏")
        except Exception:
            pass
        return None


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
