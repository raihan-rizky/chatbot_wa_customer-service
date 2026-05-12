"""Web Push notification service backed by Supabase subscriptions."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_nebius import ChatNebius
from pywebpush import WebPushException, webpush

from app.config import get_settings
from app.services.chat_history import count_user_messages, get_history
from app.services.http_client import get_supabase_client

logger = logging.getLogger(__name__)

PUSH_TABLE = "pos_push_subscriptions"
EVENT_TABLE = "waha_events"
NEGOTIATION_TABLE = "negotiations"

CLOSING_KEYWORDS = (
    "acc",
    "admin lanjut",
    "deal",
    "fix",
    "fixed",
    "gas",
    "go",
    "oke",
    "ok",
    "siap",
    "boleh",
    "setuju",
    "jadi",
    "jadikan",
    "oke jadi",
    "ok jadi",
    "jadi ya",
    "jadi kak",
    "jadi min",
    "fix order",
    "langsung proses",
    "lanjut proses",
    "tolong proses",
    "mohon proses",
    "proses aja",
    "proses ya",
    "proses kak",
    "buatkan",
    "dibuatkan",
    "dibikin",
    "saya order",
    "mau order",
    "order",
    "pesan",
    "pemesanan",
    "booking",
    "ambil",
    "jadi ambil",
    "saya ambil",
    "mau ambil",
    "lanjut",
    "bayar",
    "dp",
    "down payment",
    "lunas",
    "transfer",
    "tf",
    "qris",
    "invoice",
    "nota",
    "saya bayar",
    "saya dp",
    "saya transfer",
    "sudah transfer",
    "kirim invoice",
    "kirim nota",
)

HIGH_INTENT_CLOSING_PHRASES = (
    "deal",
    "fix order",
    "oke jadi",
    "ok jadi",
    "jadi ya",
    "jadi kak",
    "langsung proses",
    "lanjut proses",
    "tolong proses",
    "mohon proses",
    "saya order",
    "mau order",
    "saya bayar",
    "saya dp",
    "saya transfer",
    "sudah transfer",
    "kirim invoice",
)

SHORT_CONFIRMATION_CLOSINGS = {
    "acc",
    "boleh",
    "deal",
    "dp",
    "fix",
    "gas",
    "jadi",
    "lanjut",
    "lunas",
    "ok",
    "oke",
    "qris",
    "setuju",
    "siap",
    "tf",
    "transfer",
}

CLASSIFIER_HISTORY_LIMIT = 12
PERIODIC_CLASSIFIER_MESSAGE_INTERVAL = 4
LLM_CLOSING_CONFIDENCE_THRESHOLD = 0.65
PUSH_RETRY_ATTEMPTS = 2
PUSH_RETRY_DELAY_SECONDS = 2.0
ADMIN_HANDOFF_RESPONSE_PHRASE = "admin akan melanjutkan proses di chat ini."


@dataclass(frozen=True)
class ClosingClassification:
    is_closing: bool
    confidence: float
    reason: str
    trigger: str
    fallback_used: bool = False
    missing_details: tuple[str, ...] = ()


@dataclass(frozen=True)
class PushSendResult:
    subscription: dict[str, Any]
    status: str


_closing_llm: ChatNebius | None = None


def _get_closing_llm() -> ChatNebius:
    global _closing_llm
    if _closing_llm is None:
        settings = get_settings()
        model = settings.nebius_closing_model or settings.nebius_model
        _closing_llm = ChatNebius(
            api_key=settings.nebius_api_key,
            model=model,
            temperature=0.0,
            top_p=0.8,
            max_tokens=120,
        )
    return _closing_llm


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
    """Return True when a buyer message contains a closing-like keyword."""
    return bool(_matching_closing_keywords(text))


def _text_preview(text: str, limit: int = 80) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _matching_closing_keywords(text: str) -> list[str]:
    normalized = _normalize_text(text)
    return [keyword for keyword in CLOSING_KEYWORDS if _contains_phrase(normalized, keyword)]


def _has_high_intent_closing_phrase(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(_contains_phrase(normalized, phrase) for phrase in HIGH_INTENT_CLOSING_PHRASES)


def has_high_intent_closing_phrase(text: str) -> bool:
    """Return True for buyer messages that clearly close or confirm an order."""
    return _has_high_intent_closing_phrase(text)


def is_fast_closing_confirmation(text: str) -> bool:
    """Return True for short customer confirmations that should not need LLM generation."""
    normalized = _normalize_text(text)
    words = normalized.split()
    if len(words) <= 3 and normalized in SHORT_CONFIRMATION_CLOSINGS:
        return True
    return _has_high_intent_closing_phrase(text)


def _extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in classifier response")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Classifier response JSON is not an object")
    return parsed


def _format_history_for_classifier(history_rows: list[dict[str, Any]], latest_text: str) -> str:
    lines: list[str] = []
    for row in history_rows[-CLASSIFIER_HISTORY_LIMIT:]:
        role = "customer" if row.get("role") == "user" else "assistant"
        content = str(row.get("content") or "")
        lines.append(f"{role}: {_text_preview(content, 220)}")
    lines.append(f"customer_latest: {_text_preview(latest_text, 220)}")
    return "\n".join(lines)


def _parse_classifier_result(raw_content: Any, trigger: str) -> ClosingClassification:
    parsed = _extract_json_object(str(raw_content))
    confidence = float(parsed.get("confidence") or 0)
    missing_details = parsed.get("missing_details") or []
    if not isinstance(missing_details, list):
        missing_details = []
    return ClosingClassification(
        is_closing=bool(parsed.get("is_closing")) and confidence >= LLM_CLOSING_CONFIDENCE_THRESHOLD,
        confidence=confidence,
        reason=str(parsed.get("reason") or "")[:240],
        trigger=trigger,
        missing_details=tuple(str(item)[:80] for item in missing_details),
    )


def assistant_response_requests_admin_handoff(text: str) -> bool:
    """Return True when the bot response indicates admin handoff for order processing."""
    return ADMIN_HANDOFF_RESPONSE_PHRASE in _normalize_text(text)


async def classify_closing_intent(
    phone: str,
    text: str,
    *,
    force_trigger: str | None = None,
    latest_message_saved: bool = False,
    stored_user_message_count: int | None = None,
) -> ClosingClassification:
    """Classify whether the latest customer text closes an order/deal."""
    keyword_matches = _matching_closing_keywords(text)
    keyword_gate = bool(keyword_matches)
    history_rows: list[dict[str, Any]] = []
    if stored_user_message_count is None:
        stored_user_message_count = await count_user_messages(phone)
    user_message_count = stored_user_message_count if latest_message_saved else stored_user_message_count + 1

    if not force_trigger and _has_high_intent_closing_phrase(text):
        return ClosingClassification(
            True,
            1.0,
            "high-intent closing phrase matched without LLM",
            "high_intent_keyword",
        )

    if force_trigger:
        periodic_gate = False
    elif not keyword_gate:
        periodic_gate = user_message_count % PERIODIC_CLASSIFIER_MESSAGE_INTERVAL == 0
    else:
        periodic_gate = False

    if not force_trigger and not keyword_gate and not periodic_gate:
        logger.info(
            "Closing classifier skipped phone=%s preview=%r keyword_matches=0 customer_message_count=%d",
            phone,
            _text_preview(text),
            user_message_count,
        )
        return ClosingClassification(False, 0.0, "classifier gate skipped", "skipped")

    trigger = force_trigger or ("keyword_gate" if keyword_gate else "periodic_4th_message")
    if not history_rows:
        history_rows = await get_history(phone, limit=CLASSIFIER_HISTORY_LIMIT)

    logger.info(
        "Closing classifier running phone=%s trigger=%s keyword_matches=%s history_size=%d customer_message_count=%d preview=%r",
        phone,
        trigger,
        keyword_matches,
        len(history_rows),
        user_message_count,
        _text_preview(text),
    )

    system_prompt = (
        "You classify Indonesian WhatsApp customer-service messages for Toko Teladan Percetakan & ATK.\n"
        "Decide whether the customer's latest message is a real closing/order confirmation.\n"
        "Closing means the customer clearly agrees to proceed, place an order, book, pay, DP, transfer, "
        "ask the shop to process/cetak/buatkan, or confirms they will take the item.\n"
        "Not closing: asking price/stock/timing, saying maybe/nanti/lihat dulu, asking about order terms, "
        "discussing a future possibility, or using words like ambil/proses in a non-order context.\n"
        "Use recent chat context, but classify only the latest customer message.\n"
        "Return strict JSON only with keys: is_closing boolean, confidence number 0-1, reason string, "
        "missing_details array of short strings."
    )
    user_prompt = (
        "Recent conversation:\n"
        f"{_format_history_for_classifier(history_rows, text)}\n\n"
        "Return JSON only."
    )

    try:
        llm = _get_closing_llm()
        response = await asyncio.wait_for(
            llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]),
            timeout=get_settings().nebius_closing_timeout_seconds,
        )
        result = _parse_classifier_result(response.content, trigger)
        logger.info(
            "Closing classifier result phone=%s trigger=%s is_closing=%s confidence=%.2f reason=%r missing_details=%s",
            phone,
            trigger,
            result.is_closing,
            result.confidence,
            result.reason,
            list(result.missing_details),
        )
        return result
    except Exception as exc:
        fallback_is_closing = _has_high_intent_closing_phrase(text)
        logger.exception(
            "Closing classifier failed phone=%s trigger=%s fallback_is_closing=%s error=%s",
            phone,
            trigger,
            fallback_is_closing,
            exc,
        )
        return ClosingClassification(
            is_closing=fallback_is_closing,
            confidence=1.0 if fallback_is_closing else 0.0,
            reason="LLM classifier failed; conservative high-intent fallback used",
            trigger=trigger,
            fallback_used=True,
        )


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
        client = get_supabase_client()
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
        client = get_supabase_client()
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
            "url": f"/wa/chat_id={chat_id}",
            "chat_id": chat_id,
            "phone": phone,
            "message": text[:200],
        }
    )

    await _send_push_batch_with_background_retries(subscriptions, payload, context=f"closing chat_id={chat_id}")


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

    await _send_push_batch_with_background_retries(subscriptions, payload, context=f"closing_deal chat_id={chat_id}")


async def _fetch_subscriptions() -> list[dict[str, Any]]:
    params = {
        "select": "id,endpoint,p256dh,auth",
        "isActive": "eq.true"
    }
    try:
        client = get_supabase_client()
        resp = await client.get(_table_url(PUSH_TABLE), headers=_headers(), params=params)
        if resp.status_code >= 400:
            logger.error("Failed to fetch push subscriptions: %s %s", resp.status_code, resp.text)
            return []
        return resp.json()
    except Exception as exc:
        logger.exception("Failed to fetch push subscriptions: %s", exc)
        return []


async def _send_push_batch_with_background_retries(
    subscriptions: list[dict[str, Any]],
    payload: str,
    *,
    context: str,
) -> None:
    results = await asyncio.gather(*[_send_push(sub, payload) for sub in subscriptions])
    retryable_failures = [result.subscription for result in results if result.status == "failed"]
    success_count = sum(1 for result in results if result.status == "success")
    expired_count = sum(1 for result in results if result.status == "expired")

    logger.info(
        "Web Push first attempt complete context=%s total=%d success=%d retryable_failed=%d expired=%d",
        context,
        len(subscriptions),
        success_count,
        len(retryable_failures),
        expired_count,
    )

    if retryable_failures:
        task = asyncio.create_task(_retry_failed_pushes(retryable_failures, payload, context=context))
        task.add_done_callback(_log_background_retry_error)


def _log_background_retry_error(task: asyncio.Task[None]) -> None:
    try:
        task.result()
    except Exception as exc:
        logger.exception("Web Push background retry task failed: %s", exc)


async def _retry_failed_pushes(
    subscriptions: list[dict[str, Any]],
    payload: str,
    *,
    context: str,
) -> None:
    remaining = subscriptions
    for attempt in range(1, PUSH_RETRY_ATTEMPTS + 1):
        await asyncio.sleep(PUSH_RETRY_DELAY_SECONDS)
        results = await asyncio.gather(*[_send_push(sub, payload) for sub in remaining])
        remaining = [result.subscription for result in results if result.status == "failed"]
        success_count = sum(1 for result in results if result.status == "success")
        expired_count = sum(1 for result in results if result.status == "expired")
        logger.info(
            "Web Push retry attempt context=%s attempt=%d total=%d success=%d still_failed=%d expired=%d",
            context,
            attempt,
            len(results),
            success_count,
            len(remaining),
            expired_count,
        )
        if not remaining:
            return

    logger.warning("Web Push retries exhausted context=%s final_failed=%d", context, len(remaining))


async def _send_push(subscription: dict[str, Any], payload: str) -> PushSendResult:
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
        return PushSendResult(subscription=subscription, status="success")
    except WebPushException as exc:
        logger.warning("Web Push failed for subscription %s: %s", subscription.get("id"), exc)
        if getattr(exc.response, "status_code", None) in {404, 410}:
            await _delete_subscription(subscription["endpoint"])
            return PushSendResult(subscription=subscription, status="expired")
        return PushSendResult(subscription=subscription, status="failed")
    except Exception as exc:
        logger.exception("Unexpected Web Push error: %s", exc)
        return PushSendResult(subscription=subscription, status="failed")


async def _delete_subscription(endpoint: str) -> None:
    params = {"endpoint": f"eq.{endpoint}"}
    try:
        client = get_supabase_client()
        resp = await client.delete(_table_url(PUSH_TABLE), headers=_headers("return=minimal"), params=params)
        if resp.status_code >= 400:
            logger.error("Failed to delete expired push subscription: %s %s", resp.status_code, resp.text)
    except Exception as exc:
        logger.exception("Failed to delete expired push subscription: %s", exc)
