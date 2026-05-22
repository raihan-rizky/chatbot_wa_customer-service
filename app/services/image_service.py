"""Image service - download WhatsApp media & analyze designs via vision model."""

from __future__ import annotations

import base64
import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_nebius import ChatNebius

from app.config import get_settings
from app.services.http_client import get_waha_client
from app.services.product_service import (
    fetch_catalog_summary,
    fetch_matching_products,
    format_catalog_summary,
    format_products_for_prompt,
)

logger = logging.getLogger(__name__)

# -- Lazy-initialised vision LLMs ---------------------------------------
_vision_llm: ChatNebius | None = None
_vision_tagger: ChatNebius | None = None


def _get_vision_llm() -> ChatNebius:
    global _vision_llm
    if _vision_llm is None:
        settings = get_settings()
        _vision_llm = ChatNebius(
            api_key=settings.nebius_api_key,
            model=settings.nebius_vision_model,
            temperature=0.2,
            max_tokens=512,
        )
    return _vision_llm


def _get_vision_tagger() -> ChatNebius:
    """Cheap, low-token vision call used only to extract search keywords."""
    global _vision_tagger
    if _vision_tagger is None:
        settings = get_settings()
        _vision_tagger = ChatNebius(
            api_key=settings.nebius_api_key,
            model=settings.nebius_vision_model,
            temperature=0.0,
            max_tokens=96,
        )
    return _vision_tagger


TAGGER_PROMPT = (
    "Anda adalah pengekstrak kata kunci. Lihat gambar dan balas HANYA dalam JSON: "
    '{"keywords": ["kata1", "kata2", ...]}. '
    "Sertakan 3-6 kata kunci bahasa Indonesia/Inggris yang menggambarkan jenis cetakan "
    "(spanduk, stiker, banner, brosur, poster, kertas, atk, dll), bahan jika terlihat, "
    "dan ukuran (a3/a4) jika terbaca. Tanpa kalimat lain."
)

DESIGN_PROMPT_TEMPLATE = """Kamu adalah asisten percetakan ahli di Toko Teladan.
Tugasmu adalah menganalisis gambar/desain yang dikirim pelanggan dan memberikan estimasi atau saran cetak.

Panduan Analisis:
1. Deskripsikan secara singkat gambar apa itu (misal: logo, desain spanduk, brosur, atau poster).
2. Sebutkan warna-warna dominan atau elemen utama.
3. Cek deskripsi gambar apakah cocok/sesuai dengan produk yang ada pada katalog toko berikut:
{catalog}
Jika cocok, berikan informasi produk tersebut seperti harga dan stok (namun JANGAN pernah menyebutkan istilah 'costPrice' atau harga modal ke pelanggan).
4. Jika tidak berkaitan dengan produk alat tulis, perlengkapan kantor (stationary), percetakan, atau banner (luar domain/konteks toko), maka langsung jawab dengan tegas namun ramah:
"Mohon maaf, itu di luar layanan kami. Silakan tulis atau unggah gambar barang-barang yang berkaitan dengan percetakan, spanduk, atau alat tulis/kantor saja ya."
5. Jika ada teks di dalam gambar, baca dan sebutkan teks apa yang terlihat (OCR ringan).
6. JAWAB MAKSIMAL 3 KALIMAT.

Format Keluaran:
Gunakan bahasa Indonesia yang santai, ramah, dan profesional layaknya admin WhatsApp.
Gunakan emoji secukupnya. Jawab langsung dalam paragraf rapi tanpa format terstruktur (JSON).
"""


def _parse_tagger_keywords(raw: str) -> list[str]:
    """Pull the keyword list out of the tagger's JSON reply, leniently."""
    if not raw:
        return []
    text = raw.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    keywords = parsed.get("keywords") or []
    return [str(k).strip() for k in keywords if str(k).strip()]


async def _extract_image_keywords(b64_image: str) -> list[str]:
    """Run a tiny vision call that returns ~5 search keywords for the image."""
    tagger = _get_vision_tagger()
    messages = [
        SystemMessage(content=TAGGER_PROMPT),
        HumanMessage(
            content=[
                {"type": "text", "text": "Kata kunci?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                },
            ]
        ),
    ]
    try:
        response = await tagger.ainvoke(messages)
        keywords = _parse_tagger_keywords(str(response.content))
        logger.info("Vision tagger: extracted %d keywords: %s", len(keywords), keywords)
        return keywords
    except Exception as e:
        logger.exception("Vision tagger ERROR: %s", str(e))
        return []


async def _build_catalog_block(caption: str | None, b64_image: str) -> str:
    """Resolve a small, relevant catalog slice for this image.

    Order of attempts:
      1. Caption keywords (free, no extra LLM call).
      2. Vision tagger keywords (one cheap vision call).
      3. Catalog summary (categories + price ranges).
    """
    products: list[dict] = []

    if caption and caption.strip():
        products = await fetch_matching_products(caption, limit=30)
        if products:
            logger.info("Vision LLM: caption matched %d products", len(products))
            return format_products_for_prompt(products)

    keywords = await _extract_image_keywords(b64_image)
    if keywords:
        products = await fetch_matching_products(" ".join(keywords), limit=30)
        if products:
            logger.info("Vision LLM: image tags matched %d products", len(products))
            return format_products_for_prompt(products)

    summary = await fetch_catalog_summary()
    logger.info("Vision LLM: falling back to catalog summary (%d categories)", len(summary.get("categories", [])))
    return format_catalog_summary(summary)


async def download_wa_media(phone: str, msg_id: str) -> bytes:
    """Download media from WAHA API by fetching recent chat messages."""
    settings = get_settings()
    chat_id = f"{phone}@c.us"
    url = f"{settings.waha_base_url}/api/{settings.waha_session}/chats/{chat_id}/messages?limit=10&downloadMedia=true"

    headers = {}
    if settings.waha_api_key:
        headers["X-Api-Key"] = settings.waha_api_key

    logger.info("WAHA Media: START FETCH [phone=%s, msg_id=%s]", phone, msg_id)

    try:
        client = get_waha_client()
        logger.info("WAHA Media: Requesting chat history from %s", url)
        resp = await client.get(url, headers=headers)

        if resp.status_code != 200:
            logger.error("WAHA Media ERROR: Failed to get messages [status=%s, body=%s]", resp.status_code, resp.text[:200])
            return b""

        messages = resp.json()
        logger.info("WAHA Media: Retrieved %d messages from history", len(messages))

        target_media_url = None

        for i, msg in enumerate(messages):
            has_media = msg.get("hasMedia")
            media_data = msg.get("media")
            m_id = msg.get("id", "")

            if has_media and media_data:
                found_url = media_data.get("url")
                if found_url:
                    logger.info("WAHA Media: Found media in msg[%d] (id=%s). URL: %s", i, m_id, found_url)
                    target_media_url = found_url
                    if msg_id in m_id:
                        logger.info("WAHA Media: Exact msg_id match found at index %d", i)
                        break

        if not target_media_url:
            logger.error("WAHA Media ERROR: No media URL found in recent messages for msg %s", msg_id)
            return b""

        if "localhost" in target_media_url or "127.0.0.1" in target_media_url:
            from urllib.parse import urlparse
            parsed_media = urlparse(target_media_url)
            parsed_base = urlparse(settings.waha_base_url)
            new_url = target_media_url.replace(
                f"{parsed_media.scheme}://{parsed_media.netloc}",
                f"{parsed_base.scheme}://{parsed_base.netloc}"
            )
            logger.info("WAHA Media: Applied host correction: %s -> %s", target_media_url, new_url)
            target_media_url = new_url

        logger.info("WAHA Media: Downloading actual file bytes from %s", target_media_url)
        file_resp = await client.get(target_media_url, headers=headers)
        if file_resp.status_code != 200:
            logger.error("WAHA Media ERROR: Download failed [status=%s]", file_resp.status_code)
            return b""

        logger.info("WAHA Media SUCCESS: Downloaded %d bytes", len(file_resp.content))
        return file_resp.content

    except Exception as e:
        logger.exception("WAHA Media EXCEPTION: Unexpected error during fetch: %s", str(e))
        return b""


async def analyze_image(image_bytes: bytes, caption: str | None = None) -> str:
    """Analyze an image using the Nebius vision model.

    Returns:
        str: Description and design estimation.
    """
    logger.info("Vision LLM: Starting image analysis. Image size: %d bytes, caption: %s", len(image_bytes), caption)
    llm = _get_vision_llm()

    logger.info("Vision LLM: Encoding image to base64...")
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    catalog_text = await _build_catalog_block(caption, b64_image)
    system_prompt = DESIGN_PROMPT_TEMPLATE.format(catalog=catalog_text)
    logger.info("Vision LLM: Catalog block size %d bytes", len(catalog_text))

    user_text = "Tolong lihat gambar desain ini dan berikan saran cetak."
    if caption:
        user_text += f"\nCatatan dari pelanggan: {caption}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=[
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                },
            ]
        ),
    ]

    logger.info("Vision LLM: Sending request to vision model...")
    try:
        response = await llm.ainvoke(messages)
        content = response.content
        logger.info("Vision LLM SUCCESS: Received response. Preview: %s", str(content)[:100].replace('\n', ' '))
        return str(content)
    except Exception as e:
        logger.exception("Vision LLM ERROR: Vision model call failed. Exception: %s", str(e))
        return "Maaf, saya gagal menganalisa gambar ini. Coba kirim ulang dengan resolusi lebih jelas ya! \U0001F64F"