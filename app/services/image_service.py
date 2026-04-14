"""Image service — download WhatsApp media & analyze designs via vision model."""

from __future__ import annotations

import base64
import logging

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_nebius import ChatNebius

from app.config import get_settings
from app.services.product_service import fetch_products, format_products_for_prompt

logger = logging.getLogger(__name__)

# ── Lazy-initialised vision LLM ─────────────────────────────────
_vision_llm: ChatNebius | None = None


def _get_vision_llm() -> ChatNebius:
    """Return (and cache) the vision-capable ChatNebius instance."""
    global _vision_llm
    if _vision_llm is None:
        settings = get_settings()
        _vision_llm = ChatNebius(
            api_key=settings.nebius_api_key,
            model=settings.nebius_vision_model,
            temperature=0.1,  # very low temp for strict analysis
            max_tokens=2048,
        )
    return _vision_llm


# ── Design prompt template (catalog injected at runtime) ────────
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

Format Keluaran:
Gunakan bahasa Indonesia yang santai, ramah, dan profesional layaknya admin WhatsApp.
Gunakan emoji secukupnya. Jawab langsung dalam paragraf rapi tanpa format terstruktur (JSON).
"""


async def _build_design_prompt() -> str:
    """Build the design analysis prompt with live product catalog."""
    logger.info("Vision LLM: Building design prompt...")
    products = await fetch_products()
    catalog_text = format_products_for_prompt(products)
    logger.info("Vision LLM: Design prompt built. Catalog size: %d bytes", len(catalog_text))
    return DESIGN_PROMPT_TEMPLATE.format(catalog=catalog_text)


async def download_wa_media(phone: str, msg_id: str) -> bytes:
    """Download media from WAHA API by fetching recent chat messages."""
    settings = get_settings()
    chat_id = f"{phone}@c.us"
    url = f"{settings.waha_base_url}/api/{settings.waha_session}/chats/{chat_id}/messages?limit=10&downloadMedia=true"

    headers = {}
    if settings.waha_api_key:
        headers["X-Api-Key"] = settings.waha_api_key

    logger.info("WAHA Media: Fetching messages for %s to find media", chat_id)
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            logger.error("WAHA Media ERROR: Failed to get messages for %s — HTTP %s", chat_id, resp.status_code)
            return b""
        
        messages = resp.json()
        target_media_url = None
        
        # Look for the message that has Media
        for msg in messages:
            if msg.get("hasMedia") and msg.get("media") and msg["media"].get("url"):
                target_media_url = msg["media"]["url"]
                # Optionally check if this is the exact message if IDs match
                # But since ID from payload and ID from API might differ slightly,
                # we just take the first recent media if msg_id doesn't exactly match.
                msg_id_in_api = msg.get("id", "")
                if msg_id in msg_id_in_api:
                    break
        
        if not target_media_url:
            logger.error("WAHA Media ERROR: No media found in recent messages for msg %s", msg_id)
            return b""
            
        # Download the actual file from the URL found
        # Usually target_media_url is a fully qualified URL to WAHA's file server.
        # But if it's localhost and we're inside docker, we might need to replace the base URL.
        # For safety, if waha_base_url is different from the origin of target_media_url, replace it.
        if "localhost" in target_media_url or "127.0.0.1" in target_media_url:
            # Simple replacement if WAHA is running in docker (e.g., http://waha:3000)
            from urllib.parse import urlparse
            parsed_media = urlparse(target_media_url)
            parsed_base = urlparse(settings.waha_base_url)
            target_media_url = target_media_url.replace(f"{parsed_media.scheme}://{parsed_media.netloc}", f"{parsed_base.scheme}://{parsed_base.netloc}")

        logger.info("WAHA Media: Downloading from URL %s", target_media_url)
        # Download media file
        file_resp = await client.get(target_media_url, headers=headers)
        if file_resp.status_code != 200:
            logger.error("WAHA Media ERROR: Failed to download media file — HTTP %s", file_resp.status_code)
            return b""
            
        logger.info("WAHA Media SUCCESS: Downloaded %d bytes", len(file_resp.content))
        return file_resp.content


async def analyze_image(image_bytes: bytes, caption: str | None = None) -> str:
    """Analyze an image using the Nebius vision model.

    Returns:
        str: Description and design estimation.
    """
    logger.info("Vision LLM: Starting image analysis. Image size: %d bytes, caption: %s", len(image_bytes), caption)
    llm = _get_vision_llm()

    # Encode image to base64
    logger.info("Vision LLM: Encoding image to base64...")
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Build dynamic prompt with live catalog
    system_prompt = await _build_design_prompt()

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
        return "Maaf, saya gagal menganalisa gambar ini. Coba kirim ulang dengan resolusi lebih jelas ya! 🙏"

