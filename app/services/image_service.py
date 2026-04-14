"""Image service — download WhatsApp media & extract receipt info via vision model."""

from __future__ import annotations

import base64
import json
import logging
import re

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_nebius import ChatNebius

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Lazy-initialised vision LLM ─────────────────────────────────
_vision_llm: ChatNebius | None = None
_llm: ChatNebius | None = None


def _get_vision_llm() -> ChatNebius:
    """Return (and cache) the vision-capable ChatNebius instance."""
    global _vision_llm
    if _vision_llm is None:
        settings = get_settings()
        _vision_llm = ChatNebius(
            api_key=settings.nebius_api_key,
            model=settings.nebius_vision_model,
            temperature=0.1,  # very low temp for strict JSON
            max_tokens=2048,
        )
    return _vision_llm

def _get_llm() -> ChatNebius:
    """Return (and cache) the ChatNebius instance."""
    global _llm
    if _llm is None:
        settings = get_settings()
        _llm = ChatNebius(
            api_key=settings.nebius_api_key,
            model=settings.nebius_model,
            temperature=0.1,
            max_tokens=2048,
        )
    return _llm


# ── Design extraction prompt ───────────────────────────────────
DESIGN_IMAGE_PROMPT = """Kamu adalah asisten percetakan ahli di Toko Teladan.
Tugasmu adalah menganalisis gambar/desain yang dikirim pelanggan dan memberikan estimasi atau saran cetak.

Panduan Analisis:
1. Deskripsikan secara singkat gambar apa itu (misal: logo, desain spanduk, brosur, atau poster).
2. Sebutkan warna-warna dominan atau elemen utama.
3. Berikan saran bahan yang cocok (misal: Spanduk Flexi Korea, Cetak Luster, Stiker Vinyl) dan sebutkan harganya berdasarkan katalog berikut:
   - Spanduk Flexi 280gr (China) - Rp 25.000/m²
   - Spanduk Flexi 340gr (Korea) - Rp 45.000/m²
   - Spanduk Flexi 510gr (Jerman) - Rp 85.000/m²
   - Cetak Luster - Rp 115.000/m²
   - Cetak PVC Rigid - Rp 120.000/m²
   - Stiker Vinyl - Rp 75.000/m²
   - Stiker One Way Vision - Rp 85.000/m²
4. Jika ada teks di dalam gambar, baca dan sebutkan teks apa yang terlihat (OCR ringan).

Format Keluaran:
Gunakan bahasa Indonesia yang santai, ramah, dan profesional layaknya admin WhatsApp.
Gunakan emoji secukupnya. Jawab langsung dalam paragraf rapi tanpa format terstruktur (JSON).
"""

GENERAL_IMAGE_PROMPT = """Kamu adalah asisten AI yang dapat melihat gambar.
Tolong deskripsikan gambar ini dengan ramah kepada pelanggan."""

async def download_wa_media(msg_id: str) -> bytes:
    """Download media from WAHA API using the message ID."""
    settings = get_settings()
    url = f"{settings.waha_base_url}/api/{settings.waha_session}/messages/{msg_id}/download"

    headers = {}
    if settings.waha_api_key:
        headers["X-Api-Key"] = settings.waha_api_key

    async with httpx.AsyncClient(timeout=30.0) as client:
        logger.info("📥 Downloading media via WAHA for msg %s", msg_id)
        
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            logger.error("Failed to download media %s — HTTP %s", msg_id, resp.status_code)
            return b""
            
        logger.info("📥 Downloaded %d bytes", len(resp.content))
        return resp.content


async def analyze_image(image_bytes: bytes, caption: str | None = None) -> str:
    """Analyze an image using the Nebius vision model.
    
    Returns:
        str: Description and design estimation.
    """
    llm = _get_vision_llm()

    # Encode image to base64
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Build multimodal message
    system_prompt = DESIGN_IMAGE_PROMPT
    user_text = f"Tolong lihat gambar desain ini dan berikan saran cetak."
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

    try:
        response = await llm.ainvoke(messages)
        content = response.content
        logger.info("Vision LLM output: %s", str(content)[:200]) # Log first 200 chars

        return str(content)

    except Exception:
        logger.exception("Vision model call failed")
        return "Maaf, saya gagal menganalisa gambar ini. Coba kirim ulang dengan resolusi lebih jelas ya! 🙏"
