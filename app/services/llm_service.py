"""LLM service — LangChain + Nebius AI Studio with Supabase chat history."""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_nebius import ChatNebius

from app.config import get_settings
from app.services.chat_history import save_message, get_history
from app.services.product_service import fetch_products, format_products_for_prompt

logger = logging.getLogger(__name__)

# ── Base system prompt (product catalog is injected dynamically) ─
SYSTEM_PROMPT_BASE = (
    "Kamu adalah asisten AI customer service untuk 'Toko Teladan Percetakan dan ATK'. "
    "Tugas utamamu adalah membantu menjawab pertanyaan pelanggan, memberikan informasi produk, "
    "katalog harga, estimasi biaya cetak, dan panduan cara order. "
    "\n\nInformasi Toko:\n"
    "- Alamat: Jl. Temu Putih No.30, Jombang Wetan, Kec. Jombang, Kota Cilegon, Banten, 42411.\n"
    "- Jam Buka: 08:00 - 17:00 WIB.\n"
    "- Kontak Pemesanan (WhatsApp): 085959929700.\n"
    "- Pembayaran: Cash, Transfer Bank, atau QRIS.\n"
)

SYSTEM_PROMPT_RULES = (
    "\n\nCara Order Cetak:\n"
    "1. Konsultasi ukuran dan bahan (pelanggan bisa kirim gambar/desain).\n"
    "2. AI akan memberikan estimasi harga.\n"
    "3. Jika pelanggan setuju, pelanggan wajib mengirim desain final (PDF/JPG resolusi tinggi).\n"
    "4. Pembayaran DP minimal 50% atau Lunas via transfer/QRIS.\n"
    "5. Pesanan diproses.\n"
    "\n\nAturan Menjawab:\n"
    "- Jawablah dengan sangat ramah, sopan, antusias dan jelas. Gunakan bahasa Indonesia yang santai layaknya chat WhatsApp.\n"
    "- Gunakan emoji secukupnya agar chat terasa hidup.\n"
    "- Jika pelanggan mengirim gambar/desain, AI (sub-sistem) akan mendeskripsikannya. Berikan saran bahan dan estimasi harganya.\n"
    "- Jika ada pertanyaan di luar cetak/ATK, sampaikan dengan sopan bahwa kamu hanya asisten Toko Teladan.\n"
    "- Jika tidak tahu harganya atau pelanggan minta pesanan khusus/partai besar, sarankan untuk hubungi CS/Admin di toko atau via WhatsApp 085959929700.\n"
    "- Jika suatu produk bertanda STOK HABIS, beritahu pelanggan dan tawarkan alternatif lain.\n"
    "- JANGAN pernah menyebutkan istilah 'costPrice' atau harga modal ke pelanggan."
)

# ── Lazy-initialised LLM instance ───────────────────────────────
_llm: ChatNebius | None = None


def _get_llm() -> ChatNebius:
    """Return (and cache) the ChatNebius instance."""
    global _llm
    if _llm is None:
        settings = get_settings()
        _llm = ChatNebius(
            api_key=settings.nebius_api_key,
            model=settings.nebius_model,
            temperature=0.7,
            top_p=0.95,
        )
    return _llm


async def _build_system_prompt() -> str:
    """Build the full system prompt with live product catalog from Supabase."""
    logger.info("LLM: Building system prompt...")
    products = await fetch_products()
    catalog_text = format_products_for_prompt(products)
    logger.info("LLM: System prompt built. Catalog size: %d bytes", len(catalog_text))

    return (
        SYSTEM_PROMPT_BASE
        + "\n\nKatalog Produk & Harga (dari database toko):\n"
        + catalog_text
        + SYSTEM_PROMPT_RULES
    )


async def get_ai_response(phone: str, user_message: str) -> str:
    """Generate an AI response using persistent Supabase history.

    Args:
        phone: The sender's phone number (conversation key).
        user_message: The text the user sent.

    Returns:
        The AI-generated reply as a plain string.
    """
    logger.info("LLM [phone=%s]: Starting response generation...", phone)
    llm = _get_llm()
    settings = get_settings()

    # Save user message to Supabase
    logger.info("LLM [phone=%s]: Saving user message to history...", phone)
    await save_message(phone, "user", user_message)

    # Load recent history from Supabase
    logger.info("LLM [phone=%s]: Loading chat history (limit=%d)...", phone, settings.max_history_length)
    history_rows = await get_history(phone, limit=settings.max_history_length)
    logger.info("LLM [phone=%s]: Loaded %d history rows.", phone, len(history_rows))

    # Build system prompt with live product data
    system_prompt = await _build_system_prompt()

    # Convert DB rows to LangChain messages
    messages = [SystemMessage(content=system_prompt)]
    for row in history_rows:
        if row["role"] == "user":
            messages.append(HumanMessage(content=row["content"]))
        elif row["role"] == "assistant":
            messages.append(AIMessage(content=row["content"]))

    logger.info("LLM [phone=%s]: Sending request to Nebius LLM (model=%s)...", phone, settings.nebius_model)
    try:
        response = await llm.ainvoke(messages)
        reply = response.content
        logger.info("LLM [phone=%s]: Response SUCCESS. Reply length: %d chars.", phone, len(str(reply)))

        # Save AI reply to Supabase
        logger.info("LLM [phone=%s]: Saving assistant reply to history...", phone)
        await save_message(phone, "assistant", reply)

        return reply  # type: ignore[return-value]
    except Exception as e:
        logger.exception("LLM [phone=%s]: ERROR calling Nebius LLM. Exception: %s", phone, str(e))
        return "Sorry, I'm having trouble thinking right now. Please try again in a moment. 🙏"

