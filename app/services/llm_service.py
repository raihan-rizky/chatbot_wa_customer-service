"""LLM service — LangChain + Nebius AI Studio with Supabase chat history."""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_nebius import ChatNebius

from app.config import get_settings
from app.services.chat_history import save_message, get_history

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Kamu adalah asisten AI customer service untuk 'Toko Teladan Percetakan dan ATK'. "
    "Tugas utamamu adalah membantu menjawab pertanyaan pelanggan, memberikan informasi produk, "
    "katalog harga, estimasi biaya cetak, dan panduan cara order. "
    "\n\nInformasi Toko:\n"
    "- Alamat: Jl. Temu Putih No.30, Jombang Wetan, Kec. Jombang, Kota Cilegon, Banten, 42411.\n"
    "- Jam Buka: 08:00 - 17:00 WIB.\n"
    "- Kontak Pemesanan (WhatsApp): 085959929700.\n"
    "- Pembayaran: Cash, Transfer Bank, atau QRIS.\n"
    "\n\nKatalog Produk & Harga (Banner/Spanduk):\n"
    "1. Spanduk Flexi 280gr (China) - Rp 25.000/m²\n"
    "2. Spanduk Flexi 340gr (Korea) - Rp 45.000/m²\n"
    "3. Spanduk Flexi 510gr (Jerman) - Rp 85.000/m²\n"
    "4. Cetak Luster - Rp 115.000/m²\n"
    "5. Cetak PVC Rigid - Rp 120.000/m²\n"
    "6. Stiker Vinyl - Rp 75.000/m²\n"
    "7. Stiker One Way Vision - Rp 85.000/m²\n"
    "\n\nKatalog ATK (Alat Tulis Kantor):\n"
    "Tersedia berbagai kebutuhan ATK seperti pulpen, buku tulis, amplop, kertas HVS, map, penggaris, "
    "dan alat tulis sekolah/kantor lainnya dengan harga terjangkau.\n"
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
    "- Jika tidak tahu harganya atau pelanggan minta pesanan khusus/partai besar, sarankan untuk hubungi CS/Admin di toko atau via WhatsApp 085959929700."
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


async def get_ai_response(phone: str, user_message: str) -> str:
    """Generate an AI response using persistent Supabase history.

    Args:
        phone: The sender's phone number (conversation key).
        user_message: The text the user sent.

    Returns:
        The AI-generated reply as a plain string.
    """
    llm = _get_llm()
    settings = get_settings()

    # Save user message to Supabase
    await save_message(phone, "user", user_message)

    # Load recent history from Supabase
    history_rows = await get_history(phone, limit=settings.max_history_length)

    # Convert DB rows to LangChain messages
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for row in history_rows:
        if row["role"] == "user":
            messages.append(HumanMessage(content=row["content"]))
        elif row["role"] == "assistant":
            messages.append(AIMessage(content=row["content"]))

    try:
        response = await llm.ainvoke(messages)
        reply = response.content

        # Save AI reply to Supabase
        await save_message(phone, "assistant", reply)

        return reply  # type: ignore[return-value]
    except Exception:
        logger.exception("Nebius LLM call failed for phone=%s", phone)
        return "Sorry, I'm having trouble thinking right now. Please try again in a moment. 🙏"


