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
    "CS 'Toko Teladan Percetakan & ATK'. Jawab seputar produk/harga/order.\n"
    "Jl. Temu Putih No.30 Cilegon. 08:00-17:00. WA:085959929700. Cash/Trf/QRIS."
)

SYSTEM_PROMPT_RULES = (
    "\n\nATURAN WAJIB:\n"
    "- Jawab sesingkat mungkin. Maksimal 2-3 kalimat.\n"
    "- Langsung berikan harga atau info tanpa basa-basi.\n"
    "- Ramah, 1-2 emoji.\n"
    "- Gambar/desain: deskripsikan, beri saran & estimasi.\n"
    "- Tolak di luar ATK/cetak.\n"
    "- Order khusus/partai besar/tak tahu harga -> WA 085959929700.\n"
    "- STOK 0 -> tawarkan opsi lain.\n"
    "- DILARANG sebut 'costPrice'/modal.\n"
    "Alur: 1.Tanya 2.Estimasi 3.Desain 4.DP/Lunas 5.Proses."
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
            temperature=0.1, # Diturunkan agar jawaban ringkas & deterministik
            top_p=0.95,
        )
    return _llm


async def _build_system_prompt(user_message: str) -> str:
    """Build the full system prompt with live product catalog from Supabase."""
    logger.info("LLM: Building system prompt...")
    products = await fetch_products()
    
    # Keyword-Based RAG: Filter products based on user message
    user_words = [word for word in user_message.lower().split() if len(word) >= 3]
    filtered_products = []
    
    for p in products:
        searchable_text = f"{p.get('name', '')} {p.get('categoryId', '')} {p.get('material', '')}".lower()
        if any(word in searchable_text for word in user_words):
            filtered_products.append(p)
            
    if not filtered_products:
        logger.info("LLM: No matching products found for message. Omitting catalog.")
        return SYSTEM_PROMPT_BASE + SYSTEM_PROMPT_RULES

    catalog_text = format_products_for_prompt(filtered_products)
    logger.info("LLM: System prompt built. Catalog size: %d bytes (filtered %d/%d products)", len(catalog_text), len(filtered_products), len(products))

    return (
        SYSTEM_PROMPT_BASE
        + "\n\nKatalog Produk & Harga:\n"
        + catalog_text
        + "\n"
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
    system_prompt = await _build_system_prompt(user_message)

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

