"""Product catalog service - Supabase pos_products lookup with bounded payloads.

Replaces the full-catalog fetch with a small summary (categories + price ranges)
plus targeted keyword search. This keeps prompt sizes flat as the catalog grows.
"""

from __future__ import annotations

import logging
import re
import time

from app.config import get_settings
from app.services.http_client import get_supabase_client

logger = logging.getLogger(__name__)

TABLE = "pos_products"

# -- Catalog summary cache (small, refreshed every 10 min) ----------
_summary_cache: dict | None = None
_summary_cache_ts: float = 0.0
SUMMARY_CACHE_TTL = 600

# -- Per-query product search cache (bounded LRU-ish) ---------------
_search_cache: dict[str, tuple[float, list[dict]]] = {}
SEARCH_CACHE_TTL = 120
SEARCH_CACHE_MAX = 256

GENERIC_QUERY_WORDS = {
    "admin", "ada", "aja", "assalam", "assalamualaikum", "atau", "bagaimana",
    "beli", "berapa", "bisa", "bro", "buk", "cari", "dari", "dengan", "gan",
    "gimana", "hai", "hallo", "halo", "harga", "hello", "info", "ini", "itu",
    "kak", "kakak", "kalo", "kalau", "malam", "mas", "mau", "mbak", "min",
    "minta", "pada", "pagi", "pak", "saja", "siang", "sis", "sore", "tolong",
    "untuk", "yang",
}

# Indonesian/English synonyms -> canonical search tokens that exist in our catalog.
SYNONYMS: dict[str, list[str]] = {
    "banner": ["spanduk", "flexi", "banner"],
    "spanduk": ["spanduk", "flexi"],
    "flexi": ["flexi"],
    "sticker": ["stiker"],
    "stiker": ["stiker"],
    "vinyl": ["vinyl"],
    "print": ["cetak"],
    "cetak": ["cetak"],
    "outdoor": ["outdoor"],
    "indoor": ["indoor"],
    "atk": ["atk"],
    "kantor": ["atk"],
    "stationary": ["atk"],
    "stationery": ["atk"],
    "pulpen": ["pulpen", "pen"],
    "pen": ["pulpen", "pen"],
    "buku": ["buku"],
    "kertas": ["kertas", "paper"],
    "paper": ["kertas", "paper"],
    "albatros": ["albatros"],
    "pvc": ["pvc"],
    "luster": ["luster"],
    "oneway": ["one way"],
    "onewayvision": ["one way"],
    "poster": ["poster", "albatros", "luster"],
    "brosur": ["brosur", "kertas"],
    "logo": ["stiker", "kertas"],
    "xbanner": ["banner", "luster"],
    "rollup": ["banner"],
    "rollbanner": ["banner"],
}

# Short tokens that must still be considered (sizes / abbreviations).
ALLOWED_SHORT_TOKENS = {"a3", "a4", "a5", "a6", "b5", "b6", "f4", "pvc", "atk"}


def _headers() -> dict[str, str]:
    settings = get_settings()
    return {
        "apikey": settings.supabase_service_key,
        "Authorization": f"Bearer {settings.supabase_service_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _base_url() -> str:
    settings = get_settings()
    return f"{settings.supabase_url}/rest/v1/{TABLE}"


def _normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.lower())


def _expand_keywords(words: list[str]) -> list[str]:
    """Apply the synonym map and dedupe while preserving priority order."""
    expanded: list[str] = []
    seen: set[str] = set()
    for word in words:
        norm = _normalize_token(word)
        if not norm:
            continue
        for cand in SYNONYMS.get(norm, [norm]):
            cand = cand.strip().lower()
            if cand and cand not in seen:
                seen.add(cand)
                expanded.append(cand)
    return expanded


def _message_keywords(message: str, limit: int = 5) -> list[str]:
    """Extract meaningful tokens from a free-form message."""
    raw_words = re.findall(r"[a-z0-9]+", message.lower())
    candidates: list[str] = []
    for word in sorted(raw_words, key=len, reverse=True):
        if word in GENERIC_QUERY_WORDS or word in candidates:
            continue
        if len(word) < 3 and word not in ALLOWED_SHORT_TOKENS:
            continue
        candidates.append(word)
        if len(candidates) >= limit:
            break
    expanded = _expand_keywords(candidates)
    # Cap expanded list so the OR filter does not balloon.
    return expanded[: limit * 2]


# -- Catalog summary -----------------------------------------------
async def fetch_catalog_summary() -> dict:
    """Return a compact summary of the catalog (categories + price ranges).

    The payload stays tiny regardless of catalog size, so we cache aggressively.
    """
    global _summary_cache, _summary_cache_ts

    if _summary_cache is not None and (time.time() - _summary_cache_ts) < SUMMARY_CACHE_TTL:
        return _summary_cache

    params = {
        "select": "categoryId,price",
        "order": "categoryId.asc",
    }
    try:
        client = get_supabase_client()
        resp = await client.get(_base_url(), headers=_headers(), params=params)
        if resp.status_code >= 400:
            logger.error("Catalog summary failed: HTTP %s - %s", resp.status_code, resp.text[:200])
            return _summary_cache or {"categories": [], "total": 0}
        rows = resp.json()
    except Exception as e:
        logger.exception("Catalog summary ERROR: %s", str(e))
        return _summary_cache or {"categories": [], "total": 0}

    by_cat: dict[str, dict] = {}
    for row in rows:
        cat = row.get("categoryId") or "Lainnya"
        price = row.get("price") or 0
        bucket = by_cat.setdefault(cat, {"count": 0, "min": price, "max": price})
        bucket["count"] += 1
        if price:
            if price < bucket["min"] or bucket["min"] == 0:
                bucket["min"] = price
            if price > bucket["max"]:
                bucket["max"] = price

    summary = {
        "total": len(rows),
        "categories": [
            {"name": name, **stats}
            for name, stats in sorted(by_cat.items(), key=lambda kv: kv[0])
        ],
    }
    _summary_cache = summary
    _summary_cache_ts = time.time()
    logger.info(
        "Catalog summary cached: %d categories, %d products total",
        len(summary["categories"]),
        summary["total"],
    )
    return summary


def format_catalog_summary(summary: dict) -> str:
    """Render the summary dict as a short text block."""
    cats = summary.get("categories") or []
    if not cats:
        return "Katalog kosong"
    lines = [f"Total {summary.get('total', 0)} produk. Kategori:"]
    for c in cats:
        lo = int(c.get("min") or 0)
        hi = int(c.get("max") or 0)
        lines.append(
            f"- {c['name']}: {c['count']} produk, Rp{lo:,}-Rp{hi:,}".replace(",", ".")
        )
    return "\n".join(lines)


# -- Targeted product search ---------------------------------------
def _search_cache_get(key: str) -> list[dict] | None:
    entry = _search_cache.get(key)
    if not entry:
        return None
    ts, products = entry
    if time.time() - ts > SEARCH_CACHE_TTL:
        _search_cache.pop(key, None)
        return None
    return products


def _search_cache_set(key: str, products: list[dict]) -> None:
    if len(_search_cache) >= SEARCH_CACHE_MAX:
        oldest_key = min(_search_cache.items(), key=lambda kv: kv[1][0])[0]
        _search_cache.pop(oldest_key, None)
    _search_cache[key] = (time.time(), products)


async def fetch_matching_products(user_message: str, limit: int = 40) -> list[dict]:
    """Return a small slice of the catalog relevant to the user's message."""
    keywords = _message_keywords(user_message)
    if not keywords:
        return []

    cache_key = "|".join(sorted(keywords)) + f"#{limit}"
    cached = _search_cache_get(cache_key)
    if cached is not None:
        logger.info("Product search: cache hit (%d products) key=%s", len(cached), cache_key)
        return cached

    or_filters: list[str] = []
    for keyword in keywords:
        safe = _normalize_token(keyword)
        if not safe:
            continue
        wildcard = f"*{safe}*"
        or_filters.extend(
            [
                f"name.ilike.{wildcard}",
                f"categoryId.ilike.{wildcard}",
                f"material.ilike.{wildcard}",
            ]
        )
    if not or_filters:
        return []

    params = {
        "select": "name,sku,price,unit,categoryId,material,stock",
        "or": f"({','.join(or_filters)})",
        "order": "categoryId.asc,name.asc",
        "limit": str(limit),
    }

    try:
        client = get_supabase_client()
        resp = await client.get(_base_url(), headers=_headers(), params=params)
        if resp.status_code >= 400:
            logger.error("Product search failed: HTTP %s - %s", resp.status_code, resp.text[:200])
            return []
        products = resp.json()
        _search_cache_set(cache_key, products)
        logger.info("Product search SUCCESS: %d products keywords=%s", len(products), keywords)
        return products
    except Exception as e:
        logger.exception("Product search ERROR: %s", str(e))
        return []


def format_products_for_prompt(products: list[dict]) -> str:
    """Compact text representation of a product list for LLM prompts."""
    if not products:
        return "Katalog kosong"

    lines = ["Kat|Nama|Harga|Bahan|SKU|Stok"]
    for p in products:
        cat = p.get("categoryId") or "-"
        name = p.get("name", "?")
        price = f"Rp{p.get('price', 0):.0f}/{p.get('unit', 'pcs')}"
        mat = p.get("material") or "-"
        sku = p.get("sku") or "-"
        stok = p.get("stock", 0)
        stok_str = "HABIS" if stok is not None and stok <= 0 else str(stok)
        lines.append(f"{cat}|{name}|{price}|{mat}|{sku}|{stok_str}")
    return "\n".join(lines)