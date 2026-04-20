"""Product catalog service — fetch products from Supabase pos_products table."""

from __future__ import annotations

import logging
import time

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

TABLE = "pos_products"

# ── Simple in-memory cache ──────────────────────────────────────
_cache: list[dict] | None = None
_cache_ts: float = 0
CACHE_TTL = 300  # 5 minutes


def _headers() -> dict[str, str]:
    """Build Supabase REST API headers."""
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


async def fetch_products() -> list[dict]:
    """Fetch all products from Supabase, with a 5-minute cache.

    Returns:
        List of product dicts with keys: name, sku, price, unit, categoryId, material, stock.
    """
    global _cache, _cache_ts

    # Return cached data if still fresh
    if _cache is not None and (time.time() - _cache_ts) < CACHE_TTL:
        logger.info("Product fetch: Using cached products (count: %d)", len(_cache))
        return _cache

    logger.info("Product fetch: Starting request to Supabase...")
    params = {
        "select": "name,sku,price,unit,categoryId,material,stock",
        "order": "categoryId.asc,name.asc",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(_base_url(), headers=_headers(), params=params)
            if resp.status_code >= 400:
                logger.error("Product fetch failed: HTTP %s - %s", resp.status_code, resp.text)
                return _cache or []

            products = resp.json()
            _cache = products
            _cache_ts = time.time()
            logger.info("Product fetch SUCCESS: Retrieved %d products from Supabase", len(products))
            return products
    except Exception as e:
        logger.exception("Product fetch ERROR: An exception occurred during fetch_products. Exception: %s", str(e))
        return _cache or []


def format_products_for_prompt(products: list[dict]) -> str:
    """Format product list into a compressed string to save tokens."""
    if not products:
        return "Katalog kosong"

    lines = ["Kat|Nama|Harga|Bahan|SKU|Stok"]
    for p in products:
        cat = p.get("categoryId") or "-"
        name = p.get("name", "?")
        price = f"Rp{p.get('price',0):.0f}/{p.get('unit','pcs')}"
        mat = p.get("material", "") or "-"
        sku = p.get("sku", "") or "-"
        stok = p.get("stock", 0)
        stok_str = "HABIS" if stok is not None and stok <= 0 else str(stok)
        
        lines.append(f"{cat}|{name}|{price}|{mat}|{sku}|{stok_str}")

    return "\n".join(lines)
