"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration backed by .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # ── Nebius AI Studio ─────────────────────────────────────────
    nebius_api_key: str
    nebius_model: str = "Qwen/Qwen3-14B"
    nebius_vision_model: str = "Qwen/Qwen2.5-VL-72B-Instruct"

    # ── WAHA (WhatsApp HTTP API) ─────────────────────────────────
    waha_base_url: str = "https://pseudolegal-chelsey-tacitly.ngrok-free.dev"
    waha_session: str = "default"
    waha_api_key: str = ""

    # ── Supabase ─────────────────────────────────────────────────
    supabase_url: str = ""
    supabase_service_key: str = ""

    # Web Push Notifications
    vapid_private_key: str = ""
    vapid_claims_subject: str = "mailto:admin@tokoteladan.local"
    push_notify_all_subscribers: bool = True
    closing_deal_push_secret: str = ""

    # ── App ──────────────────────────────────────────────────────
    max_history_length: int = 20  # max messages kept per user
    max_images_per_request: int = 5  # max images processed per webhook batch


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()  # type: ignore[call-arg]
