"""Centralized logging setup for the WhatsApp chatbot.

Provides a single ``setup_logging()`` entry-point that:

- Configures the root logger once (idempotent).
- Renders human-friendly text logs by default, or JSON when ``LOG_FORMAT=json``.
- Reads level/format from environment variables so deployments can tune
  verbosity without code changes.
- Silences known-chatty third-party loggers.
- Exposes a ``phone`` context variable that automatically appears in every
  log record while it is set (handy for tracing a single conversation).
"""

from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
import os
import sys
from contextlib import contextmanager
from typing import Iterator

# -- Public context variable ---------------------------------------------
_phone_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "chatbot_phone", default=None
)


def set_phone(phone: str | None) -> None:
    """Set the phone context for the current asyncio task (no reset).

    Useful inside background tasks where the work outlives a ``with`` block.
    Each asyncio task has its own contextvar copy, so callers do not need to
    clean up explicitly.
    """
    _phone_var.set(phone or None)

@contextmanager
def phone_context(phone: str | None) -> Iterator[None]:
    """Tag every log record emitted in this block with ``phone``."""
    token = _phone_var.set(phone or None)
    try:
        yield
    finally:
        _phone_var.reset(token)


class _ContextFilter(logging.Filter):
    """Inject contextvars into each log record."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.phone = _phone_var.get() or "-"
        return True


class _TextFormatter(logging.Formatter):
    """Compact human-readable formatter using clean unicode separators."""

    SEP = " | "

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s" + self.SEP
            + "%(levelname)-7s" + self.SEP
            + "%(name)s" + self.SEP
            + "phone=%(phone)s" + self.SEP
            + "%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


class _JsonFormatter(logging.Formatter):
    """JSON formatter suitable for log aggregators."""

    _STD_KEYS = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "phone": getattr(record, "phone", "-"),
            "msg": record.getMessage(),
        }
        # Surface any structured extras the caller passed in.
        for key, value in record.__dict__.items():
            if key in self._STD_KEYS or key.startswith("_") or key in payload:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


# Loggers we want to keep quiet by default. Adjust here, not at call sites.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "openai",
    "openai._base_client",
    "asyncio",
)

_CONFIGURED = False


def setup_logging(
    *,
    level: str | None = None,
    fmt: str | None = None,
    force: bool = False,
) -> None:
    """Configure the root logger. Safe to call multiple times.

    Args:
        level: Minimum level (``DEBUG``, ``INFO``, ``WARNING``, ...).
            Falls back to ``LOG_LEVEL`` env, then ``INFO``.
        fmt: Either ``text`` or ``json``. Falls back to ``LOG_FORMAT`` env,
            then ``text``.
        force: When True, re-runs configuration even if already initialized.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    resolved_level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    resolved_fmt = (fmt or os.getenv("LOG_FORMAT") or "text").lower()

    formatter: logging.Formatter
    if resolved_fmt == "json":
        formatter = _JsonFormatter()
    else:
        formatter = _TextFormatter()

    # stdout so PaaS log collectors pick it up cleanly.
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(_ContextFilter())

    root = logging.getLogger()
    # Replace any pre-existing handlers (e.g. uvicorn's defaults) so we have
    # a single, predictable output stream.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(resolved_level)

    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Align uvicorn's own loggers with our handler so the format stays consistent.
    for uv_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(uv_name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True
        uv_logger.setLevel(resolved_level)

    _CONFIGURED = True
    logging.getLogger(__name__).debug(
        "Logging initialized level=%s format=%s", resolved_level, resolved_fmt
    )