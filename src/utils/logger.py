from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def configure_logger(log_level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """
    Configure Loguru for the framework session.

    Call once from conftest.py session setup.  All subsequent imports of
    ``from loguru import logger`` share the same configured sink(s).

    enqueue=True makes both sinks thread-safe, which is required when
    pytest-xdist workers write concurrently.
    """
    logger.remove()

    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
        enqueue=True,
    )

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            enqueue=True,
        )


def log_llm_event(
    event: str,
    test_id: str = "",
    model_key: str = "",
    latency_ms: float = 0.0,
    tokens: int = 0,
    **extra: object,
) -> None:
    """Emit a structured log event for LLM interactions."""
    logger.bind(
        event=event,
        test_id=test_id,
        model_key=model_key,
        latency_ms=round(latency_ms, 2),
        tokens=tokens,
        **extra,
    ).info(f"[{event}] test={test_id} model={model_key} latency={latency_ms:.1f}ms tokens={tokens}")
