from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """
    Central configuration loaded from environment variables and .env file.
    All fields map to uppercase env vars (e.g. base_url → BASE_URL).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # API Connection
    # ------------------------------------------------------------------ #
    base_url: str = "http://192.168.15.103:1234"
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0)
    retry_backoff_factor: float = Field(default=2.0, gt=0.0)
    retry_min_wait_seconds: float = Field(default=1.0, gt=0.0)
    retry_max_wait_seconds: float = Field(default=30.0, gt=0.0)

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    default_model_key: str = "google/gemma-4-e2b"

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    # Keep low: local LLM has no request queue — parallel overload causes errors.
    max_parallel_workers: int = Field(default=2, ge=1)

    # ------------------------------------------------------------------ #
    # Evaluation thresholds
    # ------------------------------------------------------------------ #
    default_pass_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    fuzzy_match_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    report_dir: Path = Path("reports")
    enable_html_report: bool = True
    enable_json_report: bool = True
    enable_csv_report: bool = True



@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the singleton Config instance (cached after first call)."""
    return Config()
