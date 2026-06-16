"""
LocalLLMClient — production HTTP client for the local LLM API.

Confirmed API contract (validated against live server 2026-06-14):

    POST /api/v1/chat
        Request : {"model": "<key>", "input": "<text>"}
        Response: {"model_instance_id": "...", "response_id": "...",
                   "output": [{"type": "reasoning"|"message", "content": "..."}],
                   "stats": {"input_tokens": N, "total_output_tokens": N,
                             "reasoning_output_tokens": N,
                             "tokens_per_second": F,
                             "time_to_first_token_seconds": F}}

    GET /api/v1/models
        Response: {"models": [...]}

Design decisions:
  - temperature and max_tokens are NOT sent (not confirmed in API schema).
  - system_prompt is prepended inline via TestCase.build_input().
  - Retry applies to network errors and HTTP 5xx/429 only.
  - HTTP 4xx (except 429) raise LLMClientError immediately — no retry.
  - response_id is accepted from the API; UUID fallback is in LLMResponse.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import httpx
from loguru import logger

from src.clients.base_client import (
    RETRYABLE_STATUS_CODES,
    LLMClientError,
    LLMConnectionError,
    LLMEmptyResponseError,
    LLMParseError,
    LLMServerError,
)
from src.schemas.llm_response import LLMResponse
from src.schemas.test_case import TestCase
from src.utils.config import Config
from src.utils.logger import log_llm_event

# Network-level exceptions that always warrant a retry.
_NET_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


class LocalLLMClient:
    """
    Synchronous HTTP client for the local LLM API.

    Thread-safety: httpx.Client is thread-safe for concurrent requests,
    so a single LocalLLMClient instance can be shared across pytest-xdist
    workers running in the same process.  Each worker process gets its own
    instance via the session fixture.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._http = httpx.Client(
            # Per-request timeout is overridden at call time via TestCase.timeout_seconds.
            # This default catches any call that forgets to pass a timeout.
            timeout=httpx.Timeout(
                connect=10.0,
                read=config.timeout_seconds,
                write=10.0,
                pool=5.0,
            ),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

    # ------------------------------------------------------------------ #
    # BaseClient interface
    # ------------------------------------------------------------------ #

    def health_check(self) -> bool:
        """
        Return True if the API is reachable and at least one model is available.
        Never raises — all exceptions are caught and logged.
        """
        try:
            data = self._request("GET", "/api/v1/models")
            models: list[dict[str, Any]] = data.get("models", [])
            if models:
                loaded = [m.get("key", "?") for m in models]
                logger.info(f"[health_check] OK — {len(models)} model(s): {loaded}")
                return True
            logger.warning("[health_check] API reachable but no models available")
            return False
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[health_check] FAILED — {type(exc).__name__}: {exc}")
            return False

    def list_models(self) -> list[dict[str, Any]]:
        """Return raw model descriptors from GET /api/v1/models."""
        data = self._request("GET", "/api/v1/models")
        return data.get("models", [])

    def chat(self, test_case: TestCase) -> LLMResponse:
        """
        POST /api/v1/chat and return a parsed LLMResponse.

        Raises:
            LLMEmptyResponseError : response message block is empty.
            LLMConnectionError    : API unreachable after all retries.
            LLMServerError        : HTTP 5xx / 429 after all retries.
            LLMClientError        : HTTP 4xx (not retried).
            LLMParseError         : response schema mismatch.
        """
        payload: dict[str, Any] = {
            "model": self._config.default_model_key,
            "input": test_case.build_input(),
        }

        log_llm_event(
            "llm_call_start",
            test_id=test_case.test_id,
            model_key=self._config.default_model_key,
        )

        t_start = time.perf_counter()
        raw = self._request(
            "POST",
            "/api/v1/chat",
            json=payload,
            timeout=test_case.timeout_seconds,
        )
        client_latency_ms = (time.perf_counter() - t_start) * 1000

        try:
            response = LLMResponse.from_api_response(raw, client_latency_ms=client_latency_ms)
        except Exception as exc:
            raise LLMParseError(
                f"Failed to parse API response for test_id={test_case.test_id}: {exc}"
            ) from exc

        log_llm_event(
            "llm_call_complete",
            test_id=test_case.test_id,
            model_key=self._config.default_model_key,
            latency_ms=client_latency_ms,
            tokens=response.stats.total_output_tokens,
        )

        if response.is_empty:
            raise LLMEmptyResponseError(
                f"Empty message block received for test_id={test_case.test_id}. "
                f"has_reasoning={response.has_reasoning}, "
                f"raw_output_blocks={len(response.output)}"
            )

        if response.has_reasoning:
            logger.debug(
                f"[chat] test_id={test_case.test_id} "
                f"reasoning_tokens={response.stats.reasoning_output_tokens} "
                f"message_tokens={response.stats.message_output_tokens} "
                f"ratio={response.reasoning_ratio:.0%}"
            )

        return response

    def close(self) -> None:
        """Close the underlying HTTPX connection pool."""
        self._http.close()

    def __enter__(self) -> "LocalLLMClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Internal HTTP layer
    # ------------------------------------------------------------------ #

    def _request(
        self,
        method: str,
        endpoint: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Execute an HTTP request with exponential-backoff retry.

        Retry policy:
          - Network errors (_NET_EXCEPTIONS)        → always retry
          - HTTP 5xx / 429                          → retry
          - HTTP 4xx (excluding 429)                → raise LLMClientError immediately
          - Success (2xx)                           → return parsed body

        Args:
            method   : HTTP verb ("GET", "POST", …)
            endpoint : path relative to base_url (e.g. "/api/v1/chat")
            timeout  : per-request read timeout in seconds; defaults to config value
            **kwargs : forwarded to httpx.Client.request (e.g. json=payload)
        """
        url = self._config.base_url.rstrip("/") + endpoint
        request_timeout = httpx.Timeout(
            connect=10.0,
            read=timeout or self._config.timeout_seconds,
            write=10.0,
            pool=5.0,
        )
        max_attempts = self._config.max_retries + 1
        last_exc: Exception = RuntimeError("Retry loop exited without result")

        for attempt in range(max_attempts):
            try:
                r = self._http.request(method, url, timeout=request_timeout, **kwargs)
                return self._parse_response(r)

            except _NET_EXCEPTIONS as exc:
                last_exc = LLMConnectionError(
                    f"{method} {endpoint} network error: {type(exc).__name__}: {exc}"
                )

            except LLMServerError as exc:
                last_exc = exc

            except LLMClientError:
                raise  # 4xx client errors are not retried

            if attempt < max_attempts - 1:
                wait = min(
                    self._config.retry_max_wait_seconds,
                    self._config.retry_min_wait_seconds
                    * (self._config.retry_backoff_factor ** attempt),
                )
                logger.warning(
                    f"[{method} {endpoint}] attempt {attempt + 1}/{max_attempts} failed "
                    f"({type(last_exc).__name__}). Retrying in {wait:.1f}s…"
                )
                time.sleep(wait)
            else:
                logger.error(
                    f"[{method} {endpoint}] all {max_attempts} attempts exhausted. "
                    f"Last error: {last_exc}"
                )

        raise last_exc

    @staticmethod
    def _parse_response(response: httpx.Response) -> dict[str, Any]:
        """
        Parse an httpx.Response into a dict.

        Handles both JSON and plain-text bodies — some error responses from the
        local API arrive as plain text rather than JSON.

        Raises:
            LLMClientError : HTTP 4xx (except 429)
            LLMServerError : HTTP 5xx or 429
        """
        # Attempt JSON parse regardless of status code — error bodies are often JSON.
        try:
            body: dict[str, Any] = response.json()
        except Exception:
            body = {"error": response.text or "(empty body)"}

        if response.is_success:
            return body

        # Extract a human-readable error message from the body.
        error_msg = (
            body.get("error")
            or body.get("message")
            or body.get("detail")
            or response.text
            or f"HTTP {response.status_code}"
        )

        status = response.status_code

        if status in RETRYABLE_STATUS_CODES:
            raise LLMServerError(
                f"API server error {status}: {error_msg}", status_code=status
            )

        raise LLMClientError(
            f"API client error {status}: {error_msg}", status_code=status
        )
