"""
OllamaClient — synchronous HTTP client for any OpenAI-compatible LLM server.

Works with:
  - Ollama    : BASE_URL=http://localhost:11434/v1
  - LM Studio : BASE_URL=http://localhost:1234/v1
  - llama.cpp : BASE_URL=http://localhost:8080/v1

API contract (OpenAI-compatible):

    POST /chat/completions
        Request:
            {
              "model": "gemma4:e2b",
              "messages": [
                {"role": "system", "content": "..."},   # optional
                {"role": "user",   "content": "..."}
              ]
            }
        Response:
            {
              "id": "chatcmpl-xxx",
              "model": "gemma4:e2b",
              "choices": [{"message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
              "usage": {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}
            }

    GET /models
        Response: {"object": "list", "data": [{"id": "gemma4:e2b"}, ...]}

Design decisions:
  - Response is normalized into the same LM Studio-shaped dict that
    LLMResponse.from_api_response() already knows how to parse.
    This keeps the schema layer unchanged for both backends.
  - system_prompt and prompt are sent as separate roles in messages[],
    which is the correct OpenAI-compatible format.
  - Retry policy mirrors LocalLLMClient exactly.
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

_NET_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


class OllamaClient:
    """
    Synchronous HTTP client for the OpenAI-compatible API.

    Thread-safety: httpx.Client is thread-safe, so a single OllamaClient
    instance can be shared across the test session.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._http = httpx.Client(
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
        Return True if the API is reachable and at least one model is listed.
        Never raises.
        """
        try:
            data = self._request("GET", "/models")
            models: list[dict[str, Any]] = data.get("data", [])
            if models:
                ids = [m.get("id", "?") for m in models]
                logger.info(f"[health_check] OK — {len(models)} model(s): {ids}")
                return True
            logger.warning("[health_check] API reachable but no models listed")
            return False
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[health_check] FAILED — {type(exc).__name__}: {exc}")
            return False

    def list_models(self) -> list[dict[str, Any]]:
        """Return model descriptors from GET /models."""
        data = self._request("GET", "/models")
        return data.get("data", [])

    def chat(self, test_case: TestCase) -> LLMResponse:
        """
        POST /chat/completions and return a parsed LLMResponse.

        system_prompt and prompt are sent as separate message roles,
        which is the semantically correct form for the OpenAI format.
        """
        messages: list[dict[str, str]] = []
        if test_case.system_prompt:
            messages.append({"role": "system", "content": test_case.system_prompt})
        messages.append({"role": "user", "content": test_case.prompt})

        payload: dict[str, Any] = {
            "model": self._config.default_model_key,
            "messages": messages,
        }

        log_llm_event(
            "llm_call_start",
            test_id=test_case.test_id,
            model_key=self._config.default_model_key,
        )

        t_start = time.perf_counter()
        raw = self._request(
            "POST",
            "/chat/completions",
            json=payload,
            timeout=test_case.timeout_seconds,
        )
        client_latency_ms = (time.perf_counter() - t_start) * 1000

        try:
            normalized = self._normalize(raw)
            response = LLMResponse.from_api_response(normalized, client_latency_ms=client_latency_ms)
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
                f"Empty message block received for test_id={test_case.test_id}."
            )

        return response

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Response normalisation
    # ------------------------------------------------------------------ #

    def _normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        """
        Convert an OpenAI-compatible response into the LM Studio-shaped dict
        that LLMResponse.from_api_response() already knows how to parse.

        This keeps all schema/evaluation code unchanged between backends.
        """
        choices: list[dict[str, Any]] = raw.get("choices", [])
        content = ""
        if choices:
            content = choices[0].get("message", {}).get("content", "")

        usage: dict[str, Any] = raw.get("usage", {})

        return {
            "model_instance_id": raw.get("model", self._config.default_model_key),
            "response_id": raw.get("id", ""),
            "output": [{"type": "message", "content": content}],
            "stats": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "total_output_tokens": usage.get("completion_tokens", 0),
                "reasoning_output_tokens": 0,
                "tokens_per_second": 0.0,
                "time_to_first_token_seconds": 0.0,
            },
            "_raw_openai": raw,
        }

    # ------------------------------------------------------------------ #
    # Internal HTTP layer — identical retry policy to LocalLLMClient
    # ------------------------------------------------------------------ #

    def _request(
        self,
        method: str,
        endpoint: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
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
                raise

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
        try:
            body: dict[str, Any] = response.json()
        except Exception:
            body = {"error": response.text or "(empty body)"}

        if response.is_success:
            return body

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
