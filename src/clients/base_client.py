"""
Base client contract and shared exception hierarchy for all LLM clients.

Hierarchy:
    BaseClient (ABC)
        ├── LocalLLMClient   — real HTTP calls to the local API
        └── MockClient       — deterministic fake responses for CI/CD
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Root exception for all client errors."""


class LLMConnectionError(LLMError):
    """API is unreachable — network failure, timeout, or DNS error."""


class LLMClientError(LLMError):
    """
    Non-retryable API error (HTTP 4xx excluding 429).
    The request itself is malformed or unauthorised.
    """
    def __init__(self, message: str, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMServerError(LLMError):
    """
    Retryable API error (HTTP 5xx or 429 Too Many Requests).
    The server is temporarily unavailable or overloaded.
    """
    def __init__(self, message: str, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMEmptyResponseError(LLMError):
    """
    API returned a structurally valid response but with empty message content.
    Treated as a hard failure — not retried by default.
    """


class LLMParseError(LLMError):
    """
    API response could not be parsed into LLMResponse.
    Indicates an unexpected schema change in the server.
    """


# ---------------------------------------------------------------------------
# Retryable HTTP status codes (used by LocalLLMClient)
# ---------------------------------------------------------------------------

RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# Network-level exceptions that always warrant a retry attempt
RETRYABLE_NETWORK_EXCEPTIONS: tuple[type[Exception], ...] = ()  # filled lazily below


def _build_retryable_network_exceptions() -> tuple[type[Exception], ...]:
    """Return retryable httpx exception types, importing lazily."""
    try:
        import httpx  # noqa: PLC0415
        return (
            httpx.TimeoutException,     # ConnectTimeout, ReadTimeout, WriteTimeout, PoolTimeout
            httpx.NetworkError,         # ConnectError, ReadError, WriteError, CloseError
            httpx.RemoteProtocolError,  # server sent invalid HTTP
        )
    except ImportError:
        return (OSError, ConnectionError)


# ---------------------------------------------------------------------------
# Abstract base client
# ---------------------------------------------------------------------------

class BaseClient(ABC):
    """
    Contract that every LLM client implementation must satisfy.

    All public methods are synchronous.  Async support is deferred to V2.
    """

    @abstractmethod
    def health_check(self) -> bool:
        """
        Return True if the API is reachable and at least one model is ready.
        Must never raise — swallow all exceptions and return False instead.
        """

    @abstractmethod
    def list_models(self) -> list[dict[str, Any]]:
        """
        Return the raw list of model descriptors from the API.
        Raises LLMConnectionError or LLMServerError on failure.
        """

    @abstractmethod
    def chat(self, test_case: Any) -> Any:
        """
        Send a chat request derived from *test_case* and return an LLMResponse.

        Implementations must:
          - call test_case.build_input() to compose the 'input' field
          - measure and populate client_latency_ms on the returned LLMResponse
          - raise LLMEmptyResponseError if the response message block is empty
          - raise LLMConnectionError on network failure
          - raise LLMServerError on retryable HTTP errors
          - raise LLMClientError on non-retryable HTTP errors
        """

    def close(self) -> None:
        """Release held resources.  Override in implementations that own connections."""

    def __enter__(self) -> "BaseClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


