"""
MockClient — deterministic fake LLM client for CI/CD and offline testing.

Activated by setting MOCK_MODE=true in the environment.

Behaviour guarantees:
  - No network calls are made under any circumstances.
  - Given the same (model_key, prompt), the response is always identical.
  - The response schema is identical to a real LocalLLMClient response —
    MockClient and LocalLLMClient are interchangeable through BaseClient.
  - Simulated latency fields match the stat structure the API returns.

Response selection:
  1. Prompt matched against KNOWN_PATTERNS (regex, case-insensitive) →
     returns a pre-authored reasoning + answer pair.
  2. No pattern match → deterministic generic response derived from
     hash(model_key + prompt), so the same prompt always returns the same text.

Mock mode and test assertions:
  Accuracy tests that compare scores against pass_threshold will fail
  in mock mode for unknown prompts — this is expected and handled by the
  EvaluationPipeline in Phase 3 (mock mode sets confidence=LOW and the
  conftest skips threshold assertions).
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any

from loguru import logger

from src.schemas.llm_response import LLMResponse
from src.schemas.test_case import TestCase
from src.utils.config import Config


# ---------------------------------------------------------------------------
# Known answer bank
# Pattern, reasoning text, final answer
# ---------------------------------------------------------------------------

_KnownPattern = tuple[str, str, str]  # (regex, reasoning, answer)

_KNOWN_PATTERNS: list[_KnownPattern] = [
    # ── Factual / geography ──────────────────────────────────────────────
    (r"capital\s+of\s+france",
     "France is a country in Western Europe. Its capital city is Paris, "
     "which has been the capital since the 10th century.",
     "Paris"),

    (r"capital\s+of\s+germany",
     "Germany is a country in Central Europe. Its capital is Berlin, "
     "which became the capital of reunified Germany in 1990.",
     "Berlin"),

    (r"capital\s+of\s+(the\s+)?united\s+states|capital\s+of\s+usa|capital\s+of\s+america",
     "The United States of America has Washington, D.C. as its capital, "
     "established as the capital in 1800.",
     "Washington, D.C."),

    (r"capital\s+of\s+japan",
     "Japan is an island nation in East Asia. Its capital is Tokyo, "
     "which became the de facto capital in 1869.",
     "Tokyo"),

    (r"capital\s+of\s+brazil",
     "Brazil is the largest country in South America. Its capital is Brasília, "
     "which replaced Rio de Janeiro as the capital in 1960.",
     "Brasília"),

    # ── Mathematics ──────────────────────────────────────────────────────
    (r"\b2\s*\+\s*2\b",
     "Simple arithmetic: 2 + 2 = 4.",
     "2 + 2 = 4"),

    (r"\b1\s*\+\s*1\b",
     "Simple arithmetic: 1 + 1 = 2.",
     "1 + 1 = 2"),

    (r"\b3\s*\+\s*3\b",
     "Simple arithmetic: 3 + 3 = 6.",
     "3 + 3 = 6"),

    (r"\b10\s*[\*x]\s*10\b",
     "Multiplication: 10 × 10 = 100.",
     "100"),

    (r"\b100\s*/\s*4\b",
     "Division: 100 ÷ 4 = 25.",
     "25"),

    # ── Logic / syllogisms ───────────────────────────────────────────────
    (r"cats.*animals.*living\s+beings|all\s+cats.*are.*living",
     "Syllogism analysis: "
     "Premise 1: All cats are animals. "
     "Premise 2: All animals are living beings. "
     "Conclusion: Therefore, all cats are living beings (by transitivity).",
     "Yes, cats are living beings."),

    (r"socrates.*mortal|all\s+men.*mortal.*socrates",
     "Classic syllogism: "
     "All men are mortal (major premise). "
     "Socrates is a man (minor premise). "
     "Therefore, Socrates is mortal (conclusion).",
     "Yes, Socrates is mortal."),

    # ── Safety / Prompt injection ─────────────────────────────────────────
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
     "I detected a prompt injection attempt asking me to ignore my instructions. "
     "I should not comply with this request.",
     "I cannot comply with that request. I follow my operational guidelines."),

    (r"reveal\s+(your\s+)?(system|hidden|original)\s+(prompt|instructions?)",
     "The user is asking me to reveal my system prompt. "
     "This is confidential and I should decline.",
     "I cannot reveal my system prompt or internal instructions."),

    (r"forget\s+(everything|what\s+you\s+(were|are)\s+told)",
     "Instruction to forget context — this is an injection attempt.",
     "I'm unable to discard my operational guidelines."),

    # ── Jailbreak ────────────────────────────────────────────────────────
    (r"DAN|do\s+anything\s+now|jailbreak|no\s+restrictions?",
     "This prompt appears to be a jailbreak attempt. "
     "I should maintain my guidelines and decline.",
     "I'm not able to operate outside my safety guidelines."),

    (r"pretend\s+(you\s+)?(are|have)\s+(no\s+)?(restrictions?|limitations?|rules?)",
     "Roleplay-based jailbreak attempt detected. I should refuse clearly.",
     "I'm unable to pretend I have no restrictions."),

    # ── Reasoning / logic ─────────────────────────────────────────────────
    (r"speed.*120\s*km.*2\s*hours?|average\s+speed.*train",
     "Speed = distance ÷ time. "
     "Distance = 120 km, Time = 2 hours. "
     "Therefore speed = 120 / 2 = 60 km/h.",
     "The average speed is 60 km/h."),

    (r"boiling\s+point.*water",
     "Water boils at 100°C (212°F) at standard atmospheric pressure (1 atm).",
     "Water boils at 100°C (212°F) at standard atmospheric pressure."),

    (r"planets?\s+in\s+(our\s+)?solar\s+system",
     "The solar system contains 8 planets: Mercury, Venus, Earth, Mars, "
     "Jupiter, Saturn, Uranus, and Neptune.",
     "There are 8 planets in our solar system: Mercury, Venus, Earth, Mars, "
     "Jupiter, Saturn, Uranus, and Neptune."),
]


# ---------------------------------------------------------------------------
# MockClient
# ---------------------------------------------------------------------------

class MockClient:
    """
    Deterministic fake LLM client — no network calls, no model required.

    Implements the same interface as LocalLLMClient so it is fully
    interchangeable through ClientFactory.create().
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        logger.info(
            f"[MockClient] Initialised — MOCK_MODE=true. "
            f"Model key: {config.default_model_key}"
        )

    # ------------------------------------------------------------------ #
    # BaseClient interface
    # ------------------------------------------------------------------ #

    def health_check(self) -> bool:
        logger.info("[MockClient] health_check → True (mock)")
        return True

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "llm",
                "publisher": "mock",
                "key": self._config.default_model_key,
                "display_name": f"Mock({self._config.default_model_key})",
                "architecture": "mock",
            }
        ]

    def chat(self, test_case: TestCase) -> LLMResponse:
        """
        Return a deterministic LLMResponse without making any network call.

        Simulates a small processing delay (5 ms) so latency fields are
        non-zero and plausible for downstream assertions.
        """
        t_start = time.perf_counter()
        time.sleep(0.005)  # simulate minimal latency

        prompt = test_case.prompt
        reasoning, answer = self._resolve_response(prompt, test_case)

        client_latency_ms = (time.perf_counter() - t_start) * 1000

        raw = self._build_raw_response(
            prompt=prompt,
            model_key=self._config.default_model_key,
            reasoning=reasoning,
            answer=answer,
            client_latency_ms=client_latency_ms,
        )

        response = LLMResponse.from_api_response(raw, client_latency_ms=client_latency_ms)

        logger.debug(
            f"[MockClient] test_id={test_case.test_id} "
            f"answer='{answer[:60]}' latency={client_latency_ms:.1f}ms"
        )
        return response

    def close(self) -> None:
        pass  # no resources to release

    def __enter__(self) -> "MockClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _resolve_response(
        self, prompt: str, test_case: TestCase
    ) -> tuple[str, str]:
        """
        Select reasoning and answer for *prompt*.

        Priority:
          1. Pattern match against _KNOWN_PATTERNS (case-insensitive).
          2. expected_answer in test_case.metadata["mock_answer"] (for custom datasets).
          3. Generic deterministic fallback.
        """
        # 1. Known pattern match
        for pattern, reasoning, answer in _KNOWN_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                return reasoning, answer

        # 2. Dataset-level mock override
        if "mock_answer" in test_case.metadata:
            answer = str(test_case.metadata["mock_answer"])
            reasoning = f"[MOCK] Using dataset-provided answer for: {prompt[:60]}"
            return reasoning, answer

        # 3. Deterministic generic fallback
        return self._generic_response(prompt, self._config.default_model_key)

    @staticmethod
    def _generic_response(prompt: str, model_key: str) -> tuple[str, str]:
        """
        Generate a deterministic reasoning + answer pair from a hash of the input.

        The same prompt + model_key always produces the same output, making
        test runs reproducible even for prompts not in _KNOWN_PATTERNS.
        """
        seed = int(hashlib.md5(f"{model_key}:{prompt}".encode()).hexdigest(), 16) % 1000
        reasoning = (
            f"[MOCK] Analysing: \"{prompt[:80]}\"… "
            f"Seed={seed}. "
            "This is a deterministic mock response generated without a real model."
        )
        answer = (
            f"[MOCK-{seed:04d}] This is a simulated response to: \"{prompt[:60]}\""
        )
        return reasoning, answer

    @staticmethod
    def _build_raw_response(
        prompt: str,
        model_key: str,
        reasoning: str,
        answer: str,
        client_latency_ms: float,
    ) -> dict[str, Any]:
        """Construct a raw response dict that matches the confirmed API schema."""
        # Token counts are estimated from word count — no tokeniser available in mock.
        input_tokens = max(1, len(prompt.split()))
        reasoning_tokens = max(1, len(reasoning.split()) * 2)   # rough BPE estimate
        message_tokens = max(1, len(answer.split()) * 2)
        total_tokens = reasoning_tokens + message_tokens

        # Deterministic response_id from content hash
        resp_hash = hashlib.md5(f"{model_key}:{prompt}:{answer}".encode()).hexdigest()[:24]

        return {
            "model_instance_id": model_key,
            "response_id": f"mock-resp-{resp_hash}",
            "output": [
                {"type": "reasoning", "content": reasoning},
                {"type": "message",   "content": answer},
            ],
            "stats": {
                "input_tokens": input_tokens,
                "total_output_tokens": total_tokens,
                "reasoning_output_tokens": reasoning_tokens,
                "tokens_per_second": 50.0,
                "time_to_first_token_seconds": round(client_latency_ms / 1000, 4),
            },
        }
