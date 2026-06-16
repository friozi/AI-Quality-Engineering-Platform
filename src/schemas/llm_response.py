"""
LLMResponse schema — mirrors the confirmed local API response contract.

API contract (POST /api/v1/chat):

    Request:
        {"model": "google/gemma-4-e2b", "input": "<text>"}

    Response:
        {
          "model_instance_id": "...",
          "response_id": "...",         # may be absent — UUID fallback applied
          "output": [
            {"type": "reasoning", "content": "..."},
            {"type": "message",   "content": "..."}
          ],
          "stats": {
            "input_tokens": 24,
            "total_output_tokens": 110,
            "reasoning_output_tokens": 99,
            "tokens_per_second": 52.83,
            "time_to_first_token_seconds": 0.11
          }
        }

Parsing rules applied here:
  - Unknown output block types are silently skipped (preserved in raw_response).
  - Multiple reasoning blocks → concatenated with newline.
  - Multiple message blocks  → concatenated with newline.
  - Missing response_id      → replaced with a fresh UUID.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------

class OutputBlockType(str, Enum):
    REASONING = "reasoning"
    MESSAGE = "message"
    # Additional block types introduced by future API versions are handled
    # gracefully in LLMResponse.from_api_response by skipping unknown types.


class OutputBlock(BaseModel):
    type: OutputBlockType
    content: str


class LLMStats(BaseModel):
    input_tokens: int = 0
    total_output_tokens: int = 0
    reasoning_output_tokens: int = 0
    tokens_per_second: float = 0.0
    time_to_first_token_seconds: float = 0.0

    @property
    def message_output_tokens(self) -> int:
        """Tokens attributed to the final answer (not reasoning)."""
        return max(0, self.total_output_tokens - self.reasoning_output_tokens)


# ---------------------------------------------------------------------------
# Internal parse helper (module-level for testability)
# ---------------------------------------------------------------------------

def _parse_output_blocks(output: list[OutputBlock]) -> tuple[Optional[str], str]:
    """
    Separate reasoning and message content from the output block list.

    Returns (reasoning, final_answer).
    reasoning is None when no reasoning block is present.
    """
    reasoning_parts = [b.content for b in output if b.type == OutputBlockType.REASONING]
    message_parts = [b.content for b in output if b.type == OutputBlockType.MESSAGE]

    reasoning: Optional[str] = "\n".join(reasoning_parts).strip() or None
    final_answer: str = "\n".join(message_parts).strip()

    return reasoning, final_answer


# ---------------------------------------------------------------------------
# Main response model
# ---------------------------------------------------------------------------

class LLMResponse(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    model_instance_id: str
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # ------------------------------------------------------------------
    # Raw API payload — preserved for debugging and reporting
    # ------------------------------------------------------------------
    raw_response: dict[str, Any]

    # ------------------------------------------------------------------
    # Parsed output structure
    # ------------------------------------------------------------------
    output: list[OutputBlock]
    stats: LLMStats

    # ------------------------------------------------------------------
    # Derived fields — populated by model_validator after construction
    # ------------------------------------------------------------------
    reasoning: Optional[str] = Field(
        default=None,
        description="Concatenated content of all 'reasoning' output blocks",
    )
    final_answer: str = Field(
        default="",
        description="Concatenated content of all 'message' output blocks",
    )

    # ------------------------------------------------------------------
    # Client-side timing (wall-clock, set by LocalLLMClient)
    # ------------------------------------------------------------------
    client_latency_ms: float = Field(
        default=0.0,
        description="Total elapsed milliseconds measured by the HTTP client",
    )

    # ------------------------------------------------------------------
    # Validator — auto-populate reasoning and final_answer from output
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _populate_parsed_fields(self) -> "LLMResponse":
        self.reasoning, self.final_answer = _parse_output_blocks(self.output)
        return self

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def has_reasoning(self) -> bool:
        return bool(self.reasoning and self.reasoning.strip())

    @property
    def is_empty(self) -> bool:
        return not self.final_answer.strip()

    @property
    def reasoning_ratio(self) -> float:
        """Fraction of total output tokens spent on reasoning (0.0 – 1.0)."""
        total = self.stats.total_output_tokens
        return 0.0 if total == 0 else self.stats.reasoning_output_tokens / total

    @property
    def ttft_seconds(self) -> float:
        return self.stats.time_to_first_token_seconds

    @property
    def tokens_per_second(self) -> float:
        return self.stats.tokens_per_second

    # ------------------------------------------------------------------
    # Factory — defensive parsing from raw API dict
    # ------------------------------------------------------------------

    @classmethod
    def from_api_response(
        cls,
        raw: dict[str, Any],
        client_latency_ms: float = 0.0,
    ) -> "LLMResponse":
        """
        Build an LLMResponse from the raw API response dict.

        Defensive rules applied:
          - Output blocks with unknown types are logged and skipped.
          - Missing response_id is replaced with a fresh UUID.
          - Missing or malformed stats fields default to zero.
          - The original dict is stored in raw_response unchanged.
        """
        from loguru import logger  # local import to avoid circular dep at module load

        output_blocks: list[OutputBlock] = []
        known_types = {t.value for t in OutputBlockType}

        for block in raw.get("output", []):
            block_type = block.get("type", "")
            if block_type not in known_types:
                logger.warning(
                    f"[LLMResponse] Unknown output block type '{block_type}' — skipped. "
                    f"Preserved in raw_response."
                )
                continue
            output_blocks.append(
                OutputBlock(
                    type=OutputBlockType(block_type),
                    content=block.get("content", ""),
                )
            )

        stats_raw: dict[str, Any] = raw.get("stats", {})
        stats = LLMStats(
            input_tokens=stats_raw.get("input_tokens", 0),
            total_output_tokens=stats_raw.get("total_output_tokens", 0),
            reasoning_output_tokens=stats_raw.get("reasoning_output_tokens", 0),
            tokens_per_second=float(stats_raw.get("tokens_per_second", 0.0)),
            time_to_first_token_seconds=float(
                stats_raw.get("time_to_first_token_seconds", 0.0)
            ),
        )

        return cls(
            model_instance_id=raw.get("model_instance_id", "unknown"),
            response_id=raw.get("response_id") or str(uuid.uuid4()),
            output=output_blocks,
            stats=stats,
            raw_response=raw,
            client_latency_ms=client_latency_ms,
        )
