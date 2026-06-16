from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.schemas.test_case import EvalTier, RiskLevel


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ConfidenceLevel(str, Enum):
    """
    How reliable is the score produced by this evaluator?

    HIGH   : Deterministic rule-based checks (exact match, regex, structural)
    MEDIUM : Heuristic with known limitations (fuzzy match, proxy hallucination)
    LOW    : Proxy signal only — result requires human verification
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class StabilityStatus(str, Enum):
    """Populated when a test case is run more than once (N_RUNS > 1)."""
    STABLE = "stable"       # Same pass/fail verdict on every run
    FLAKY = "flaky"         # Verdict flipped across runs
    UNTESTED = "untested"   # Only one run performed (default)


class ProcessQualityStatus(str, Enum):
    """
    Two-axis quality assessment enabled by the separated reasoning block.

    IDEAL : Correct final answer AND sound reasoning chain.
    LUCKY : Correct final answer BUT weak or absent reasoning.
    ERROR : Sound reasoning chain BUT wrong final answer (e.g. arithmetic slip).
    FAIL  : Wrong answer AND poor/absent reasoning.
    N_A   : Reasoning quality was not evaluated for this test case.
    """
    IDEAL = "ideal"
    LUCKY = "lucky"
    ERROR = "error"
    FAIL = "fail"
    NOT_EVALUATED = "n_a"


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------

class EvaluationResult(BaseModel):
    """
    The complete output of a single evaluator run against one test case.

    Evaluators construct this object and return it.  The EvaluationPipeline
    and MetricsCollector consume it.  Reporters serialise it.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    test_id: str
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    evaluator_name: str
    tier: EvalTier

    # ------------------------------------------------------------------
    # Timing (merged from API stats and client measurement)
    # ------------------------------------------------------------------
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    client_latency_ms: float = 0.0
    ttft_seconds: Optional[float] = None
    tokens_per_second: Optional[float] = None

    # ------------------------------------------------------------------
    # Model context
    # ------------------------------------------------------------------
    model_name: str = ""
    model_key: str = ""
    model_instance_id: str = ""
    response_id: str = ""

    # ------------------------------------------------------------------
    # Token accounting
    # ------------------------------------------------------------------
    input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    message_tokens: Optional[int] = None

    # ------------------------------------------------------------------
    # Inputs / Outputs
    # ------------------------------------------------------------------
    prompt: str
    response: str = Field(description="final_answer from LLMResponse")
    reasoning: Optional[str] = Field(
        default=None,
        description="Reasoning block content (None if model did not reason)",
    )

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH

    # Breakdown by evaluation method (populated by composite evaluators)
    sub_scores: dict[str, float] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Process quality (reasoning-aware)
    # ------------------------------------------------------------------
    has_reasoning: bool = False
    reasoning_ratio: float = 0.0
    reasoning_answer_consistent: Optional[bool] = None
    process_quality: ProcessQualityStatus = ProcessQualityStatus.NOT_EVALUATED

    # ------------------------------------------------------------------
    # Risk context
    # ------------------------------------------------------------------
    risk_level: RiskLevel = RiskLevel.MEDIUM

    # ------------------------------------------------------------------
    # Human-readable detail
    # ------------------------------------------------------------------
    explanation: str = ""
    failure_reason: Optional[str] = None
    recommendation: str = ""
    limitation_note: Optional[str] = Field(
        default=None,
        description=(
            "Populated when the evaluator uses proxy signals only. "
            "Surfaces in reports so consumers understand score reliability."
        ),
    )

    # ------------------------------------------------------------------
    # Stability (populated when N_RUNS > 1)
    # ------------------------------------------------------------------
    stability_status: StabilityStatus = StabilityStatus.UNTESTED

    # ------------------------------------------------------------------
    # Pass-through metadata (for filtering and reporting)
    # ------------------------------------------------------------------
    category: str = ""
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def populate_from_llm_response(self, response: Any) -> None:
        """
        Copy token stats and reasoning metadata from an LLMResponse.

        Called by the EvaluationPipeline after construction to keep all
        telemetry in one place without tight coupling between schemas.
        """
        self.client_latency_ms = response.client_latency_ms
        self.ttft_seconds = response.ttft_seconds
        self.tokens_per_second = response.tokens_per_second
        self.input_tokens = response.stats.input_tokens
        self.total_output_tokens = response.stats.total_output_tokens
        self.reasoning_tokens = response.stats.reasoning_output_tokens
        self.message_tokens = response.stats.message_output_tokens
        self.has_reasoning = response.has_reasoning
        self.reasoning_ratio = response.reasoning_ratio
        self.model_instance_id = response.model_instance_id
        self.response_id = response.response_id

    def derive_process_quality(self) -> None:
        """
        Compute process_quality from score and reasoning_answer_consistent.
        Call after all score fields are finalised.
        """
        if not self.has_reasoning:
            self.process_quality = (
                ProcessQualityStatus.IDEAL if self.passed else ProcessQualityStatus.FAIL
            )
            return

        answer_ok = self.passed
        # reasoning_answer_consistent=None means consistency was not evaluated.
        chain_ok = self.reasoning_answer_consistent is not False

        if answer_ok and chain_ok:
            self.process_quality = ProcessQualityStatus.IDEAL
        elif answer_ok:          # chain_ok is implicitly False
            self.process_quality = ProcessQualityStatus.LUCKY
        elif chain_ok:           # answer_ok is implicitly False
            self.process_quality = ProcessQualityStatus.ERROR
        else:
            self.process_quality = ProcessQualityStatus.FAIL

    def to_flat_dict(self) -> dict[str, Any]:
        """
        Return a flat dict suitable for CSV export.

        Nested / collection fields are stringified; metadata is omitted
        because it may contain complex structures not expressible in CSV.
        """
        d = self.model_dump(mode="json")
        d["sub_scores"] = "; ".join(f"{k}={v:.3f}" for k, v in self.sub_scores.items())
        d["tags"] = ", ".join(self.tags)
        d["timestamp"] = self.timestamp.isoformat()
        d["tier"] = self.tier.name
        d["confidence"] = self.confidence.value
        d["stability_status"] = self.stability_status.value
        d["process_quality"] = self.process_quality.value
        d["risk_level"] = self.risk_level.value
        d.pop("metadata", None)
        d.pop("raw_response", None)
        return d
