"""
BaseEvaluator — abstract contract for all V1 (rule-based) evaluators.

Every evaluator subclass must:
  1. Declare TIER and EVALUATOR_NAME as ClassVar.
  2. Implement evaluate(test_case, response) → EvaluationResult.
  3. Use _make_result() to construct the result — never instantiate
     EvaluationResult directly inside an evaluator.

Extensibility hooks for V2:
  - can_upgrade_to() lists the V2 evaluator names that supersede this one.
  - upgrade_hint() surfaces a human-readable note in reports.

All V1 evaluators operate at EvalTier.RULE_BASED and must not make any
network calls, import embedding models, or call external APIs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Optional

from src.schemas.evaluation_result import (
    ConfidenceLevel,
    EvaluationResult,
    ProcessQualityStatus,
)
from src.schemas.llm_response import LLMResponse
from src.schemas.test_case import EvalTier, RiskLevel, TestCase
from src.utils.text_utils import contains_any


class BaseEvaluator(ABC):
    """
    Abstract base for all evaluators.

    Subclasses implement a single method — evaluate() — and declare two
    class-level constants so the framework can identify them in reports.
    """

    TIER: ClassVar[EvalTier] = EvalTier.RULE_BASED
    EVALUATOR_NAME: ClassVar[str] = "BaseEvaluator"

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    @abstractmethod
    def evaluate(self, test_case: TestCase, response: LLMResponse) -> EvaluationResult:
        """
        Score *response* against the expectations in *test_case*.

        Implementations MUST:
          - call self._make_result() to build the return value
          - not raise exceptions for model failures (score=0.0, passed=False)
          - not make network calls or load external models
          - be deterministic given the same inputs
        """

    # ------------------------------------------------------------------ #
    # V2 extensibility hooks (no-ops in V1, override when V2 lands)
    # ------------------------------------------------------------------ #

    def can_upgrade_to(self) -> list[str]:
        """
        Names of V2 evaluators that enhance or replace this one.
        Used by reporting to surface upgrade recommendations.
        """
        return []

    def upgrade_hint(self) -> Optional[str]:
        """Human-readable description of what a V2 upgrade would provide."""
        return None

    # ------------------------------------------------------------------ #
    # Protected helpers (for use by subclasses only)
    # ------------------------------------------------------------------ #

    def _make_result(
        self,
        test_case: TestCase,
        response: LLMResponse,
        score: float,
        passed: bool,
        confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
        explanation: str = "",
        failure_reason: Optional[str] = None,
        recommendation: str = "",
        sub_scores: Optional[dict[str, float]] = None,
        limitation_note: Optional[str] = None,
        reasoning_answer_consistent: Optional[bool] = None,
    ) -> EvaluationResult:
        """
        Construct a fully-populated EvaluationResult.

        Handles:
          - clamping score to [0.0, 1.0]
          - copying telemetry from LLMResponse
          - deriving process_quality from (passed, reasoning_answer_consistent)
          - propagating category, tags, risk_level from TestCase
        """
        score = max(0.0, min(1.0, score))

        result = EvaluationResult(
            test_id=test_case.test_id,
            evaluator_name=self.EVALUATOR_NAME,
            tier=self.TIER,
            prompt=test_case.prompt,
            response=response.final_answer,
            reasoning=response.reasoning,
            score=score,
            passed=passed,
            confidence=confidence,
            sub_scores=sub_scores or {},
            explanation=explanation,
            failure_reason=failure_reason,
            recommendation=recommendation,
            limitation_note=limitation_note,
            reasoning_answer_consistent=reasoning_answer_consistent,
            category=test_case.category,
            risk_level=test_case.risk_level,
            tags=list(test_case.tags),
        )

        result.populate_from_llm_response(response)
        result.derive_process_quality()
        return result

    def _forbidden_gate(
        self,
        text: str,
        forbidden: list[str],
    ) -> tuple[bool, list[str]]:
        """
        Check for forbidden content.

        Returns (blocked, matched_terms).
        blocked=True means the response must hard-fail regardless of other scores.
        """
        return contains_any(text, forbidden, case_sensitive=False)

    @staticmethod
    def _empty_response_result(
        test_case: TestCase,
        evaluator_name: str,
        tier: EvalTier,
    ) -> EvaluationResult:
        """
        Produce a standard FAIL result when the model returned an empty response.
        Used by EvaluationPipeline when LLMEmptyResponseError is caught.
        """
        return EvaluationResult(
            test_id=test_case.test_id,
            evaluator_name=evaluator_name,
            tier=tier,
            prompt=test_case.prompt,
            response="",
            score=0.0,
            passed=False,
            confidence=ConfidenceLevel.HIGH,
            explanation="Model returned an empty response.",
            failure_reason="empty_response",
            recommendation="Increase max_tokens or simplify the prompt.",
            category=test_case.category,
            risk_level=test_case.risk_level,
            tags=list(test_case.tags),
            process_quality=ProcessQualityStatus.FAIL,
        )
