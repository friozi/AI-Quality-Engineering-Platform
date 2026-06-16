"""
RelevanceEvaluator — measures whether the response addresses the right topics
and maintains appropriate length proportional to the question.

Scoring signals (rule-based proxies):
  concept_coverage (0.60) — fraction of required_concepts present in the response.
  verbosity        (0.30) — penalises responses that are far too terse or verbose
                            relative to expected_answer length.
  non_empty        (0.10) — baseline: response exists.

V2 upgrade: semantic embedding similarity will replace keyword coverage for
true topical relevance regardless of synonym or phrasing variation.
"""

from __future__ import annotations

from typing import Optional

from src.evaluators.base_evaluator import BaseEvaluator
from src.schemas.evaluation_result import ConfidenceLevel, EvaluationResult
from src.schemas.llm_response import LLMResponse
from src.schemas.test_case import EvalTier, TestCase
from src.utils.text_utils import (
    check_reasoning_answer_consistency,
    keyword_coverage,
    matched_keywords,
    missing_keywords,
    verbosity_ratio,
)

_VERBOSITY_MIN = 0.3   # response must be at least 30% the length of expected_answer
_VERBOSITY_MAX = 8.0   # response must not exceed 8× the length of expected_answer


class RelevanceEvaluator(BaseEvaluator):
    TIER = EvalTier.RULE_BASED
    EVALUATOR_NAME = "RelevanceEvaluator"

    def evaluate(self, test_case: TestCase, response: LLMResponse) -> EvaluationResult:
        if response.is_empty:
            return self._empty_response_result(test_case, self.EVALUATOR_NAME, self.TIER)

        text = response.final_answer

        # ── Forbidden content gate ────────────────────────────────────────
        if test_case.forbidden_content:
            blocked, matched = self._forbidden_gate(text, test_case.forbidden_content)
            if blocked:
                return self._make_result(
                    test_case, response,
                    score=0.0, passed=False,
                    confidence=ConfidenceLevel.HIGH,
                    explanation=f"Forbidden content: {matched}",
                    failure_reason="forbidden_content",
                )

        sub: dict[str, float] = {}

        # ── Signal 1: required_concepts coverage ──────────────────────────
        if test_case.required_concepts:
            cov = keyword_coverage(text, test_case.required_concepts)
            sub["concept_coverage"] = round(cov, 4)

        # ── Signal 2: verbosity vs expected_answer ─────────────────────────
        if test_case.expected_answer:
            vr = verbosity_ratio(text, test_case.expected_answer)
            sub["verbosity_raw"] = round(vr, 4)   # diagnostic only, not weighted
            if vr < _VERBOSITY_MIN:
                sub["verbosity"] = vr / _VERBOSITY_MIN
            elif vr > _VERBOSITY_MAX:
                sub["verbosity"] = max(0.0, 1.0 - (vr - _VERBOSITY_MAX) / 20.0)
            else:
                sub["verbosity"] = 1.0

        # ── Signal 3: non-empty baseline ──────────────────────────────────
        sub["non_empty"] = 1.0 if text.strip() else 0.0

        # ── Score ─────────────────────────────────────────────────────────
        scoring_keys = {"concept_coverage", "verbosity", "non_empty"}
        scoring = {k: v for k, v in sub.items() if k in scoring_keys}
        weights: dict[str, float] = {"concept_coverage": 0.60, "verbosity": 0.30, "non_empty": 0.10}
        total_w = sum(weights.get(k, 0.1) for k in scoring)
        score = sum(scoring[k] * weights.get(k, 0.1) for k in scoring) / total_w if scoring else 0.5

        confidence = ConfidenceLevel.MEDIUM if "concept_coverage" in sub else ConfidenceLevel.LOW

        # ── Reasoning consistency ─────────────────────────────────────────
        reasoning_consistent: Optional[bool] = None
        if test_case.evaluate_consistency and response.has_reasoning and test_case.required_concepts:
            consistent, _ = check_reasoning_answer_consistency(
                response.reasoning or "", text, test_case.required_concepts
            )
            reasoning_consistent = consistent

        passed = score >= test_case.pass_threshold
        missing = missing_keywords(text, test_case.required_concepts) if test_case.required_concepts else []

        return self._make_result(
            test_case, response,
            score=score,
            passed=passed,
            confidence=confidence,
            explanation=f"score={score:.3f} signals={scoring}",
            failure_reason=(f"missing concepts: {missing}" if missing and not passed else None),
            sub_scores=sub,
            reasoning_answer_consistent=reasoning_consistent,
            limitation_note=(
                "Relevance uses keyword coverage as a proxy. "
                "V2 will use embedding similarity for true semantic relevance."
            ),
        )

    def can_upgrade_to(self) -> list[str]:
        return ["EmbeddingRelevanceEvaluator"]

    def upgrade_hint(self) -> Optional[str]:
        return (
            "V2: semantic embedding similarity scores relevance even when the "
            "response uses synonyms or different phrasing."
        )
