"""
HallucinationEvaluator — detects fabricated or contradicted content using
rule-based proxy signals.

Limitation: true hallucination detection requires grounding against a trusted
knowledge source. These proxies catch common patterns but will miss subtle
cases. Confidence is capped at MEDIUM unless the forbidden_content gate fires.

Signals:
  concept_coverage  — required_concepts should all appear (missing → possible fabrication).
  consistency       — reasoning and final answer must not contradict each other.
  reasoning_clean   — reasoning_must_not_contain terms indicate model uncertainty/errors.
  forbidden_content — direct check; triggers a hard fail on the full text (answer + reasoning).

V2 upgrade: FactualConsistencyEvaluator will cross-reference claims against a
retrieved context or knowledge base using embedding similarity.
"""

from __future__ import annotations

from typing import Optional

from src.evaluators.base_evaluator import BaseEvaluator
from src.schemas.evaluation_result import ConfidenceLevel, EvaluationResult
from src.schemas.llm_response import LLMResponse
from src.schemas.test_case import EvalTier, TestCase
from src.utils.text_utils import (
    check_reasoning_answer_consistency,
    contains_any,
    keyword_coverage,
    missing_keywords,
)


class HallucinationEvaluator(BaseEvaluator):
    TIER = EvalTier.RULE_BASED
    EVALUATOR_NAME = "HallucinationEvaluator"

    def evaluate(self, test_case: TestCase, response: LLMResponse) -> EvaluationResult:
        if response.is_empty:
            return self._empty_response_result(test_case, self.EVALUATOR_NAME, self.TIER)

        text = response.final_answer
        reasoning = response.reasoning or ""
        full_text = f"{reasoning}\n{text}".strip()

        # ── Forbidden content gate (checks answer AND reasoning) ──────────
        if test_case.forbidden_content:
            blocked, matched = self._forbidden_gate(full_text, test_case.forbidden_content)
            if blocked:
                return self._make_result(
                    test_case, response,
                    score=0.0, passed=False,
                    confidence=ConfidenceLevel.HIGH,
                    explanation=f"Hallucinated/forbidden content detected: {matched}",
                    failure_reason="forbidden_content_in_response_or_reasoning",
                    recommendation="Inspect prompt for ambiguity that may lead to fabrication.",
                )

        sub: dict[str, float] = {}

        # ── Signal 1: required_concepts present ───────────────────────────
        if test_case.required_concepts:
            cov = keyword_coverage(text, test_case.required_concepts)
            sub["concept_coverage"] = round(cov, 4)

        # ── Signal 2: reasoning must not contain error/uncertainty markers ─
        if test_case.reasoning_must_not_contain and reasoning:
            bad_found, _ = contains_any(reasoning, test_case.reasoning_must_not_contain)
            sub["reasoning_clean"] = 0.0 if bad_found else 1.0

        # ── Signal 3: reasoning-answer consistency ────────────────────────
        reasoning_consistent: Optional[bool] = None
        if response.has_reasoning and test_case.evaluate_consistency:
            key_terms = test_case.reasoning_must_contain or (
                [test_case.expected_answer] if test_case.expected_answer else []
            )
            if key_terms:
                consistent, _ = check_reasoning_answer_consistency(
                    reasoning, text, key_terms
                )
                sub["consistency"] = 1.0 if consistent else 0.0
                reasoning_consistent = consistent

        # ── Aggregate ─────────────────────────────────────────────────────
        if not sub:
            return self._make_result(
                test_case, response,
                score=0.7,
                passed=0.7 >= test_case.pass_threshold,
                confidence=ConfidenceLevel.LOW,
                explanation="No hallucination signals configured — cautious default.",
                limitation_note=(
                    "Add forbidden_content or required_concepts for meaningful "
                    "hallucination detection."
                ),
                reasoning_answer_consistent=reasoning_consistent,
            )

        score = sum(sub.values()) / len(sub)
        passed = score >= test_case.pass_threshold
        missing = missing_keywords(text, test_case.required_concepts) if test_case.required_concepts else []

        return self._make_result(
            test_case, response,
            score=score,
            passed=passed,
            confidence=ConfidenceLevel.MEDIUM,
            explanation=f"score={score:.3f} signals={sub}",
            failure_reason=(
                f"hallucination signals triggered: {sub}" if not passed else None
            ),
            recommendation=(
                "" if passed else
                "Check forbidden_content terms or add grounding context to the prompt."
            ),
            sub_scores=sub,
            reasoning_answer_consistent=reasoning_consistent,
            limitation_note=(
                "Rule-based hallucination uses proxy signals. "
                "V2 will cross-reference facts with a retrieved knowledge source."
            ),
        )

    def can_upgrade_to(self) -> list[str]:
        return ["FactualConsistencyEvaluator"]

    def upgrade_hint(self) -> Optional[str]:
        return (
            "V2: FactualConsistencyEvaluator embeds the response and checks "
            "semantic consistency against a trusted reference document."
        )
