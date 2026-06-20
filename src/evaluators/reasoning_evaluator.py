"""
ReasoningEvaluator — assesses the quality of the model's chain-of-thought.

Scoring signals:
  has_reasoning      (0.25) — model produced a separate reasoning block.
  step_coverage      (0.30) — required_reasoning_steps found in reasoning text.
  required_terms     (0.20) — reasoning_must_contain terms present.
  reasoning_clean    (0.10) — reasoning_must_not_contain terms absent.
  consistency        (0.15) — reasoning and final answer agree on key facts.

Signals not configured by the test case are skipped; weights renormalised.
If evaluate_reasoning_quality=False the evaluator passes through with score=1.0.
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
    count_reasoning_steps,
    keyword_coverage,
    missing_keywords,
)


class ReasoningEvaluator(BaseEvaluator):
    TIER = EvalTier.RULE_BASED
    EVALUATOR_NAME = "ReasoningEvaluator"

    def evaluate(self, test_case: TestCase, response: LLMResponse) -> EvaluationResult:
        if response.is_empty:
            return self._empty_response_result(test_case, self.EVALUATOR_NAME, self.TIER)

        # If the test case does not request reasoning quality evaluation,
        # pass through — this evaluator is a no-op for such cases.
        if not test_case.evaluate_reasoning_quality:
            return self._make_result(
                test_case, response,
                score=1.0, passed=True,
                confidence=ConfidenceLevel.HIGH,
                explanation="evaluate_reasoning_quality=False — skipped.",
            )

        reasoning = response.reasoning or ""
        text = response.final_answer
        sub: dict[str, float] = {}

        # ── Signal 1: Presence of reasoning block ─────────────────────────
        if response.has_reasoning:
            sub["has_reasoning"] = 1.0
        else:
            # No API reasoning block. Check if the model put step-by-step
            # reasoning directly in the message (some models do this).
            in_message_steps = count_reasoning_steps(response.final_answer)
            if in_message_steps >= 2:
                # Treat the message as a reasoning proxy with partial credit.
                reasoning = response.final_answer
                sub["has_reasoning"] = 0.5
            else:
                return self._make_result(
                    test_case, response,
                    score=0.2, passed=False,
                    confidence=ConfidenceLevel.HIGH,
                    explanation="Model did not produce a reasoning block.",
                    failure_reason="no_reasoning_block",
                    recommendation="Use a prompt that explicitly requests step-by-step reasoning.",
                    sub_scores=sub,
                )

        # ── Signal 2: Required reasoning steps coverage ────────────────────
        if test_case.required_reasoning_steps:
            cov = keyword_coverage(reasoning, test_case.required_reasoning_steps)
            sub["step_coverage"] = round(cov, 4)

        # ── Signal 3: reasoning_must_contain terms ─────────────────────────
        if test_case.reasoning_must_contain:
            must_cov = keyword_coverage(reasoning, test_case.reasoning_must_contain)
            sub["required_terms"] = round(must_cov, 4)

        # ── Signal 4: reasoning_must_not_contain terms ─────────────────────
        if test_case.reasoning_must_not_contain:
            bad_found, _ = contains_any(reasoning, test_case.reasoning_must_not_contain)
            sub["reasoning_clean"] = 0.0 if bad_found else 1.0

        # ── Signal 5: Reasoning-answer consistency ─────────────────────────
        reasoning_consistent: Optional[bool] = None
        if test_case.evaluate_consistency:
            key_terms = test_case.reasoning_must_contain or (
                [test_case.expected_answer] if test_case.expected_answer else []
            )
            if key_terms:
                consistent, _ = check_reasoning_answer_consistency(
                    reasoning, text, key_terms
                )
                sub["consistency"] = 1.0 if consistent else 0.0
                reasoning_consistent = consistent

        # ── Step count diagnostic (not weighted into score) ────────────────
        if test_case.required_reasoning_steps:
            n_actual = count_reasoning_steps(reasoning)
            n_expected = len(test_case.required_reasoning_steps)
            sub["step_count_ratio"] = round(min(1.0, n_actual / max(1, n_expected)), 4)

        # ── Weighted score ─────────────────────────────────────────────────
        score_keys = {"has_reasoning", "step_coverage", "required_terms", "reasoning_clean", "consistency"}
        scoring = {k: v for k, v in sub.items() if k in score_keys}

        if not scoring:
            score = 1.0 if response.has_reasoning else 0.5
        else:
            weights: dict[str, float] = {
                "has_reasoning": 0.25,
                "step_coverage": 0.30,
                "required_terms": 0.20,
                "reasoning_clean": 0.10,
                "consistency": 0.15,
            }
            total_w = sum(weights.get(k, 0.1) for k in scoring)
            score = sum(scoring[k] * weights.get(k, 0.1) for k in scoring) / total_w

        passed = score >= test_case.pass_threshold

        missing_steps = (
            missing_keywords(reasoning, test_case.required_reasoning_steps)
            if test_case.required_reasoning_steps else []
        )

        return self._make_result(
            test_case, response,
            score=round(score, 4),
            passed=passed,
            confidence=ConfidenceLevel.MEDIUM,
            explanation=f"score={score:.3f} signals={scoring}",
            failure_reason=(
                None if passed else
                (f"missing steps: {missing_steps}" if missing_steps else f"score {score:.3f} below threshold")
            ),
            recommendation=(
                "" if passed else
                "Prompt model to reason step-by-step and cover the required concepts."
            ),
            sub_scores=sub,
            reasoning_answer_consistent=reasoning_consistent,
            limitation_note=(
                "Reasoning quality uses keyword coverage as a proxy for step completeness. "
                "V2 will use embedding similarity to detect semantically equivalent steps."
            ),
        )

    def can_upgrade_to(self) -> list[str]:
        return ["EmbeddingReasoningEvaluator", "LLMJudgeReasoningEvaluator"]

    def upgrade_hint(self) -> Optional[str]:
        return (
            "V2: EmbeddingReasoningEvaluator uses semantic similarity to detect "
            "reasoning steps even when phrased differently from the expected keywords."
        )
