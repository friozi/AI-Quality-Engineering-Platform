"""
AccuracyEvaluator — rule-based factual accuracy scoring.

Routes evaluation to one of seven strategies based on test_case.evaluation_strategy:

  EXACT      — expected_answer must appear as a substring of the response.
  FUZZY      — rapidfuzz token_set_ratio against expected_answer + valid_alternatives.
  MATH       — extract numbers from both sides and compare with tolerance.
  COVERAGE   — keyword_coverage of required_concepts in the response.
  REGEX      — response must match at least one pattern from metadata["regex_patterns"].
  STRUCTURAL — response must be valid JSON containing metadata["required_fields"].
  COMPOSITE  — weighted combination of exact, fuzzy, and coverage signals (default).

All strategies:
  1. Apply the forbidden_content gate first (hard fail if triggered).
  2. Optionally check reasoning-answer consistency when evaluate_consistency=True.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from src.evaluators.base_evaluator import BaseEvaluator
from src.schemas.evaluation_result import ConfidenceLevel, EvaluationResult
from src.schemas.llm_response import LLMResponse
from src.schemas.test_case import EvalStrategy, EvalTier, TestCase
from src.utils.text_utils import (
    best_fuzzy_score,
    check_reasoning_answer_consistency,
    extract_numbers,
    keyword_coverage,
    matched_keywords,
    missing_keywords,
    numbers_match,
    matches_any_pattern,
)


class AccuracyEvaluator(BaseEvaluator):
    TIER = EvalTier.RULE_BASED
    EVALUATOR_NAME = "AccuracyEvaluator"

    def evaluate(self, test_case: TestCase, response: LLMResponse) -> EvaluationResult:
        if response.is_empty:
            return self._empty_response_result(test_case, self.EVALUATOR_NAME, self.TIER)

        text = response.final_answer

        # ── 1. Forbidden content gate ─────────────────────────────────────
        if test_case.forbidden_content:
            blocked, matched = self._forbidden_gate(text, test_case.forbidden_content)
            if blocked:
                return self._make_result(
                    test_case, response,
                    score=0.0, passed=False,
                    confidence=ConfidenceLevel.HIGH,
                    explanation=f"Forbidden content detected: {matched}",
                    failure_reason="forbidden_content",
                    recommendation="Review response for policy violations.",
                )

        # ── 2. Strategy dispatch ──────────────────────────────────────────
        strategy = test_case.evaluation_strategy

        dispatch = {
            EvalStrategy.EXACT:      self._score_exact,
            EvalStrategy.FUZZY:      self._score_fuzzy,
            EvalStrategy.MATH:       self._score_math,
            EvalStrategy.COVERAGE:   self._score_coverage,
            EvalStrategy.REGEX:      self._score_regex,
            EvalStrategy.STRUCTURAL: self._score_structural,
        }
        scorer = dispatch.get(strategy, self._score_composite)
        score, sub_scores, confidence = scorer(test_case, text)

        # ── 3. Optional reasoning consistency ────────────────────────────
        reasoning_consistent: Optional[bool] = None
        if test_case.evaluate_consistency and response.has_reasoning:
            key_terms = test_case.reasoning_must_contain or (
                [test_case.expected_answer] if test_case.expected_answer else []
            )
            if key_terms:
                consistent, _ = check_reasoning_answer_consistency(
                    response.reasoning or "", text, key_terms
                )
                reasoning_consistent = consistent

        passed = score >= test_case.pass_threshold
        return self._make_result(
            test_case, response,
            score=score,
            passed=passed,
            confidence=confidence,
            explanation=self._explanation(strategy, score, sub_scores),
            failure_reason=None if passed else self._failure_reason(strategy, score, test_case, text),
            recommendation="" if passed else "Check expected_answer or widen pass_threshold.",
            sub_scores=sub_scores,
            reasoning_answer_consistent=reasoning_consistent,
        )

    # ------------------------------------------------------------------ #
    # Strategy scorers — each returns (score, sub_scores, confidence)
    # ------------------------------------------------------------------ #

    def _score_exact(
        self, test_case: TestCase, text: str
    ) -> tuple[float, dict[str, float], ConfidenceLevel]:
        refs = self._refs(test_case)
        found = any(r.lower() in text.lower() for r in refs if r)
        score = 1.0 if found else 0.0
        return score, {"exact_match": score}, ConfidenceLevel.HIGH

    def _score_fuzzy(
        self, test_case: TestCase, text: str
    ) -> tuple[float, dict[str, float], ConfidenceLevel]:
        refs = self._refs(test_case)
        if not refs:
            return 0.5, {}, ConfidenceLevel.LOW
        score = best_fuzzy_score(text, refs)
        return score, {"fuzzy_score": round(score, 4)}, ConfidenceLevel.MEDIUM

    def _score_math(
        self, test_case: TestCase, text: str
    ) -> tuple[float, dict[str, float], ConfidenceLevel]:
        if not test_case.expected_answer:
            return 0.5, {}, ConfidenceLevel.LOW

        expected_nums = extract_numbers(test_case.expected_answer)
        if not expected_nums:
            # No parsable numbers — fall back to substring check.
            found = test_case.expected_answer.lower() in text.lower()
            return (1.0 if found else 0.0), {"substring": 1.0 if found else 0.0}, ConfidenceLevel.MEDIUM

        response_nums = extract_numbers(text)
        hits = sum(
            1 for en in expected_nums
            if any(numbers_match(en, rn) for rn in response_nums)
        )
        score = hits / len(expected_nums)
        return score, {"number_match": round(score, 4), "expected": expected_nums}, ConfidenceLevel.HIGH

    def _score_coverage(
        self, test_case: TestCase, text: str
    ) -> tuple[float, dict[str, float], ConfidenceLevel]:
        if not test_case.required_concepts:
            return 0.5, {}, ConfidenceLevel.LOW
        score = keyword_coverage(text, test_case.required_concepts)
        return score, {
            "concept_coverage": round(score, 4),
            "found": len(matched_keywords(text, test_case.required_concepts)),
            "missing": len(missing_keywords(text, test_case.required_concepts)),
        }, ConfidenceLevel.MEDIUM

    def _score_regex(
        self, test_case: TestCase, text: str
    ) -> tuple[float, dict[str, float], ConfidenceLevel]:
        patterns: list[str] = test_case.metadata.get("regex_patterns", [])
        if not patterns:
            return self._score_composite(test_case, text)
        matched, pat = matches_any_pattern(text, patterns, re.IGNORECASE)
        return (1.0 if matched else 0.0), {"regex_match": 1.0 if matched else 0.0}, ConfidenceLevel.HIGH

    def _score_structural(
        self, test_case: TestCase, text: str
    ) -> tuple[float, dict[str, float], ConfidenceLevel]:
        required_fields: list[str] = test_case.metadata.get("required_fields", [])
        if not required_fields:
            return self._score_composite(test_case, text)

        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return 0.0, {"json_valid": 0.0}, ConfidenceLevel.HIGH

        if not isinstance(parsed, dict):
            return 0.0, {"json_valid": 1.0, "is_object": 0.0}, ConfidenceLevel.HIGH

        hits = sum(1 for f in required_fields if f in parsed)
        score = hits / len(required_fields)
        return score, {"field_coverage": round(score, 4)}, ConfidenceLevel.HIGH

    def _score_composite(
        self, test_case: TestCase, text: str
    ) -> tuple[float, dict[str, float], ConfidenceLevel]:
        sub: dict[str, float] = {}

        refs = self._refs(test_case)
        if refs:
            sub["exact_match"] = 1.0 if any(r.lower() in text.lower() for r in refs if r) else 0.0
            sub["fuzzy_score"] = round(best_fuzzy_score(text, refs), 4)

        if test_case.required_concepts:
            sub["concept_coverage"] = round(keyword_coverage(text, test_case.required_concepts), 4)

        if not sub:
            return 0.5, {}, ConfidenceLevel.LOW

        weights: dict[str, float] = {"exact_match": 0.40, "fuzzy_score": 0.35, "concept_coverage": 0.25}
        total_w = sum(weights.get(k, 0.2) for k in sub)
        score = sum(sub[k] * weights.get(k, 0.2) for k in sub) / total_w
        return score, sub, ConfidenceLevel.MEDIUM

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _refs(test_case: TestCase) -> list[str]:
        refs = []
        if test_case.expected_answer:
            refs.append(test_case.expected_answer)
        refs.extend(test_case.valid_alternatives)
        return refs

    @staticmethod
    def _explanation(
        strategy: EvalStrategy, score: float, sub_scores: dict[str, float]
    ) -> str:
        return f"strategy={strategy.value} score={score:.3f} signals={sub_scores}"

    @staticmethod
    def _failure_reason(
        strategy: EvalStrategy, score: float, test_case: TestCase, text: str
    ) -> str:
        if strategy == EvalStrategy.EXACT:
            return f"expected '{test_case.expected_answer}' not found in response"
        if strategy == EvalStrategy.FUZZY:
            return f"fuzzy score {score:.3f} below threshold {test_case.pass_threshold}"
        if strategy == EvalStrategy.MATH:
            return f"expected number(s) from '{test_case.expected_answer}' not found in response"
        if strategy == EvalStrategy.COVERAGE:
            missing = missing_keywords(text, test_case.required_concepts)
            return f"missing required concepts: {missing}"
        return f"score {score:.3f} below threshold {test_case.pass_threshold}"

    # ------------------------------------------------------------------ #
    # V2 extensibility
    # ------------------------------------------------------------------ #

    def can_upgrade_to(self) -> list[str]:
        return ["EmbeddingAccuracyEvaluator"]

    def upgrade_hint(self) -> Optional[str]:
        return (
            "V2: EmbeddingAccuracyEvaluator uses semantic similarity so paraphrased "
            "correct answers score highly even when they don't share exact tokens."
        )
