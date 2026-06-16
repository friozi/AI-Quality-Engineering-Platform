"""
LatencyEvaluator — evaluates model response timing against SLA thresholds.

Signals:
  total_latency  (0.60) — client-measured round-trip time vs test_case.timeout_seconds.
  ttft           (0.25) — time-to-first-token vs a 5-second default.
  throughput     (0.15) — tokens/second vs a 5 tok/s minimum.

Signals with missing data are skipped; weights are renormalised.
Score degrades gracefully (not binary) so reports show how far over the SLA
the model was rather than just pass/fail.
"""

from __future__ import annotations

from typing import Optional

from src.evaluators.base_evaluator import BaseEvaluator
from src.schemas.evaluation_result import ConfidenceLevel, EvaluationResult
from src.schemas.llm_response import LLMResponse
from src.schemas.test_case import EvalTier, TestCase

_TTFT_THRESHOLD_S = 5.0    # seconds — time to first token SLA
_MIN_TPS = 5.0             # tokens per second minimum


class LatencyEvaluator(BaseEvaluator):
    TIER = EvalTier.RULE_BASED
    EVALUATOR_NAME = "LatencyEvaluator"

    def evaluate(self, test_case: TestCase, response: LLMResponse) -> EvaluationResult:
        # Latency evaluation does not require a non-empty response —
        # a slow empty response is still a latency failure.

        sub: dict[str, float] = {}
        threshold_ms = test_case.timeout_seconds * 1000.0
        latency_ms = response.client_latency_ms

        # ── Signal 1: Total round-trip latency ────────────────────────────
        if latency_ms > 0:
            if latency_ms <= threshold_ms:
                sub["total_latency"] = 1.0
            else:
                sub["total_latency"] = round(threshold_ms / latency_ms, 4)

        # ── Signal 2: Time to first token ─────────────────────────────────
        ttft = response.ttft_seconds
        if ttft is not None and ttft > 0:
            if ttft <= _TTFT_THRESHOLD_S:
                sub["ttft"] = 1.0
            else:
                sub["ttft"] = round(_TTFT_THRESHOLD_S / ttft, 4)

        # ── Signal 3: Tokens per second ───────────────────────────────────
        tps = response.tokens_per_second
        if tps is not None and tps > 0:
            if tps >= _MIN_TPS:
                sub["throughput"] = 1.0
            else:
                sub["throughput"] = round(tps / _MIN_TPS, 4)

        if not sub:
            return self._make_result(
                test_case, response,
                score=0.5, passed=False,
                confidence=ConfidenceLevel.LOW,
                explanation="No timing data available from LLMResponse.",
                failure_reason="no_timing_data",
                limitation_note="Timing signals require a real LLM response (not available in this run).",
            )

        # ── Weighted score ─────────────────────────────────────────────────
        weights: dict[str, float] = {"total_latency": 0.60, "ttft": 0.25, "throughput": 0.15}
        total_w = sum(weights.get(k, 0.1) for k in sub)
        score = sum(sub[k] * weights.get(k, 0.1) for k in sub) / total_w

        passed = score >= test_case.pass_threshold
        over_ms = max(0.0, latency_ms - threshold_ms)

        return self._make_result(
            test_case, response,
            score=round(score, 4),
            passed=passed,
            confidence=ConfidenceLevel.HIGH,
            explanation=(
                f"latency={latency_ms:.0f}ms threshold={threshold_ms:.0f}ms "
                f"score={score:.3f} signals={sub}"
            ),
            failure_reason=(
                None if passed else
                f"latency {latency_ms:.0f}ms exceeded threshold {threshold_ms:.0f}ms by {over_ms:.0f}ms"
            ),
            recommendation=(
                "" if passed else
                "Consider increasing test_case.timeout_seconds or optimising the prompt length."
            ),
            sub_scores=sub,
        )

    def can_upgrade_to(self) -> list[str]:
        return []

    def upgrade_hint(self) -> Optional[str]:
        return None
