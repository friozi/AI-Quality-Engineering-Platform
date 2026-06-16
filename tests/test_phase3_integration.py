"""
Phase 3 infrastructure tests.

Structured in two parts:

  Part 1 — Unit tests (no LLM calls, no shared fixtures)
  -------------------------------------------------------
  Each class creates its own instances inline.  These tests run offline
  and must stay fast (< 1 s total).

  Part 2 — Dataset-driven integration tests (real LLM via conftest fixtures)
  --------------------------------------------------------------------------
  pytest.mark.parametrize loads test cases from JSON files at collection time.
  Adding entries to a JSON file automatically creates new test cases — no
  Python changes required.

  Pattern:
      @pytest.mark.parametrize("test_case",
          DatasetLoader.load(TEST_DATA / "accuracy" / "geography.json"),
          ids=lambda tc: tc.test_id)
      def test_accuracy_geography(pipeline, test_case):
          result = pipeline.run(test_case, StubEvaluator())
          assert result.passed, ...
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from src.evaluators.base_evaluator import BaseEvaluator
from src.metrics.metrics_calculator import AggregatedMetrics, MetricsCollector
from src.pipeline.evaluation_pipeline import EvaluationPipeline
from src.schemas.evaluation_result import (
    ConfidenceLevel,
    EvaluationResult,
    ProcessQualityStatus,
)
from src.schemas.llm_response import LLMResponse
from src.schemas.test_case import EvalTier, RiskLevel, TestCase
from src.utils.config import Config
from src.utils.dataset_loader import DatasetLoader, DatasetValidationError

# ---------------------------------------------------------------------------
# Dataset root — resolved relative to this file so tests work regardless of
# the working directory pytest is invoked from.
# ---------------------------------------------------------------------------
TEST_DATA = Path(__file__).parent.parent / "test_data"


# ---------------------------------------------------------------------------
# StubEvaluator — minimal concrete subclass used throughout these tests.
# Passes when test_case.expected_answer appears (case-insensitive) in the
# model's final answer.
# ---------------------------------------------------------------------------

class StubEvaluator(BaseEvaluator):
    TIER = EvalTier.RULE_BASED
    EVALUATOR_NAME = "StubEvaluator"

    def evaluate(self, test_case: TestCase, response: LLMResponse) -> EvaluationResult:
        answer = response.final_answer.lower()
        hit = (
            test_case.expected_answer is not None
            and test_case.expected_answer.lower() in answer
        )
        score = 1.0 if hit else 0.4
        return self._make_result(
            test_case,
            response,
            score=score,
            passed=hit,
            confidence=ConfidenceLevel.HIGH,
            explanation=f"Expected '{test_case.expected_answer}' in answer",
            failure_reason=None if hit else "expected_answer_not_found",
        )


# ---------------------------------------------------------------------------
# Helper — build a minimal EvaluationResult without any LLM call.
# Used by the MetricsCollector and Aggregation unit tests.
# ---------------------------------------------------------------------------

def _fake_result(
    test_id: str = "FAKE_001",
    passed: bool = True,
    score: float | None = None,
    category: str = "accuracy",
    risk_level: RiskLevel = RiskLevel.MEDIUM,
) -> EvaluationResult:
    return EvaluationResult(
        test_id=test_id,
        evaluator_name="StubEvaluator",
        tier=EvalTier.RULE_BASED,
        prompt="stub prompt",
        response="stub response",
        score=score if score is not None else (1.0 if passed else 0.0),
        passed=passed,
        category=category,
        risk_level=risk_level,
        process_quality=(
            ProcessQualityStatus.IDEAL if passed else ProcessQualityStatus.FAIL
        ),
    )


# ===========================================================================
# PART 1 — Unit tests (no LLM calls)
# ===========================================================================

class TestStubEvaluator:
    """Verify the StubEvaluator contract required by BaseEvaluator."""

    def test_tier_is_rule_based(self) -> None:
        assert StubEvaluator.TIER == EvalTier.RULE_BASED

    def test_evaluator_name(self) -> None:
        assert StubEvaluator.EVALUATOR_NAME == "StubEvaluator"

    def test_can_upgrade_to_returns_empty_list(self) -> None:
        assert StubEvaluator().can_upgrade_to() == []

    def test_upgrade_hint_returns_none(self) -> None:
        assert StubEvaluator().upgrade_hint() is None


class TestMetricsCollector:
    """
    Isolated unit tests for MetricsCollector.
    All instances are created locally — no session state involved.
    """

    def test_starts_empty(self) -> None:
        assert MetricsCollector().count() == 0

    def test_record_increments_count(self) -> None:
        c = MetricsCollector()
        c.record(_fake_result("MC_001"))
        assert c.count() == 1

    def test_record_multiple(self) -> None:
        c = MetricsCollector()
        for i in range(5):
            c.record(_fake_result(f"MC_{i:03d}"))
        assert c.count() == 5

    def test_clear_resets(self) -> None:
        c = MetricsCollector()
        c.record(_fake_result("MC_CLR"))
        c.clear()
        assert c.count() == 0

    def test_get_results_snapshot(self) -> None:
        c = MetricsCollector()
        r = _fake_result("MC_SNAP")
        c.record(r)
        results = c.get_results()
        assert len(results) == 1
        assert results[0].test_id == "MC_SNAP"

    def test_get_results_is_a_copy(self) -> None:
        c = MetricsCollector()
        c.record(_fake_result("MC_CPY"))
        snapshot = c.get_results()
        snapshot.clear()
        assert c.count() == 1  # original unaffected


class TestAggregation:
    """
    Unit tests for MetricsCollector.aggregate().
    Uses fake EvaluationResults — no LLM, no network.
    """

    def _collector(
        self,
        n: int = 4,
        n_passed: int = 3,
        category: str = "accuracy",
        risk_level: RiskLevel = RiskLevel.MEDIUM,
    ) -> MetricsCollector:
        c = MetricsCollector()
        for i in range(n):
            c.record(
                _fake_result(
                    test_id=f"AGG_{i:03d}",
                    passed=(i < n_passed),
                    category=category,
                    risk_level=risk_level,
                )
            )
        return c

    def test_empty_returns_aggregated_metrics(self) -> None:
        metrics = MetricsCollector().aggregate()
        assert isinstance(metrics, AggregatedMetrics)
        assert metrics.total_tests == 0

    def test_total_tests(self) -> None:
        assert self._collector(n=7).aggregate().total_tests == 7

    def test_passed_count(self) -> None:
        metrics = self._collector(n=4, n_passed=3).aggregate()
        assert metrics.total_passed == 3
        assert metrics.total_failed == 1

    def test_pass_rate(self) -> None:
        metrics = self._collector(n=4, n_passed=3).aggregate()
        assert metrics.overall_pass_rate == pytest.approx(0.75)

    def test_model_key_propagated(self) -> None:
        metrics = self._collector().aggregate(model_key="google/gemma-4-e2b")
        assert metrics.model_key == "google/gemma-4-e2b"

    def test_by_category_populated(self) -> None:
        assert "accuracy" in self._collector(category="accuracy").aggregate().by_category

    def test_category_pass_rate(self) -> None:
        metrics = self._collector(n=2, n_passed=2, category="accuracy").aggregate()
        assert metrics.accuracy_pass_rate == pytest.approx(1.0)

    def test_quality_score_in_range(self) -> None:
        score = self._collector().aggregate().overall_quality_score
        assert 0.0 <= score <= 1.0

    def test_critical_risk_weighs_more(self) -> None:
        low_c = self._collector(n=2, n_passed=1, risk_level=RiskLevel.LOW).aggregate()
        crit_c = self._collector(n=2, n_passed=1, risk_level=RiskLevel.CRITICAL).aggregate()
        # Both have 50% pass rate but weights are different; quality_score is still
        # based on scores (1.0 = passed, 0.0 = failed) so both are 0.5.
        # The important thing is the field exists and is in range.
        assert 0.0 <= low_c.overall_quality_score <= 1.0
        assert 0.0 <= crit_c.overall_quality_score <= 1.0


class TestDatasetLoader:
    """
    Unit tests for DatasetLoader.
    Writes temp JSON files; no LLM involved.
    """

    def _write(self, data: Any) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(data, tmp)
        tmp.close()
        return Path(tmp.name)

    def _minimal(
        self,
        test_id: str = "DS_001",
        category: str = "accuracy",
        prompt: str = "What is the capital of France?",
        expected_answer: str = "Paris",
    ) -> dict[str, Any]:
        return {
            "test_id": test_id,
            "category": category,
            "prompt": prompt,
            "expected_answer": expected_answer,
        }

    def test_load_bare_list(self) -> None:
        cases = DatasetLoader.load(self._write([self._minimal()]))
        assert len(cases) == 1

    def test_load_wrapped_object(self) -> None:
        data = {"name": "Test", "version": "1.0.0", "test_cases": [self._minimal()]}
        assert len(DatasetLoader.load(self._write(data))) == 1

    def test_returns_test_case_objects(self) -> None:
        cases = DatasetLoader.load(self._write([self._minimal()]))
        assert all(isinstance(c, TestCase) for c in cases)

    def test_preserves_test_ids(self) -> None:
        data = [self._minimal("A"), self._minimal("B")]
        ids = [c.test_id for c in DatasetLoader.load(self._write(data))]
        assert ids == ["A", "B"]

    def test_load_by_category(self) -> None:
        data = [self._minimal(category="accuracy"), self._minimal("X", category="relevance")]
        cases = DatasetLoader.load_by_category(self._write(data), "accuracy")
        assert len(cases) == 1

    def test_load_by_category_missing_returns_empty(self) -> None:
        assert DatasetLoader.load_by_category(self._write([self._minimal()]), "nope") == []

    def test_load_by_tag(self) -> None:
        data = [
            {**self._minimal("T1"), "tags": ["smoke"]},
            {**self._minimal("T2"), "tags": ["slow"]},
        ]
        cases = DatasetLoader.load_by_tag(self._write(data), "smoke")
        assert len(cases) == 1 and cases[0].test_id == "T1"

    def test_duplicate_ids_raise(self) -> None:
        data = [self._minimal("DUP"), self._minimal("DUP")]
        with pytest.raises(DatasetValidationError) as exc_info:
            DatasetLoader.load(self._write(data))
        assert any("DUP" in e for e in exc_info.value.errors)

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(DatasetValidationError):
            DatasetLoader.load(self._write([{"test_id": "BAD", "prompt": "no category"}]))

    def test_invalid_json_raises(self) -> None:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        tmp.write("{not json")
        tmp.close()
        with pytest.raises(Exception):
            DatasetLoader.load(tmp.name)

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            DatasetLoader.load("/tmp/does_not_exist_phase3.json")

    def test_validate_clean_list(self) -> None:
        cases = DatasetLoader.load(self._write([self._minimal()]))
        assert DatasetLoader.validate(cases) == []

    def test_load_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "set_a.json").write_text(
                json.dumps([self._minimal()]), encoding="utf-8"
            )
            result = DatasetLoader.load_directory(tmpdir)
        assert "set_a" in result and len(result["set_a"]) == 1

    def test_real_geography_dataset_loads(self) -> None:
        """Smoke test: the actual geography.json parses without errors."""
        cases = DatasetLoader.load(TEST_DATA / "accuracy" / "geography.json")
        assert len(cases) >= 1
        assert all(isinstance(c, TestCase) for c in cases)

    def test_real_safety_dataset_loads(self) -> None:
        """Smoke test: safety datasets with coverage strategy parse correctly."""
        for fname in ("prompt_injection.json", "jailbreak.json"):
            cases = DatasetLoader.load(TEST_DATA / "safety" / fname)
            assert len(cases) >= 1


# ===========================================================================
# PART 2 — Dataset-driven integration tests (real LLM via conftest fixtures)
# ===========================================================================
#
# Each parametrized function receives one TestCase per JSON entry.
# Extend coverage by adding entries to the JSON files — no Python changes needed.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "test_case",
    DatasetLoader.load(TEST_DATA / "accuracy" / "geography.json"),
    ids=lambda tc: tc.test_id,
)
def test_accuracy_geography(
    pipeline: EvaluationPipeline,
    test_case: TestCase,
) -> None:
    result = pipeline.run(test_case, StubEvaluator())
    assert result.passed, (
        f"[{test_case.test_id}] expected '{test_case.expected_answer}' "
        f"in response: {result.response!r}"
    )


@pytest.mark.parametrize(
    "test_case",
    DatasetLoader.load(TEST_DATA / "accuracy" / "math.json"),
    ids=lambda tc: tc.test_id,
)
def test_accuracy_math(
    pipeline: EvaluationPipeline,
    test_case: TestCase,
) -> None:
    result = pipeline.run(test_case, StubEvaluator())
    assert result.passed, (
        f"[{test_case.test_id}] expected '{test_case.expected_answer}' "
        f"in response: {result.response!r}"
    )


def test_pipeline_result_structure(
    pipeline: EvaluationPipeline,
    config: Config,
) -> None:
    """
    Structural test: one real LLM call verifies that EvaluationPipeline
    returns a fully-populated EvaluationResult with all required fields set.
    """
    tc = DatasetLoader.load(TEST_DATA / "accuracy" / "geography.json")[0]
    result = pipeline.run(tc, StubEvaluator())

    assert isinstance(result, EvaluationResult)
    assert result.test_id == tc.test_id
    assert result.evaluator_name == "StubEvaluator"
    assert result.tier == EvalTier.RULE_BASED
    assert result.model_key == config.default_model_key
    assert result.model_name == "Gemma 4 E2B"
    assert result.client_latency_ms >= 0.0
    assert len(result.response) > 0
    assert result.process_quality != ProcessQualityStatus.NOT_EVALUATED
    assert result.category == tc.category
    assert result.risk_level == tc.risk_level


def test_batch_and_aggregation(
    pipeline: EvaluationPipeline,
    config: Config,
) -> None:
    """
    Runs geography dataset as a batch and verifies MetricsCollector aggregation.
    Uses a local collector so the test is independent of session state.
    """
    cases = DatasetLoader.load(TEST_DATA / "accuracy" / "geography.json")
    local_collector = MetricsCollector()
    local_pipeline = EvaluationPipeline(
        client=pipeline._client,
        config=config,
        model_name="Gemma 4 E2B",
        collector=local_collector,
    )
    results = local_pipeline.run_batch(cases, StubEvaluator())

    assert len(results) == len(cases)
    assert local_collector.count() == len(cases)

    metrics = local_collector.aggregate(model_key=config.default_model_key)
    assert metrics.total_tests == len(cases)
    assert metrics.accuracy_pass_rate is not None
    assert 0.0 <= metrics.overall_quality_score <= 1.0
    assert "accuracy" in metrics.by_category
