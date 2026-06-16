"""
Accuracy evaluation tests — verifies factual correctness using AccuracyEvaluator.

Datasets:
  test_data/accuracy/geography.json  — capital cities (COMPOSITE strategy)
  test_data/accuracy/math.json       — arithmetic & word problems (MATH strategy)
  test_data/accuracy/history.json    — historical facts (EXACT / COMPOSITE strategy)

Each parametrized function drives one test case per JSON entry.
To add new test cases, append entries to the JSON files — no Python changes needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluators.accuracy_evaluator import AccuracyEvaluator
from src.pipeline.evaluation_pipeline import EvaluationPipeline
from src.schemas.test_case import TestCase
from src.utils.dataset_loader import DatasetLoader

TEST_DATA = Path(__file__).parent.parent / "test_data"


# ---------------------------------------------------------------------------
# Geography — capitals of countries
# ---------------------------------------------------------------------------

@pytest.mark.accuracy
@pytest.mark.parametrize(
    "test_case",
    DatasetLoader.load(TEST_DATA / "accuracy" / "geography.json"),
    ids=lambda tc: tc.test_id,
)
def test_accuracy_geography(
    pipeline: EvaluationPipeline,
    accuracy_evaluator: AccuracyEvaluator,
    test_case: TestCase,
) -> None:
    result = pipeline.run(test_case, accuracy_evaluator)
    assert result.passed, (
        f"[{test_case.test_id}] {result.failure_reason}\n"
        f"  strategy : {test_case.evaluation_strategy.value}\n"
        f"  expected : {test_case.expected_answer!r}\n"
        f"  response : {result.response!r}\n"
        f"  score    : {result.score:.3f} (threshold {test_case.pass_threshold})"
    )


# ---------------------------------------------------------------------------
# Math — arithmetic and word problems
# ---------------------------------------------------------------------------

@pytest.mark.accuracy
@pytest.mark.parametrize(
    "test_case",
    DatasetLoader.load(TEST_DATA / "accuracy" / "math.json"),
    ids=lambda tc: tc.test_id,
)
def test_accuracy_math(
    pipeline: EvaluationPipeline,
    accuracy_evaluator: AccuracyEvaluator,
    test_case: TestCase,
) -> None:
    result = pipeline.run(test_case, accuracy_evaluator)
    assert result.passed, (
        f"[{test_case.test_id}] {result.failure_reason}\n"
        f"  strategy : {test_case.evaluation_strategy.value}\n"
        f"  expected : {test_case.expected_answer!r}\n"
        f"  response : {result.response!r}\n"
        f"  score    : {result.score:.3f} (threshold {test_case.pass_threshold})\n"
        f"  signals  : {result.sub_scores}"
    )


# ---------------------------------------------------------------------------
# History — dates and historical figures
# ---------------------------------------------------------------------------

@pytest.mark.accuracy
@pytest.mark.parametrize(
    "test_case",
    DatasetLoader.load(TEST_DATA / "accuracy" / "history.json"),
    ids=lambda tc: tc.test_id,
)
def test_accuracy_history(
    pipeline: EvaluationPipeline,
    accuracy_evaluator: AccuracyEvaluator,
    test_case: TestCase,
) -> None:
    result = pipeline.run(test_case, accuracy_evaluator)
    assert result.passed, (
        f"[{test_case.test_id}] {result.failure_reason}\n"
        f"  strategy : {test_case.evaluation_strategy.value}\n"
        f"  expected : {test_case.expected_answer!r}\n"
        f"  response : {result.response!r}\n"
        f"  score    : {result.score:.3f} (threshold {test_case.pass_threshold})"
    )
