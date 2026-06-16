"""
Reasoning evaluation tests — verifies chain-of-thought quality using ReasoningEvaluator.

Datasets:
  test_data/reasoning/logic.json           — syllogisms with required_reasoning_steps
  test_data/reasoning/chain_of_thought.json — multi-step problems (math strategy + reasoning)

Note: ReasoningEvaluator returns score=0.2 and fails if the model produces no
reasoning block. This is intentional — these tests genuinely check reasoning quality.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluators.reasoning_evaluator import ReasoningEvaluator
from src.pipeline.evaluation_pipeline import EvaluationPipeline
from src.schemas.test_case import TestCase
from src.utils.dataset_loader import DatasetLoader

TEST_DATA = Path(__file__).parent.parent / "test_data"


# ---------------------------------------------------------------------------
# Logic — syllogism reasoning
# ---------------------------------------------------------------------------

@pytest.mark.reasoning
@pytest.mark.parametrize(
    "test_case",
    DatasetLoader.load(TEST_DATA / "reasoning" / "logic.json"),
    ids=lambda tc: tc.test_id,
)
def test_reasoning_logic(
    pipeline: EvaluationPipeline,
    reasoning_evaluator: ReasoningEvaluator,
    test_case: TestCase,
) -> None:
    result = pipeline.run(test_case, reasoning_evaluator)
    assert result.passed, (
        f"[{test_case.test_id}] {result.failure_reason}\n"
        f"  expected concepts : {test_case.required_concepts}\n"
        f"  required steps    : {test_case.required_reasoning_steps}\n"
        f"  response          : {result.response!r}\n"
        f"  score             : {result.score:.3f} (threshold {test_case.pass_threshold})\n"
        f"  signals           : {result.sub_scores}"
    )


# ---------------------------------------------------------------------------
# Chain-of-thought — multi-step problems
# ---------------------------------------------------------------------------

@pytest.mark.reasoning
@pytest.mark.parametrize(
    "test_case",
    DatasetLoader.load(TEST_DATA / "reasoning" / "chain_of_thought.json"),
    ids=lambda tc: tc.test_id,
)
def test_reasoning_chain_of_thought(
    pipeline: EvaluationPipeline,
    reasoning_evaluator: ReasoningEvaluator,
    test_case: TestCase,
) -> None:
    result = pipeline.run(test_case, reasoning_evaluator)
    assert result.passed, (
        f"[{test_case.test_id}] {result.failure_reason}\n"
        f"  expected          : {test_case.expected_answer!r}\n"
        f"  required concepts : {test_case.required_concepts}\n"
        f"  response          : {result.response!r}\n"
        f"  score             : {result.score:.3f} (threshold {test_case.pass_threshold})\n"
        f"  signals           : {result.sub_scores}"
    )
