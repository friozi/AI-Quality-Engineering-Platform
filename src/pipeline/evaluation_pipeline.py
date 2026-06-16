"""
EvaluationPipeline — the single point of execution for every test case.

This is the ONLY module that calls client.chat().  All evaluators receive
an LLMResponse that was produced here; they never touch the client directly.

Responsibilities:
  1. Call client.chat(test_case) → LLMResponse
  2. Forward (test_case, response) to the evaluator → EvaluationResult
  3. Annotate the result with model context (key, name)
  4. Hand the result to the MetricsCollector (if one is registered)
  5. Return the result to the test function

Error contract:
  LLMEmptyResponseError  → converted to FAIL EvaluationResult (model quality issue)
  LLMConnectionError     → re-raised  (infrastructure issue — stop the test)
  LLMServerError         → re-raised  (infrastructure issue — stop the test)
  LLMClientError         → re-raised  (framework bug — stop the test)
  LLMParseError          → re-raised  (API schema changed — stop the test)
  Any other exception    → re-raised after logging
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from loguru import logger

from src.clients.base_client import LLMEmptyResponseError
from src.evaluators.base_evaluator import BaseEvaluator
from src.schemas.evaluation_result import EvaluationResult
from src.schemas.test_case import TestCase
from src.utils.config import Config

if TYPE_CHECKING:
    from src.clients.base_client import BaseClient
    from src.metrics.metrics_calculator import MetricsCollector


class EvaluationPipeline:
    """
    Orchestrates one evaluation: client call → evaluator → result → collector.

    Typical usage in a pytest test::

        result = pipeline.run(test_case, AccuracyEvaluator())
        assert result.passed, result.failure_reason

    Batch usage::

        results = pipeline.run_batch(test_cases, AccuracyEvaluator())
        pass_rate = sum(r.passed for r in results) / len(results)
    """

    def __init__(
        self,
        client: "BaseClient",
        config: Config,
        model_name: str = "",
        collector: Optional["MetricsCollector"] = None,
    ) -> None:
        """
        Args:
            client     : LocalLLMClient or MockClient.
            config     : framework configuration (used for model_key annotation).
            model_name : human-readable display name (e.g. "Gemma 4 E2B").
                         Pass empty string to skip annotation.
            collector  : MetricsCollector to record every result into.
                         Can be set later via set_collector().
        """
        self._client = client
        self._config = config
        self._model_name = model_name
        self._collector = collector

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def set_collector(self, collector: "MetricsCollector") -> None:
        """Register a MetricsCollector after construction (e.g. from conftest)."""
        self._collector = collector

    def run(
        self,
        test_case: TestCase,
        evaluator: BaseEvaluator,
    ) -> EvaluationResult:
        """
        Execute the full evaluation loop for one test case.

        Returns an EvaluationResult in all cases where the infrastructure
        is healthy.  Never returns None.  Raises only on infrastructure
        failures (network down, parse error, etc.).
        """
        logger.debug(
            f"[pipeline] START test_id={test_case.test_id} "
            f"evaluator={evaluator.EVALUATOR_NAME} "
            f"category={test_case.category}"
        )

        t_pipeline_start = time.perf_counter()

        result = self._execute(test_case, evaluator)

        pipeline_ms = (time.perf_counter() - t_pipeline_start) * 1000

        # Annotate with model context from config.
        result.model_key = self._config.default_model_key
        result.model_name = self._model_name

        self._log_result(result, pipeline_ms)

        if self._collector is not None:
            self._collector.record(result)

        return result

    def run_batch(
        self,
        test_cases: list[TestCase],
        evaluator: BaseEvaluator,
    ) -> list[EvaluationResult]:
        """
        Run multiple test cases sequentially through the same evaluator.

        Stops on the first infrastructure error (connection down, parse error).
        Empty-response failures are captured per-test as FAIL results and do
        not stop the batch.
        """
        results: list[EvaluationResult] = []
        for tc in test_cases:
            results.append(self.run(tc, evaluator))
        return results

    # ------------------------------------------------------------------ #
    # Internal execution
    # ------------------------------------------------------------------ #

    def _execute(
        self,
        test_case: TestCase,
        evaluator: BaseEvaluator,
    ) -> EvaluationResult:
        """
        Core execution: call client → call evaluator → return result.

        LLMEmptyResponseError is the only client exception that is converted
        to a FAIL result.  Everything else propagates.
        """
        from src.schemas.llm_response import LLMResponse  # local import — avoids circular

        try:
            response: LLMResponse = self._client.chat(test_case)
        except LLMEmptyResponseError as exc:
            logger.warning(
                f"[pipeline] Empty response for test_id={test_case.test_id}: {exc}"
            )
            return BaseEvaluator._empty_response_result(
                test_case,
                evaluator_name=evaluator.EVALUATOR_NAME,
                tier=evaluator.TIER,
            )

        # Infrastructure exceptions (connection, server, parse) propagate here.
        return evaluator.evaluate(test_case, response)

    @staticmethod
    def _log_result(result: EvaluationResult, pipeline_ms: float) -> None:
        verdict = "PASS" if result.passed else "FAIL"
        logger.info(
            f"[pipeline] {verdict} "
            f"test_id={result.test_id} "
            f"score={result.score:.3f} "
            f"latency={result.client_latency_ms:.0f}ms "
            f"pipeline={pipeline_ms:.0f}ms "
            f"evaluator={result.evaluator_name}"
        )
        if not result.passed and result.failure_reason:
            logger.debug(
                f"[pipeline] failure_reason={result.failure_reason!r} "
                f"recommendation={result.recommendation!r}"
            )
