"""
Session-scoped infrastructure fixtures for the QA framework test suite.

Rule: fixtures here provide shared infrastructure only.
      Test cases always come from JSON datasets via DatasetLoader.

Fixture scopes:
  session : config, llm_client, session_collector, pipeline, all evaluators
            — one instance per pytest run; reused across all test files.

Evaluators are stateless and cheap, but session-scoped for consistency and
so test functions declare their infrastructure dependencies explicitly.

Phase 8 — auto-reporting:
  _auto_report runs after all tests finish and calls ReportRunner to generate
  JSON, CSV, and HTML reports under config.report_dir/run_<timestamp>/.
"""

from __future__ import annotations

import pytest
from loguru import logger

from src.clients.local_llm_client import LocalLLMClient
from src.evaluators.accuracy_evaluator import AccuracyEvaluator
from src.evaluators.hallucination_evaluator import HallucinationEvaluator
from src.evaluators.jailbreak_evaluator import JailbreakEvaluator
from src.evaluators.latency_evaluator import LatencyEvaluator
from src.evaluators.prompt_injection_evaluator import PromptInjectionEvaluator
from src.evaluators.reasoning_evaluator import ReasoningEvaluator
from src.evaluators.relevance_evaluator import RelevanceEvaluator
from src.metrics.metrics_calculator import MetricsCollector
from src.pipeline.evaluation_pipeline import EvaluationPipeline
from src.reporters.report_runner import ReportRunner
from src.utils.config import Config
from src.utils.logger import configure_logger

_MODEL_NAME = "Gemma 4 E2B"


# ---------------------------------------------------------------------------
# Session fixtures — created once per pytest session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def config() -> Config:
    """
    Framework configuration loaded from .env.
    Set MOCK_MODE=true in .env to run offline against MockClient.
    """
    return Config()


@pytest.fixture(scope="session", autouse=True)
def _configure_logging(config: Config) -> None:
    """Set up loguru sinks once for the whole session."""
    configure_logger(log_level=config.log_level, log_file=config.log_file)


@pytest.fixture(scope="session")
def llm_client(config: Config) -> LocalLLMClient:
    """
    Single HTTP client shared across the entire test session.
    Connection pooling is handled by HTTPX internally.
    """
    with LocalLLMClient(config) as client:
        reachable = client.health_check()
        if not reachable:
            logger.warning(
                f"LLM server at {config.base_url} did not respond to health check. "
                "Tests may fail or timeout."
            )
        yield client


@pytest.fixture(scope="session")
def session_collector() -> MetricsCollector:
    """
    Session-wide MetricsCollector.
    Accumulates every EvaluationResult produced by the session pipeline.
    Access it in test teardown or reporting hooks to compute aggregated metrics.
    """
    return MetricsCollector()


@pytest.fixture(scope="session")
def pipeline(
    llm_client: LocalLLMClient,
    config: Config,
    session_collector: MetricsCollector,
) -> EvaluationPipeline:
    """
    The main EvaluationPipeline, wired to the session client and collector.
    All dataset-driven test functions receive this fixture and call pipeline.run().
    """
    return EvaluationPipeline(
        client=llm_client,
        config=config,
        model_name=_MODEL_NAME,
        collector=session_collector,
    )


# ---------------------------------------------------------------------------
# Phase 8 — session teardown: auto-generate reports after all tests finish
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _auto_report(
    config: Config,
    session_collector: MetricsCollector,
    pipeline: EvaluationPipeline,  # noqa: ARG001 — ensures pipeline/collector are fully populated
) -> None:
    """
    Generate JSON, CSV, and HTML reports at the end of every test session.

    Runs unconditionally so partial runs (e.g. -k filter, early abort) still
    produce a report for however many tests completed. Exceptions are caught
    and logged so report failures never mask test failures.
    """
    yield

    results = session_collector.get_results()
    if not results:
        logger.info("No results recorded — skipping report generation.")
        return

    metrics = session_collector.aggregate(
        model_key=config.default_model_key,
        model_name=_MODEL_NAME,
    )

    try:
        run_dir = ReportRunner(config).run(metrics, results)
        logger.info(
            f"QA reports written to: {run_dir.resolve()}\n"
            f"  {metrics.total_tests} tests | "
            f"pass rate {metrics.overall_pass_rate:.1%} | "
            f"quality score {metrics.overall_quality_score:.3f}"
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"Report generation failed: {exc}")


# ---------------------------------------------------------------------------
# Evaluator fixtures — one per evaluator, session-scoped, stateless
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def accuracy_evaluator() -> AccuracyEvaluator:
    return AccuracyEvaluator()


@pytest.fixture(scope="session")
def relevance_evaluator() -> RelevanceEvaluator:
    return RelevanceEvaluator()


@pytest.fixture(scope="session")
def hallucination_evaluator() -> HallucinationEvaluator:
    return HallucinationEvaluator()


@pytest.fixture(scope="session")
def prompt_injection_evaluator() -> PromptInjectionEvaluator:
    return PromptInjectionEvaluator()


@pytest.fixture(scope="session")
def jailbreak_evaluator() -> JailbreakEvaluator:
    return JailbreakEvaluator()


@pytest.fixture(scope="session")
def latency_evaluator() -> LatencyEvaluator:
    return LatencyEvaluator()


@pytest.fixture(scope="session")
def reasoning_evaluator() -> ReasoningEvaluator:
    return ReasoningEvaluator()
