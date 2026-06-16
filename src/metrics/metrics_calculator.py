"""
MetricsCollector and AggregatedMetrics — thread-safe result accumulation
and statistical aggregation for the QA framework.

Design:
  MetricsCollector is a session-scoped singleton (created once in conftest.py).
  It is safe for concurrent writes from pytest-xdist workers in the same
  process because all mutations hold a threading.Lock.

  When all tests finish, conftest calls collector.aggregate() to compute
  summary statistics, which are then passed to the reporters.

AggregatedMetrics is a Pydantic model so it can be serialised directly
to JSON or dict for the dashboard and HTML report.
"""

from __future__ import annotations

import threading
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from src.schemas.evaluation_result import EvaluationResult
from src.schemas.test_case import RiskLevel

# ---------------------------------------------------------------------------
# Risk weight table — CRITICAL failures penalise the quality score more.
# ---------------------------------------------------------------------------
_RISK_WEIGHTS: dict[RiskLevel, float] = {
    RiskLevel.LOW: 1.0,
    RiskLevel.MEDIUM: 2.0,
    RiskLevel.HIGH: 3.0,
    RiskLevel.CRITICAL: 4.0,
}


# ---------------------------------------------------------------------------
# Per-category breakdown model
# ---------------------------------------------------------------------------

class CategoryMetrics(BaseModel):
    category: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    avg_score: float
    p50_latency_ms: float
    p95_latency_ms: float


# ---------------------------------------------------------------------------
# Top-level aggregated metrics (dashboard data)
# ---------------------------------------------------------------------------

class AggregatedMetrics(BaseModel):
    """
    Full summary of a QA run.  Serialised to reports/dashboard_summary.json
    and consumed by HTMLReporter and DashboardSummary.
    """

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_key: str = ""
    model_name: str = ""

    # ── Overall counts ──────────────────────────────────────────────────
    total_tests: int = 0
    total_passed: int = 0
    total_failed: int = 0
    overall_pass_rate: float = 0.0
    overall_avg_score: float = 0.0

    # ── Latency (ms) ────────────────────────────────────────────────────
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # ── Token stats ─────────────────────────────────────────────────────
    avg_tokens_per_second: float = 0.0
    avg_reasoning_ratio: float = 0.0

    # ── Per-category pass rates (None = category not tested) ────────────
    accuracy_pass_rate: Optional[float] = None
    relevance_pass_rate: Optional[float] = None
    hallucination_pass_rate: Optional[float] = None
    prompt_injection_pass_rate: Optional[float] = None
    jailbreak_pass_rate: Optional[float] = None
    reasoning_pass_rate: Optional[float] = None
    latency_pass_rate: Optional[float] = None

    # ── Per-category average scores ─────────────────────────────────────
    accuracy_avg_score: Optional[float] = None
    reasoning_avg_score: Optional[float] = None

    # ── Process quality breakdown ────────────────────────────────────────
    ideal_count: int = 0      # correct answer + correct reasoning
    lucky_count: int = 0      # correct answer + inconsistent reasoning
    error_count: int = 0      # wrong answer  + consistent reasoning
    fail_count: int = 0       # wrong answer  + poor reasoning

    # ── Weighted quality score ───────────────────────────────────────────
    overall_quality_score: float = 0.0
    """
    Weighted mean of scores where CRITICAL=4×, HIGH=3×, MEDIUM=2×, LOW=1×.
    Ranges 0.0–1.0.  Higher is better.
    """

    # ── Per-category detail ──────────────────────────────────────────────
    by_category: dict[str, CategoryMetrics] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Thread-safe accumulator for EvaluationResult objects produced during a
    pytest session.

    Usage (in conftest.py session fixture)::

        collector = MetricsCollector()
        pipeline.set_collector(collector)

        yield  # tests run

        metrics = collector.aggregate(model_key=config.default_model_key)
        JSONReporter().write(metrics, config.json_report_dir / "dashboard.json")
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._results: list[EvaluationResult] = []

    # ------------------------------------------------------------------ #
    # Write (called from pipeline, possibly across threads)
    # ------------------------------------------------------------------ #

    def record(self, result: EvaluationResult) -> None:
        """Append *result* to the session store.  Thread-safe."""
        with self._lock:
            self._results.append(result)

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #

    def get_results(self) -> list[EvaluationResult]:
        """Return a snapshot of all recorded results."""
        with self._lock:
            return list(self._results)

    def count(self) -> int:
        with self._lock:
            return len(self._results)

    def clear(self) -> None:
        """Reset the collector (useful between test runs in the same process)."""
        with self._lock:
            self._results.clear()

    # ------------------------------------------------------------------ #
    # Aggregation
    # ------------------------------------------------------------------ #

    def aggregate(
        self,
        model_key: str = "",
        model_name: str = "",
    ) -> AggregatedMetrics:
        """
        Compute summary statistics across all recorded results.

        Safe to call multiple times — reads a snapshot and does not modify
        the internal store.
        """
        results = self.get_results()

        if not results:
            return AggregatedMetrics(model_key=model_key, model_name=model_name)

        # ── Basic counts ─────────────────────────────────────────────────
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        # ── Scores ───────────────────────────────────────────────────────
        scores = [r.score for r in results]
        avg_score = float(np.mean(scores))

        # ── Latency ──────────────────────────────────────────────────────
        latencies = [r.client_latency_ms for r in results if r.client_latency_ms > 0]
        avg_lat = float(np.mean(latencies)) if latencies else 0.0
        p50_lat = float(np.percentile(latencies, 50)) if latencies else 0.0
        p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
        p99_lat = float(np.percentile(latencies, 99)) if latencies else 0.0

        # ── Token stats ───────────────────────────────────────────────────
        tps_vals = [r.tokens_per_second for r in results if r.tokens_per_second]
        avg_tps = float(np.mean(tps_vals)) if tps_vals else 0.0

        rr_vals = [r.reasoning_ratio for r in results if r.has_reasoning]
        avg_rr = float(np.mean(rr_vals)) if rr_vals else 0.0

        # ── Per-category breakdown ────────────────────────────────────────
        by_cat: dict[str, list[EvaluationResult]] = defaultdict(list)
        for r in results:
            by_cat[r.category].append(r)

        cat_metrics = {
            cat: _compute_category_metrics(cat, rs)
            for cat, rs in by_cat.items()
        }

        # ── Category-specific pass rates ──────────────────────────────────
        def _cat_pass_rate(name: str) -> Optional[float]:
            return cat_metrics[name].pass_rate if name in cat_metrics else None

        def _cat_avg_score(name: str) -> Optional[float]:
            return cat_metrics[name].avg_score if name in cat_metrics else None

        # ── Process quality breakdown ─────────────────────────────────────
        from src.schemas.evaluation_result import ProcessQualityStatus  # local import
        pq_counts = defaultdict(int)
        for r in results:
            pq_counts[r.process_quality] += 1

        # ── Weighted quality score ─────────────────────────────────────────
        weighted_sum = sum(
            r.score * _RISK_WEIGHTS.get(r.risk_level, 1.0) for r in results
        )
        weight_total = sum(_RISK_WEIGHTS.get(r.risk_level, 1.0) for r in results)
        quality_score = weighted_sum / weight_total if weight_total > 0 else 0.0

        return AggregatedMetrics(
            model_key=model_key,
            model_name=model_name,
            # counts
            total_tests=total,
            total_passed=passed,
            total_failed=failed,
            overall_pass_rate=round(passed / total, 4),
            overall_avg_score=round(avg_score, 4),
            # latency
            avg_latency_ms=round(avg_lat, 2),
            p50_latency_ms=round(p50_lat, 2),
            p95_latency_ms=round(p95_lat, 2),
            p99_latency_ms=round(p99_lat, 2),
            # tokens
            avg_tokens_per_second=round(avg_tps, 2),
            avg_reasoning_ratio=round(avg_rr, 4),
            # per-category pass rates
            accuracy_pass_rate=_cat_pass_rate("accuracy"),
            relevance_pass_rate=_cat_pass_rate("relevance"),
            hallucination_pass_rate=_cat_pass_rate("hallucination"),
            prompt_injection_pass_rate=_cat_pass_rate("prompt_injection"),
            jailbreak_pass_rate=_cat_pass_rate("jailbreak"),
            reasoning_pass_rate=_cat_pass_rate("reasoning"),
            latency_pass_rate=_cat_pass_rate("latency"),
            # per-category scores
            accuracy_avg_score=_cat_avg_score("accuracy"),
            reasoning_avg_score=_cat_avg_score("reasoning"),
            # process quality
            ideal_count=pq_counts[ProcessQualityStatus.IDEAL],
            lucky_count=pq_counts[ProcessQualityStatus.LUCKY],
            error_count=pq_counts[ProcessQualityStatus.ERROR],
            fail_count=pq_counts[ProcessQualityStatus.FAIL],
            # weighted quality
            overall_quality_score=round(quality_score, 4),
            # breakdown
            by_category=cat_metrics,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_category_metrics(
    category: str,
    results: list[EvaluationResult],
) -> CategoryMetrics:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    scores = [r.score for r in results]
    latencies = [r.client_latency_ms for r in results if r.client_latency_ms > 0]

    return CategoryMetrics(
        category=category,
        total=total,
        passed=passed,
        failed=total - passed,
        pass_rate=round(passed / total, 4) if total else 0.0,
        avg_score=round(float(np.mean(scores)), 4) if scores else 0.0,
        p50_latency_ms=round(float(np.percentile(latencies, 50)), 2) if latencies else 0.0,
        p95_latency_ms=round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
    )
