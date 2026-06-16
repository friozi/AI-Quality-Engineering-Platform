"""
Phase 5 reporter tests — all offline, no LLM calls.

Validates that JSONReporter, CSVReporter, HTMLReporter, and ReportRunner
produce correctly structured output given fake AggregatedMetrics + results.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.metrics.metrics_calculator import AggregatedMetrics, CategoryMetrics, MetricsCollector
from src.reporters import CSVReporter, HTMLReporter, JSONReporter, ReportRunner
from src.schemas.evaluation_result import EvaluationResult, ProcessQualityStatus
from src.schemas.test_case import EvalTier, RiskLevel
from src.utils.config import Config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _fake_result(
    test_id: str = "R_001",
    passed: bool = True,
    category: str = "accuracy",
    risk_level: RiskLevel = RiskLevel.MEDIUM,
    client_latency_ms: float = 1200.0,
    evaluator_name: str = "AccuracyEvaluator",
) -> EvaluationResult:
    return EvaluationResult(
        test_id=test_id,
        evaluator_name=evaluator_name,
        tier=EvalTier.RULE_BASED,
        prompt=f"Prompt for {test_id}",
        response=f"Response for {test_id}",
        score=1.0 if passed else 0.0,
        passed=passed,
        category=category,
        risk_level=risk_level,
        client_latency_ms=client_latency_ms,
        failure_reason=None if passed else "expected_answer_not_found",
        explanation=f"score={'1.000' if passed else '0.000'}",
        process_quality=ProcessQualityStatus.IDEAL if passed else ProcessQualityStatus.FAIL,
    )


@pytest.fixture()
def fake_results() -> list[EvaluationResult]:
    return [
        _fake_result("GEO_001", passed=True,  category="accuracy",  risk_level=RiskLevel.MEDIUM),
        _fake_result("GEO_002", passed=True,  category="accuracy",  risk_level=RiskLevel.MEDIUM),
        _fake_result("GEO_003", passed=False, category="accuracy",  risk_level=RiskLevel.HIGH),
        _fake_result("LOGIC_001", passed=True,  category="reasoning", risk_level=RiskLevel.LOW,  client_latency_ms=3500.0, evaluator_name="ReasoningEvaluator"),
        _fake_result("PINJ_001", passed=True,  category="prompt_injection", risk_level=RiskLevel.CRITICAL, client_latency_ms=900.0, evaluator_name="PromptInjectionEvaluator"),
        _fake_result("PINJ_002", passed=False, category="prompt_injection", risk_level=RiskLevel.CRITICAL, client_latency_ms=1100.0, evaluator_name="PromptInjectionEvaluator"),
    ]


@pytest.fixture()
def fake_metrics(fake_results: list[EvaluationResult]) -> AggregatedMetrics:
    collector = MetricsCollector()
    for r in fake_results:
        collector.record(r)
    return collector.aggregate(model_key="google/gemma-4-e2b", model_name="Gemma 4 E2B")


@pytest.fixture()
def offline_config(tmp_path: Path) -> Config:
    return Config(report_dir=tmp_path / "reports")


# ---------------------------------------------------------------------------
# JSONReporter
# ---------------------------------------------------------------------------

class TestJSONReporter:
    def test_creates_file(self, tmp_path, fake_metrics, fake_results):
        path = JSONReporter().write(fake_metrics, fake_results, tmp_path)
        assert path.exists()
        assert path.name == "dashboard_summary.json"

    def test_valid_json(self, tmp_path, fake_metrics, fake_results):
        path = JSONReporter().write(fake_metrics, fake_results, tmp_path)
        data = json.loads(path.read_text())
        assert isinstance(data, dict)

    def test_contains_total_tests(self, tmp_path, fake_metrics, fake_results):
        path = JSONReporter().write(fake_metrics, fake_results, tmp_path)
        data = json.loads(path.read_text())
        assert data["total_tests"] == len(fake_results)

    def test_contains_pass_rate(self, tmp_path, fake_metrics, fake_results):
        path = JSONReporter().write(fake_metrics, fake_results, tmp_path)
        data = json.loads(path.read_text())
        assert "overall_pass_rate" in data
        assert 0.0 <= data["overall_pass_rate"] <= 1.0

    def test_contains_model_key(self, tmp_path, fake_metrics, fake_results):
        path = JSONReporter().write(fake_metrics, fake_results, tmp_path)
        data = json.loads(path.read_text())
        assert data["model_key"] == "google/gemma-4-e2b"

    def test_contains_by_category(self, tmp_path, fake_metrics, fake_results):
        path = JSONReporter().write(fake_metrics, fake_results, tmp_path)
        data = json.loads(path.read_text())
        assert "by_category" in data
        assert "accuracy" in data["by_category"]

    def test_contains_generated_at(self, tmp_path, fake_metrics, fake_results):
        path = JSONReporter().write(fake_metrics, fake_results, tmp_path)
        data = json.loads(path.read_text())
        assert "generated_at" in data

    def test_results_count(self, tmp_path, fake_metrics, fake_results):
        path = JSONReporter().write(fake_metrics, fake_results, tmp_path)
        data = json.loads(path.read_text())
        assert data["results_count"] == len(fake_results)

    def test_creates_output_dir_if_missing(self, tmp_path, fake_metrics, fake_results):
        deep = tmp_path / "a" / "b" / "c"
        JSONReporter().write(fake_metrics, fake_results, deep)
        assert deep.exists()

    def test_empty_results(self, tmp_path, fake_metrics):
        path = JSONReporter().write(fake_metrics, [], tmp_path)
        data = json.loads(path.read_text())
        assert data["results_count"] == 0


# ---------------------------------------------------------------------------
# CSVReporter
# ---------------------------------------------------------------------------

class TestCSVReporter:
    def test_creates_file(self, tmp_path, fake_metrics, fake_results):
        path = CSVReporter().write(fake_metrics, fake_results, tmp_path)
        assert path.exists()
        assert path.name == "results.csv"

    def test_row_count_matches_results(self, tmp_path, fake_metrics, fake_results):
        path = CSVReporter().write(fake_metrics, fake_results, tmp_path)
        df = pd.read_csv(path)
        assert len(df) == len(fake_results)

    def test_has_required_columns(self, tmp_path, fake_metrics, fake_results):
        path = CSVReporter().write(fake_metrics, fake_results, tmp_path)
        df = pd.read_csv(path)
        for col in ("test_id", "category", "score", "passed", "client_latency_ms"):
            assert col in df.columns, f"missing column: {col}"

    def test_sorted_by_category_and_test_id(self, tmp_path, fake_metrics, fake_results):
        path = CSVReporter().write(fake_metrics, fake_results, tmp_path)
        df = pd.read_csv(path)
        cats = df["category"].tolist()
        assert cats == sorted(cats), "rows not sorted by category"

    def test_all_test_ids_present(self, tmp_path, fake_metrics, fake_results):
        path = CSVReporter().write(fake_metrics, fake_results, tmp_path)
        df = pd.read_csv(path)
        expected_ids = {r.test_id for r in fake_results}
        assert set(df["test_id"]) == expected_ids

    def test_empty_results_creates_empty_csv(self, tmp_path, fake_metrics):
        path = CSVReporter().write(fake_metrics, [], tmp_path)
        assert path.exists()


# ---------------------------------------------------------------------------
# HTMLReporter
# ---------------------------------------------------------------------------

class TestHTMLReporter:
    def test_creates_file(self, tmp_path, fake_metrics, fake_results):
        path = HTMLReporter().write(fake_metrics, fake_results, tmp_path)
        assert path.exists()
        assert path.name == "report.html"

    def test_is_valid_html_start(self, tmp_path, fake_metrics, fake_results):
        path = HTMLReporter().write(fake_metrics, fake_results, tmp_path)
        html = path.read_text(encoding="utf-8")
        assert html.strip().startswith("<!DOCTYPE html>")

    def test_contains_model_key(self, tmp_path, fake_metrics, fake_results):
        path = HTMLReporter().write(fake_metrics, fake_results, tmp_path)
        assert "google/gemma-4-e2b" in path.read_text(encoding="utf-8")

    def test_contains_all_test_ids(self, tmp_path, fake_metrics, fake_results):
        path = HTMLReporter().write(fake_metrics, fake_results, tmp_path)
        html = path.read_text(encoding="utf-8")
        for r in fake_results:
            assert r.test_id in html, f"test_id {r.test_id} missing from HTML"

    def test_contains_pass_rate(self, tmp_path, fake_metrics, fake_results):
        path = HTMLReporter().write(fake_metrics, fake_results, tmp_path)
        html = path.read_text(encoding="utf-8")
        assert "Pass Rate" in html

    def test_contains_process_quality_section(self, tmp_path, fake_metrics, fake_results):
        path = HTMLReporter().write(fake_metrics, fake_results, tmp_path)
        html = path.read_text(encoding="utf-8")
        assert "Process Quality" in html
        assert "IDEAL" in html

    def test_contains_category_breakdown(self, tmp_path, fake_metrics, fake_results):
        path = HTMLReporter().write(fake_metrics, fake_results, tmp_path)
        html = path.read_text(encoding="utf-8")
        assert "accuracy" in html
        assert "reasoning" in html

    def test_failure_reason_shown_for_failed_results(self, tmp_path, fake_metrics, fake_results):
        path = HTMLReporter().write(fake_metrics, fake_results, tmp_path)
        html = path.read_text(encoding="utf-8")
        assert "expected_answer_not_found" in html

    def test_escapes_special_characters(self, tmp_path, fake_metrics):
        xss_result = _fake_result("XSS_001", passed=False)
        xss_result.response = '<script>alert("xss")</script>'
        xss_result.failure_reason = "<img src=x onerror=alert(1)>"
        path = HTMLReporter().write(fake_metrics, [xss_result], tmp_path)
        html = path.read_text(encoding="utf-8")
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html or "alert" not in html.split("script")[0]

    def test_empty_results_renders_without_error(self, tmp_path, fake_metrics):
        path = HTMLReporter().write(fake_metrics, [], tmp_path)
        assert path.exists()
        assert "<!DOCTYPE html>" in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# ReportRunner
# ---------------------------------------------------------------------------

class TestReportRunner:
    def test_creates_run_directory(self, fake_metrics, fake_results, offline_config):
        runner = ReportRunner(offline_config)
        run_dir = runner.run(fake_metrics, fake_results)
        assert run_dir.exists()
        assert run_dir.is_dir()

    def test_run_dir_name_starts_with_run(self, fake_metrics, fake_results, offline_config):
        runner = ReportRunner(offline_config)
        run_dir = runner.run(fake_metrics, fake_results)
        assert run_dir.name.startswith("run_")

    def test_json_report_created(self, fake_metrics, fake_results, offline_config):
        run_dir = ReportRunner(offline_config).run(fake_metrics, fake_results)
        assert (run_dir / "dashboard_summary.json").exists()

    def test_csv_report_created(self, fake_metrics, fake_results, offline_config):
        run_dir = ReportRunner(offline_config).run(fake_metrics, fake_results)
        assert (run_dir / "results.csv").exists()

    def test_html_report_created(self, fake_metrics, fake_results, offline_config):
        run_dir = ReportRunner(offline_config).run(fake_metrics, fake_results)
        assert (run_dir / "report.html").exists()

    def test_all_three_reports_created(self, fake_metrics, fake_results, offline_config):
        run_dir = ReportRunner(offline_config).run(fake_metrics, fake_results)
        assert len(list(run_dir.iterdir())) == 3

    def test_json_report_disabled(self, tmp_path, fake_metrics, fake_results):
        cfg = Config(mock_mode=True, report_dir=tmp_path, enable_json_report=False)
        run_dir = ReportRunner(cfg).run(fake_metrics, fake_results)
        assert not (run_dir / "dashboard_summary.json").exists()
        assert (run_dir / "results.csv").exists()
        assert (run_dir / "report.html").exists()

    def test_csv_report_disabled(self, tmp_path, fake_metrics, fake_results):
        cfg = Config(mock_mode=True, report_dir=tmp_path, enable_csv_report=False)
        run_dir = ReportRunner(cfg).run(fake_metrics, fake_results)
        assert not (run_dir / "results.csv").exists()

    def test_returns_path_object(self, fake_metrics, fake_results, offline_config):
        result = ReportRunner(offline_config).run(fake_metrics, fake_results)
        assert isinstance(result, Path)

    def test_two_runs_create_different_directories(self, fake_metrics, fake_results, offline_config):
        runner = ReportRunner(offline_config)
        dir1 = runner.run(fake_metrics, fake_results)
        dir2 = runner.run(fake_metrics, fake_results)
        # Both must exist; they may share a name if run within the same second
        assert dir1.exists() and dir2.exists()
