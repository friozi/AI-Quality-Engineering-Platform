"""
ReportRunner — orchestrates all enabled reporters for a completed QA run.

Creates a timestamped sub-directory under config.report_dir and delegates
to JSONReporter, CSVReporter, and HTMLReporter according to the config flags.

Usage (typically from conftest.py session teardown)::

    runner = ReportRunner(config)
    run_dir = runner.run(collector.aggregate(), collector.get_results())
    logger.info(f"Reports written to {run_dir}")
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.metrics.metrics_calculator import AggregatedMetrics
from src.reporters.csv_reporter import CSVReporter
from src.reporters.html_reporter import HTMLReporter
from src.reporters.json_reporter import JSONReporter
from src.schemas.evaluation_result import EvaluationResult
from src.utils.config import Config


class ReportRunner:
    """
    Runs all enabled reporters against a completed session's data.

    Reporters are selected from config flags:
      enable_json_report → JSONReporter
      enable_csv_report  → CSVReporter
      enable_html_report → HTMLReporter
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._reporters = []
        if config.enable_json_report:
            self._reporters.append(JSONReporter())
        if config.enable_csv_report:
            self._reporters.append(CSVReporter())
        if config.enable_html_report:
            self._reporters.append(HTMLReporter())

    def run(
        self,
        metrics: AggregatedMetrics,
        results: list[EvaluationResult],
    ) -> Path:
        """
        Write all reports to a new timestamped directory.

        Returns the run directory path.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = Path(self._config.report_dir) / f"run_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        for reporter in self._reporters:
            try:
                path = reporter.write(metrics, results, run_dir)
                written.append(path)
                logger.info(f"[reporter] {reporter.__class__.__name__} → {path}")
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[reporter] {reporter.__class__.__name__} failed: {exc}")

        logger.info(
            f"[reporter] run complete — {len(written)}/{len(self._reporters)} reports "
            f"written to {run_dir}"
        )
        return run_dir
