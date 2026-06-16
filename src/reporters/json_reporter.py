"""
JSONReporter — serialises AggregatedMetrics to dashboard_summary.json.

Output is pretty-printed JSON suitable for CI artefact storage, dashboard
ingestion, or diff-based regression tracking across runs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.metrics.metrics_calculator import AggregatedMetrics
from src.reporters.base_reporter import BaseReporter
from src.schemas.evaluation_result import EvaluationResult


class JSONReporter(BaseReporter):
    FILENAME = "dashboard_summary.json"

    def write(
        self,
        metrics: AggregatedMetrics,
        results: list[EvaluationResult],
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / self.FILENAME

        data = metrics.model_dump(mode="json")
        data["generated_at"] = datetime.now(timezone.utc).isoformat()
        data["results_count"] = len(results)

        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path
