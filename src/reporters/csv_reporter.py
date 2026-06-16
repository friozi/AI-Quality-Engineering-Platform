"""
CSVReporter — writes all EvaluationResults as a flat CSV table.

Each row is one test case result.  Columns come from EvaluationResult.to_flat_dict()
which flattens sub_scores and tags into semicolon-separated strings.
Rows are sorted by (category, test_id) for stable diffs across runs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.metrics.metrics_calculator import AggregatedMetrics
from src.reporters.base_reporter import BaseReporter
from src.schemas.evaluation_result import EvaluationResult


class CSVReporter(BaseReporter):
    FILENAME = "results.csv"

    def write(
        self,
        metrics: AggregatedMetrics,
        results: list[EvaluationResult],
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / self.FILENAME

        if not results:
            pd.DataFrame().to_csv(path, index=False)
            return path

        rows = [r.to_flat_dict() for r in results]
        df = pd.DataFrame(rows)

        # Stable sort for reproducible diffs
        sort_cols = [c for c in ("category", "test_id") if c in df.columns]
        if sort_cols:
            df.sort_values(sort_cols, inplace=True, ignore_index=True)

        df.to_csv(path, index=False, encoding="utf-8")
        return path
