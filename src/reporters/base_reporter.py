"""Abstract base for all reporters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.metrics.metrics_calculator import AggregatedMetrics
from src.schemas.evaluation_result import EvaluationResult


class BaseReporter(ABC):
    """
    Write a report from AggregatedMetrics + raw EvaluationResults.

    Each concrete subclass writes one file and returns its path.
    output_dir is created if it does not exist.
    """

    @abstractmethod
    def write(
        self,
        metrics: AggregatedMetrics,
        results: list[EvaluationResult],
        output_dir: Path,
    ) -> Path:
        """Write the report and return the path of the generated file."""
