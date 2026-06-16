"""
HTMLReporter — renders a visual evaluation dashboard using Jinja2.

The template (report.html.j2) is loaded from the templates/ directory
alongside this file.  Output is a single self-contained HTML file with
inline CSS — no CDN or internet connection required.

autoescape=True ensures test prompts and model responses are safely escaped.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.metrics.metrics_calculator import AggregatedMetrics
from src.reporters.base_reporter import BaseReporter
from src.schemas.evaluation_result import EvaluationResult

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_NAME = "report.html.j2"


class HTMLReporter(BaseReporter):
    FILENAME = "report.html"

    def write(
        self,
        metrics: AggregatedMetrics,
        results: list[EvaluationResult],
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / self.FILENAME

        env = Environment(
            loader=FileSystemLoader(str(_TEMPLATE_DIR)),
            autoescape=True,
        )
        template = env.get_template(_TEMPLATE_NAME)

        # Sort results: failures first within each category so they stand out
        sorted_results = sorted(
            [r.to_flat_dict() for r in results],
            key=lambda r: (r.get("category", ""), r.get("passed", True), r.get("test_id", "")),
        )

        html = template.render(
            metrics=metrics.model_dump(mode="json"),
            results=sorted_results,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        )

        path.write_text(html, encoding="utf-8")
        return path
