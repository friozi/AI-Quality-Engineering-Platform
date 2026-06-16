"""
DatasetLoader — load, validate, and serve JSON test datasets.

Supported file format (two variants accepted):

    Variant A — bare list:
        [{"test_id": "ACC_001", ...}, ...]

    Variant B — wrapped object (allows dataset-level metadata):
        {
            "name": "Accuracy Dataset",
            "version": "1.0.0",
            "description": "...",
            "test_cases": [{"test_id": "ACC_001", ...}, ...]
        }

Validation is two-pass:
    Pass 1 — Pydantic:  field types, required fields, value ranges.
    Pass 2 — Semantic:  duplicate test_ids, cross-field consistency.

On any error the loader raises DatasetValidationError with a full list of
all problems found (not just the first), so the author can fix everything
at once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from src.schemas.test_case import TestCase


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DatasetValidationError(Exception):
    """
    Raised when a dataset file fails Pydantic or semantic validation.

    Attributes:
        errors : full list of error messages (one per invalid test case or issue)
        path   : the file that caused the error (if known)
    """

    def __init__(self, errors: list[str], path: Optional[Path] = None) -> None:
        self.errors = errors
        self.path = path
        header = f"Dataset validation failed{f' ({path})' if path else ''}"
        detail = "\n  ".join(errors)
        super().__init__(f"{header} — {len(errors)} error(s):\n  {detail}")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class DatasetLoader:
    """
    Stateless utility for loading QA test datasets from JSON files.

    All methods are static — no instance state is needed.
    """

    @staticmethod
    def load(path: Path | str) -> list[TestCase]:
        """
        Load and validate all test cases from a single JSON file.

        Raises:
            FileNotFoundError      : file does not exist.
            DatasetValidationError : any test case fails Pydantic or semantic checks.
            json.JSONDecodeError   : file is not valid JSON.
        """
        file_path = Path(path)
        raw_text = file_path.read_text(encoding="utf-8")
        raw: Any = json.loads(raw_text)

        items = DatasetLoader._extract_items(raw, file_path)
        test_cases = DatasetLoader._parse_items(items, file_path)
        DatasetLoader._validate_semantics(test_cases, file_path)

        return test_cases

    @staticmethod
    def load_by_category(path: Path | str, category: str) -> list[TestCase]:
        """Load a dataset and return only test cases matching *category*."""
        return [tc for tc in DatasetLoader.load(path) if tc.category == category]

    @staticmethod
    def load_by_tag(path: Path | str, tag: str) -> list[TestCase]:
        """Return test cases that include *tag* in their tags list."""
        return [tc for tc in DatasetLoader.load(path) if tag in tc.tags]

    @staticmethod
    def load_directory(directory: Path | str) -> dict[str, list[TestCase]]:
        """
        Load every ``*.json`` file in *directory*.

        Returns a dict keyed by filename stem (e.g. "accuracy_dataset").
        Files that fail validation raise DatasetValidationError immediately.
        """
        dir_path = Path(directory)
        result: dict[str, list[TestCase]] = {}
        for json_file in sorted(dir_path.glob("*.json")):
            result[json_file.stem] = DatasetLoader.load(json_file)
        return result

    @staticmethod
    def validate(test_cases: list[TestCase]) -> list[str]:
        """
        Run semantic validation on an already-parsed list of TestCases.

        Returns a list of error messages (empty list = valid).
        Does not raise — callers decide whether to act on the errors.
        """
        errors: list[str] = []
        seen_ids: set[str] = set()

        for tc in test_cases:
            if tc.test_id in seen_ids:
                errors.append(f"Duplicate test_id: {tc.test_id!r}")
            seen_ids.add(tc.test_id)

            # expected_answer required for composite/exact/fuzzy/math strategies
            from src.schemas.test_case import EvalStrategy  # local import
            needs_answer = tc.evaluation_strategy in (
                EvalStrategy.EXACT,
                EvalStrategy.FUZZY,
                EvalStrategy.COMPOSITE,
                EvalStrategy.MATH,
            )
            if needs_answer and tc.expected_answer is None and not tc.required_concepts:
                errors.append(
                    f"{tc.test_id}: evaluation_strategy={tc.evaluation_strategy.value!r} "
                    f"requires expected_answer or required_concepts"
                )

        return errors

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_items(raw: Any, path: Path) -> list[dict[str, Any]]:
        """Extract the list of raw test case dicts from the parsed JSON."""
        if isinstance(raw, list):
            return raw

        if isinstance(raw, dict):
            if "test_cases" in raw:
                items = raw["test_cases"]
                if isinstance(items, list):
                    return items
                raise DatasetValidationError(
                    [f"'test_cases' field must be a list, got {type(items).__name__}"],
                    path=path,
                )
            raise DatasetValidationError(
                ["Dataset object must contain a 'test_cases' key"],
                path=path,
            )

        raise DatasetValidationError(
            [f"Dataset must be a JSON list or object, got {type(raw).__name__}"],
            path=path,
        )

    @staticmethod
    def _parse_items(
        items: list[dict[str, Any]],
        path: Path,
    ) -> list[TestCase]:
        """
        Parse each raw dict into a TestCase via Pydantic.

        Collects ALL errors before raising so the dataset author sees
        everything that needs fixing in one pass.
        """
        results: list[TestCase] = []
        errors: list[str] = []

        for idx, item in enumerate(items):
            test_id = item.get("test_id", f"[index {idx}]")
            try:
                results.append(TestCase(**item))
            except ValidationError as exc:
                for e in exc.errors():
                    field = ".".join(str(loc) for loc in e["loc"])
                    errors.append(
                        f"{test_id} — field '{field}': {e['msg']} (got {e.get('input')!r})"
                    )
            except TypeError as exc:
                errors.append(f"{test_id} — unexpected error: {exc}")

        if errors:
            raise DatasetValidationError(errors, path=path)

        return results

    @staticmethod
    def _validate_semantics(test_cases: list[TestCase], path: Path) -> None:
        """Run semantic checks and raise if any are found."""
        errors = DatasetLoader.validate(test_cases)
        if errors:
            raise DatasetValidationError(errors, path=path)
