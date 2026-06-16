from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations — defined here, imported by other schemas and evaluators
# ---------------------------------------------------------------------------

class EvalStrategy(str, Enum):
    """Controls which comparison method(s) an evaluator applies."""
    EXACT = "exact"           # Normalised exact string match only
    FUZZY = "fuzzy"           # Fuzzy ratio only (rapidfuzz)
    COVERAGE = "coverage"     # Keyword / concept coverage only
    REGEX = "regex"           # Regex pattern matching
    MATH = "math"             # Numeric extraction + tolerance comparison
    STRUCTURAL = "structural" # JSON schema / structural validation (tool calls)
    COMPOSITE = "composite"   # Weighted combination of applicable methods (default)


class EvalTier(int, Enum):
    """
    Evaluation reliability tier — declared by every evaluator.

    RULE_BASED : V1 — deterministic, no model calls.
    EMBEDDING  : V2 — cosine similarity via embedding model (not yet available).
    LLM_JUDGE  : V2 — secondary LLM judges quality (not yet available).
    """
    RULE_BASED = 1
    EMBEDDING = 2   # V2 placeholder
    LLM_JUDGE = 3   # V2 placeholder


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# TestCase schema
# ---------------------------------------------------------------------------

class TestCase(BaseModel):
    """
    Represents a single evaluation test case loaded from a JSON dataset file.

    All evaluators receive a TestCase and an LLMResponse; they must not
    perform API calls or load additional data.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    test_id: str = Field(..., description="Globally unique test identifier, e.g. ACC_001")
    version: str = Field(default="1.0.0", description="Dataset schema version")
    category: str = Field(..., description="Evaluation category, e.g. 'accuracy', 'reasoning'")

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------
    prompt: str = Field(..., description="The user input sent to the LLM")
    system_prompt: Optional[str] = Field(
        default=None,
        description=(
            "Optional system context.  Prepended to 'input' inline "
            "as 'System: {system}\\n\\nUser: {prompt}' because the local API "
            "does not have a separate system field."
        ),
    )

    # Reserved for V2 RAG tests — ignored by all V1 evaluators.
    context: Optional[str] = Field(default=None, description="Document context (V2 RAG use)")

    # ------------------------------------------------------------------
    # Answer validation
    # ------------------------------------------------------------------
    expected_answer: Optional[str] = Field(
        default=None,
        description="Canonical ground-truth answer for comparison",
    )
    valid_alternatives: list[str] = Field(
        default_factory=list,
        description="Acceptable answer phrasings (exact/fuzzy match accepted)",
    )
    required_concepts: list[str] = Field(
        default_factory=list,
        description="Terms or phrases that MUST appear in the response",
    )
    forbidden_content: list[str] = Field(
        default_factory=list,
        description="Strings that MUST NOT appear in the response (hard block)",
    )

    # ------------------------------------------------------------------
    # Reasoning validation (populated for category='reasoning')
    # ------------------------------------------------------------------
    required_reasoning_steps: list[str] = Field(
        default_factory=list,
        description="Phrases that must appear in the reasoning block",
    )
    reasoning_must_contain: list[str] = Field(
        default_factory=list,
        description="Additional concepts required in the reasoning block",
    )
    reasoning_must_not_contain: list[str] = Field(
        default_factory=list,
        description="Strings that must NOT appear in the reasoning block",
    )
    evaluate_reasoning_quality: bool = Field(
        default=False,
        description="If True, ReasoningEvaluator scores the chain quality separately",
    )
    evaluate_consistency: bool = Field(
        default=True,
        description=(
            "If True and response.has_reasoning, evaluators check that the "
            "reasoning block and final answer are internally consistent"
        ),
    )

    # ------------------------------------------------------------------
    # Evaluation config
    # ------------------------------------------------------------------
    evaluation_strategy: EvalStrategy = Field(
        default=EvalStrategy.COMPOSITE,
        description="Which comparison method(s) to apply",
    )
    pass_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum score required for PASS verdict",
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.MEDIUM,
        description="Business risk if this test fails",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form labels for filtering and reporting",
    )
    timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="Per-test HTTP timeout override",
    )

    # ------------------------------------------------------------------
    # Extension point — evaluator-specific data
    # (tool definitions, expected_tool_calls, etc. stored here in V1)
    # ------------------------------------------------------------------
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("test_id")
    @classmethod
    def test_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("test_id must not be empty")
        return v.strip()

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must not be empty")
        return v

    @field_validator("category")
    @classmethod
    def category_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("category must not be empty")
        return v.strip().lower()

    @model_validator(mode="after")
    def reasoning_fields_require_reasoning_category(self) -> "TestCase":
        # Advisory only — not a hard error; mixed datasets are valid.
        return self

    def build_input(self) -> str:
        """
        Compose the final string sent to the API's 'input' field.
        System prompt is prepended inline when present.
        """
        if self.system_prompt:
            return f"System: {self.system_prompt}\n\nUser: {self.prompt}"
        return self.prompt
