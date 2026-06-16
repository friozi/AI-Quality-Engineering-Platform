"""
Deterministic text utilities — the shared foundation for all Tier 1 evaluators.

No external model calls are made here.  Every function is pure: same input
always produces the same output.  Evaluators must not implement string
comparison logic directly; they delegate to these functions instead.
"""

from __future__ import annotations

import re
import string
from typing import Optional

from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "on", "at", "by", "for", "with",
        "about", "against", "between", "into", "through", "during", "before",
        "after", "above", "below", "from", "up", "down", "out", "off", "over",
        "under", "again", "then", "once", "and", "but", "or", "nor", "so",
        "yet", "both", "either", "neither", "not", "only", "own", "same",
        "than", "too", "very", "just", "because", "if", "while", "i", "me",
        "my", "myself", "we", "our", "you", "your", "he", "she", "it", "they",
        "them", "their", "this", "that", "these", "those", "what", "which",
        "who", "whom", "how", "all", "each", "every", "few", "more", "most",
        "other", "some", "such", "no", "s", "t",
    }
)

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(
    text: str,
    lowercase: bool = True,
    strip_punctuation: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    """Return a cleaned version of *text* suitable for comparison."""
    if lowercase:
        text = text.lower()
    if strip_punctuation:
        text = text.translate(_PUNCT_TABLE)
    if collapse_whitespace:
        text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Exact and fuzzy matching
# ---------------------------------------------------------------------------

def exact_match(a: str, b: str, case_sensitive: bool = False) -> bool:
    """Normalised exact string match."""
    return normalize_text(a, lowercase=not case_sensitive) == normalize_text(
        b, lowercase=not case_sensitive
    )


def fuzzy_ratio(a: str, b: str) -> float:
    """
    Token-set ratio via rapidfuzz (0.0 – 1.0).

    Token-set ratio is preferred over simple ratio because it handles word-
    order differences well: "Paris is the capital" vs "The capital is Paris"
    score close to 1.0 instead of ~0.5.
    """
    return fuzz.token_set_ratio(normalize_text(a), normalize_text(b)) / 100.0


def fuzzy_partial_ratio(a: str, b: str) -> float:
    """
    Partial ratio via rapidfuzz (0.0 – 1.0).

    Useful when the expected answer is likely a substring of a longer response,
    e.g. expected="Paris" inside "The answer is Paris, the capital of France."
    """
    return fuzz.partial_ratio(normalize_text(a), normalize_text(b)) / 100.0


def best_fuzzy_score(candidate: str, references: list[str]) -> float:
    """Return the highest fuzzy_ratio score between *candidate* and any reference."""
    if not references:
        return 0.0
    return max(fuzzy_ratio(candidate, ref) for ref in references)


# ---------------------------------------------------------------------------
# Keyword and concept coverage
# ---------------------------------------------------------------------------

def keyword_coverage(text: str, keywords: list[str]) -> float:
    """
    Return the fraction of *keywords* found (case-insensitive substring match)
    in *text*.  Returns 1.0 if *keywords* is empty (vacuously satisfied).
    """
    if not keywords:
        return 1.0
    normalized = normalize_text(text)
    matched = sum(1 for kw in keywords if normalize_text(kw) in normalized)
    return matched / len(keywords)


def matched_keywords(text: str, keywords: list[str]) -> list[str]:
    """Return the subset of *keywords* found in *text*."""
    normalized = normalize_text(text)
    return [kw for kw in keywords if normalize_text(kw) in normalized]


def missing_keywords(text: str, keywords: list[str]) -> list[str]:
    """Return the subset of *keywords* NOT found in *text*."""
    normalized = normalize_text(text)
    return [kw for kw in keywords if normalize_text(kw) not in normalized]


def contains_any(
    text: str,
    terms: list[str],
    case_sensitive: bool = False,
) -> tuple[bool, list[str]]:
    """
    Check whether *text* contains any string from *terms*.

    Returns ``(found_any, list_of_matched_terms)``.
    """
    check = text if case_sensitive else text.lower()
    found = [t for t in terms if (t if case_sensitive else t.lower()) in check]
    return bool(found), found


# ---------------------------------------------------------------------------
# Tokenisation and set similarity
# ---------------------------------------------------------------------------

def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Split normalised text into tokens, optionally removing stopwords."""
    tokens = normalize_text(text).split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def jaccard_similarity(a: str, b: str, remove_stopwords: bool = True) -> float:
    """
    Token-level Jaccard similarity between *a* and *b* (0.0 – 1.0).

    Measures overlap of meaningful word sets, independent of word order.
    Useful as a fast proxy for semantic overlap when embeddings are unavailable.
    """
    set_a = set(tokenize(a, remove_stopwords))
    set_b = set(tokenize(b, remove_stopwords))
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ---------------------------------------------------------------------------
# Number and answer extraction
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def extract_numbers(text: str) -> list[float]:
    """
    Extract all numeric values from *text*.

    Handles integers, decimals (dot or comma separator), and negative values.
    """
    results: list[float] = []
    for match in _NUMBER_RE.finditer(text):
        try:
            results.append(float(match.group().replace(",", ".")))
        except ValueError:
            pass
    return results


def numbers_match(
    a: str,
    b: str,
    tolerance: float = 0.001,
) -> bool:
    """
    Return True if *a* and *b* each contain at least one number and their
    first extracted numbers are equal within *tolerance*.

    Used by the MathEval strategy in ReasoningEvaluator.
    """
    nums_a = extract_numbers(a)
    nums_b = extract_numbers(b)
    if not nums_a or not nums_b:
        return False
    return abs(nums_a[0] - nums_b[0]) <= tolerance


_ANSWER_LABEL_RE = re.compile(
    r"(?:answer|result|therefore|so|thus|conclusion|final answer)\s*[:\-–]\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)


def extract_final_answer(text: str) -> str:
    """
    Extract the final answer from a response string.

    Strategy (in order of preference):
    1. Labelled answer line: "Answer: X", "Therefore: X", "Result — X", etc.
    2. Last non-empty line of the text.
    """
    match = _ANSWER_LABEL_RE.search(text)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


# ---------------------------------------------------------------------------
# Reasoning chain analysis
# ---------------------------------------------------------------------------

_NUMBERED_STEP_RE = re.compile(
    r"^\s*(?:\d+[.):]|\bstep\s+\d+\b)",
    re.MULTILINE | re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^\s*[-•*]\s", re.MULTILINE)
_CONNECTIVE_RE = re.compile(
    r"\b(therefore|because|since|first|second|third|fourth|fifth|"
    r"next|then|finally|thus|hence|consequently|as\s+a\s+result)\b",
    re.IGNORECASE,
)


def count_reasoning_steps(text: str) -> int:
    """
    Estimate the number of reasoning steps in *text* using heuristics.

    Priority: numbered lists → bullet points → logical connectives.
    Returns 0 if no structure is detected.
    """
    numbered = _NUMBERED_STEP_RE.findall(text)
    if numbered:
        return len(numbered)

    bullets = _BULLET_RE.findall(text)
    if bullets:
        return len(bullets)

    connectives = _CONNECTIVE_RE.findall(text)
    return len(set(m.lower() for m in connectives))


def check_reasoning_answer_consistency(
    reasoning: str,
    final_answer: str,
    key_terms: Optional[list[str]] = None,
) -> tuple[bool, list[str]]:
    """
    Detect internal inconsistency between the reasoning chain and final answer.

    Two checks are applied:

    1. **Key-term drift** (requires *key_terms*): a term that appears in
       *reasoning* but is absent from *final_answer* signals that the model
       reasoned toward one answer but stated another.
       Example — reasoning: "Paris is the capital", answer: "Lyon" → drift on "Paris".

    2. **Number drift**: every number extracted from *final_answer* must also
       appear in *reasoning*.  A number in the answer that was never mentioned
       in the reasoning suggests the model introduced it without derivation.

    Returns ``(is_consistent, list_of_inconsistent_entities)``.
    Limitations: surface-level string matching only — paraphrased
    contradictions are not detected (V2 embedding evaluators will address this).
    """
    inconsistent: list[str] = []
    norm_reasoning = normalize_text(reasoning)
    norm_answer = normalize_text(final_answer)

    # Check 1: key terms present in reasoning that are absent from final_answer.
    if key_terms:
        for term in key_terms:
            norm_term = normalize_text(term)
            in_reasoning = norm_term in norm_reasoning
            in_answer = norm_term in norm_answer
            if in_reasoning and not in_answer:
                inconsistent.append(term)

    # Check 2: numbers stated in final_answer must appear in reasoning.
    answer_numbers = extract_numbers(final_answer)
    reasoning_numbers = set(extract_numbers(reasoning))
    for num in answer_numbers:
        if all(abs(num - rn) >= 0.001 for rn in reasoning_numbers):
            num_str = str(int(num)) if num == int(num) else str(num)
            inconsistent.append(num_str)

    return len(inconsistent) == 0, inconsistent


# ---------------------------------------------------------------------------
# Length and style heuristics
# ---------------------------------------------------------------------------

def word_count(text: str) -> int:
    return len(text.split())


def sentence_count(text: str) -> int:
    sentences = re.split(r"[.!?]+", text)
    return sum(1 for s in sentences if s.strip())


def average_words_per_sentence(text: str) -> float:
    n_sentences = sentence_count(text)
    if n_sentences == 0:
        return 0.0
    return word_count(text) / n_sentences


def verbosity_ratio(response: str, reference: str) -> float:
    """
    Ratio of response word count to reference word count.

    ratio > 3.0 combined with low keyword coverage = hallucination verbosity risk.
    """
    ref_words = word_count(reference)
    if ref_words == 0:
        return 0.0
    return word_count(response) / ref_words


# ---------------------------------------------------------------------------
# Pattern helpers
# ---------------------------------------------------------------------------

def matches_any_pattern(
    text: str,
    patterns: list[str],
    flags: int = re.IGNORECASE,
) -> tuple[bool, str | None]:
    """
    Return (matched, first_matching_pattern_string) for the first pattern that
    matches *text*, or (False, None) if none match.
    """
    for pattern in patterns:
        if re.search(pattern, text, flags):
            return True, pattern
    return False, None
