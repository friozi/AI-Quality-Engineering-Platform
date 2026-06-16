# Enterprise AI QA Framework

Production-grade evaluation framework for local LLMs. Runs a full test suite against a running model and auto-generates JSON, CSV, and HTML reports at the end of every session.

---

## Requirements

- Python 3.9+
- A local LLM server running and accessible via HTTP (e.g. LM Studio, Ollama, llama.cpp)

---

## Installation

**1. Clone and enter the project:**

```bash
git clone <repo-url>
cd llm-qa-framework
```

**2. Create and activate a virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Configure the environment:**

```bash
cp .env.example .env
```

Open `.env` and set `BASE_URL` to the address of your local LLM server:

```env
BASE_URL=http://localhost:1234
DEFAULT_MODEL_KEY=google/gemma-4-e2b
MOCK_MODE=false
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `httpx` | ≥ 0.27 | HTTP client for LLM API calls |
| `pydantic` | ≥ 2.7 | Schema validation and data models |
| `pydantic-settings` | ≥ 2.3 | Config loading from `.env` |
| `pytest` | ≥ 8.2 | Test runner |
| `pytest-html` | ≥ 4.1 | Optional pytest HTML output |
| `pytest-xdist` | ≥ 3.5 | Parallel test execution |
| `loguru` | ≥ 0.7 | Structured logging |
| `python-dotenv` | ≥ 1.0 | `.env` file loading |
| `jinja2` | ≥ 3.1 | HTML report templating |
| `pandas` | ≥ 2.2 | CSV report generation |
| `numpy` | ≥ 1.26 | Latency percentiles and score aggregation |
| `rapidfuzz` | ≥ 3.9 | Fuzzy string matching for accuracy evaluation |

---

## Running the tests

### Run everything (all evaluators, all datasets)

```bash
pytest
```

Reports are auto-generated at the end under `reports/run_<timestamp>/`.

---

### Run only specific test categories

```bash
pytest -m accuracy        # factual correctness tests
pytest -m reasoning       # chain-of-thought and logic tests
pytest -m safety          # prompt injection + jailbreak tests
pytest -m prompt_injection
pytest -m jailbreak
pytest -m latency
```

---

### Run a specific test file

```bash
pytest tests/test_accuracy.py -v
pytest tests/test_reasoning.py -v
pytest tests/test_safety.py -v
pytest tests/test_phase3_integration.py -v
pytest tests/test_phase5_reporters.py -v
```

---

### Run offline (no LLM needed)

```bash
MOCK_MODE=true pytest tests/test_phase5_reporters.py -v
```

Or set `MOCK_MODE=true` in `.env` before running.

---

### Run a single test by name

```bash
pytest -k "GEO_001"                     # run one test case by its ID
pytest -k "geography or history"        # run by keyword
pytest -k "not safety"                  # exclude a category
```

---

### Run in parallel (faster on large datasets)

```bash
pytest -n 2      # 2 workers — matches MAX_PARALLEL_WORKERS default
pytest -n auto   # use all available CPU cores
```

---

## Test data

Test cases are JSON files in `test_data/`. Adding new cases requires **no Python changes** — just append entries to the relevant file.

```
test_data/
  accuracy/
    geography.json        # capital cities (5 cases)
    math.json             # arithmetic and word problems (5 cases)
    history.json          # historical facts (3 cases)
  reasoning/
    logic.json            # syllogism reasoning (3 cases)
    chain_of_thought.json # multi-step problems (3 cases)
  safety/
    prompt_injection.json # injected command resistance (3 cases)
    jailbreak.json        # roleplay/persona trick resistance (3 cases)
```

Each test case entry follows this structure:

```json
{
  "test_id": "GEO_001",
  "category": "accuracy",
  "prompt": "What is the capital of France?",
  "expected_answer": "Paris",
  "evaluation_strategy": "composite",
  "risk_level": "medium",
  "pass_threshold": 0.75
}
```

---

## Reports

After every `pytest` session, three files are written to `reports/run_<timestamp>/`:

| File | Contents |
|---|---|
| `dashboard_summary.json` | Aggregated metrics: pass rates, latency percentiles, quality score, per-category breakdown |
| `results.csv` | One row per test case — scores, latencies, pass/fail, failure reasons |
| `report.html` | Full visual report — summary cards, category table, per-result detail |

To disable a report type, set in `.env`:

```env
ENABLE_HTML_REPORT=false
ENABLE_JSON_REPORT=false
ENABLE_CSV_REPORT=false
```

---

## Project structure

```
llm-qa-framework/
  src/
    clients/          # HTTP client for local LLM API
    evaluators/       # 7 evaluators: accuracy, relevance, hallucination,
                      #   prompt_injection, jailbreak, latency, reasoning
    metrics/          # MetricsCollector and AggregatedMetrics
    pipeline/         # EvaluationPipeline (orchestrates evaluator + collector)
    reporters/        # JSONReporter, CSVReporter, HTMLReporter, ReportRunner
    schemas/          # Pydantic models: TestCase, LLMResponse, EvaluationResult
    utils/            # Config, DatasetLoader, logger, text utilities
  tests/
    conftest.py                   # Session fixtures + auto-report teardown
    test_accuracy.py              # Accuracy evaluator tests
    test_reasoning.py             # Reasoning evaluator tests
    test_safety.py                # Safety evaluator tests
    test_phase3_integration.py    # Integration tests (pipeline + metrics)
    test_phase5_reporters.py      # Reporter unit tests (offline)
  test_data/                      # JSON test case datasets
  reports/                        # Generated reports (git-ignored)
  .env.example                    # Configuration template
  requirements.txt                # V1 dependencies
  requirements-v2-optional.txt    # V2 dependencies (embeddings, LLM-judge)
  pytest.ini                      # Pytest configuration and marker definitions
```

---

## V2 dependencies (optional, not required for V1)

`requirements-v2-optional.txt` lists packages for future evaluators — embedding-based scoring, toxicity classification, and LLM-as-Judge. Do not install these for V1.

```bash
# Only when ready for V2
pip install sentence-transformers detoxify anthropic
```
