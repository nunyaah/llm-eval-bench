# llm-eval-bench

**Statistically rigorous evaluation framework for LLMs**

Most LLM evaluations report a single average on a small test set and declare a winner. That conclusion may be pure noise.

`llm-eval-bench` is a production-oriented evaluation harness that applies **proper statistical methods**bootstrap confidence intervals, paired hypothesis testing, and normalized scoringto measure model performance reliably and answer the question: *is one model actually better, or is the difference chance?*

---

## Why this exists

Typical evaluation:

* Model A: 82%
* Model B: 79%
   "A is better"

Reality: that 3-point gap may be **noise**, not a real improvement.

This project answers:

* Is one model actually better, with statistical confidence?
* How wide are the confidence intervals?
* What is the cost-vs-performance tradeoff?
* Did a new model version regress?

---

## What is implemented

### Evaluators (`src/evaluators/`)

| Evaluator | Description |
|---|---|
| `ExactMatchEvaluator` | Normalized exact match  strips whitespace, lowercases, removes punctuation, normalizes numeric forms (`4.0`  `4`). Optional article removal (`the`/`a`/`an`) for QA mode. |
| `SemanticSimilarityEvaluator` | TF-IDF cosine similarity. |

The normalization pipeline lives in `src/evaluators/normalization.py` and exposes:
- `normalize_text(text)`  strip, lowercase, collapse whitespace/newlines, remove punctuation
- `normalize_numeric(text)`  `"4.0"`  `"4"`, `"1,000"`  `"1000"`
- `normalize_answer(text, remove_articles=False)`  full pipeline used by exact match

### Statistical analysis (`src/statistics/`)

* **Bootstrap confidence intervals** (`bootstrap_confidence_interval`)  2 000 resamples by default; returns `mean`, `lower`, `upper`, `n_samples`, `ci_method`, and a `warning` flag when n < 30.
* **Paired bootstrap test** (`paired_bootstrap_test`)  tests whether two models differ significantly; returns `p_value`, `is_significant`, `comparison_method`, `n_samples`, `winner`, and an interpretation string.

### Cost & latency tracking (`src/tracking/`)

* `CostLatencyTracker` records per-request latency, token usage, and cost (via LiteLLM).
* Summaries include `avg_latency_ms`, `p50/p95/p99_latency_ms`, `total_cost`, `total_tokens`.

### Database (`src/tracking/database.py`)

SQLite-backed storage with three tables:

| Table | Key columns |
|---|---|
| `eval_runs` | `id`, `name`, `dataset_path`, `models`, `evaluators`, `primary_metric`, `sample_count`, `status`, `created_at` |
| `model_summaries` | `run_id`, `model_name`, `exact_match`, `semantic_similarity`, `ci_lower`, `ci_upper`, `avg_latency_ms`, `total_cost` |
| `eval_results` | `run_id`, `model`, `input`, `expected_output`, `actual_output`, `normalized_actual`, `scores`, `latency_ms`, `cost` |

The schema auto-migrates existing databases when new columns are added.

### REST API (`src/api/`)

Built with FastAPI. Interactive docs at `/docs`.

| Endpoint | Description |
|---|---|
| `POST /api/run-eval` | Run an evaluation comparing one or more models on a dataset |
| `GET /api/results/{run_id}` | Full results: run metadata, per-model summaries, all per-sample rows |
| `GET /api/compare/{run_id}` | Statistical comparison: scores, CI, p-value, significance, winner |
| `GET /api/runs` | List all past runs |

### Dashboard (`/dashboard`)

A single-page HTML interface served by the API that shows:

* **Run Summary**  run ID, dataset, sample count, primary metric
* **Model Scores**  score, 95% CI, avg latency, total cost, winner badge
* **Statistical Result**  p-value, difference, significance, winner, method details, small-sample warning
* **Per-Sample Failures**  raw output and normalized output side-by-side for every failed sample

---

## Project Structure

```
llm-eval-bench/
 src/
    config.py                  # Env-based config (DATABASE_PATH, API_HOST, API_PORT)
    runner.py                  # Core evaluate() function
    evaluators/
       base.py                # BaseEvaluator ABC
       normalization.py       # normalize_text / normalize_numeric / normalize_answer
       exact_match.py         # Normalized ExactMatchEvaluator
       semantic_similarity.py # TF-IDF cosine similarity
    statistics/
       bootstrap.py           # bootstrap_confidence_interval
       comparison.py          # paired_bootstrap_test
    datasets/
       loader.py              # load_dataset (JSON  list[dict])
    tracking/
       database.py            # SQLite Database class
       tracker.py             # CostLatencyTracker
    api/
       main.py                # FastAPI app
       routes.py              # API route handlers
       schemas.py             # Pydantic request/response models
    dashboard/
        app.py                 # Single-page dashboard (HTML)
 tests/
    test_evaluators.py         # 37 tests: normalization, exact match, semantic similarity
    test_statistics.py         # Bootstrap CI and paired test
    test_tracking.py           # Database and tracker
    test_datasets.py           # Dataset loader
    test_api.py                # API endpoint tests
 examples/
    compare_two_models.py      # Compare two cloud models via runner
    regerssion_test_suite.py   # Detect performance regressions
 data/
    sample_qa.json             # 10-question QA dataset
 run_comparison.py              # End-to-end Ollama demo (llama3.2:1b vs 3b)
 Dockerfile
 docker-compose.yml
 requirements.txt
```

---

## Getting Started

### Prerequisites

* Python 3.11+
* (Optional) [Ollama](https://ollama.com) for the local demo  no API key required

### 1. Clone and install

```bash
git clone https://github.com/your-username/llm-eval-bench.git
cd llm-eval-bench
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

Supported variables:

```
DATABASE_PATH=eval_results.db   # default: project root
API_HOST=0.0.0.0
API_PORT=8000
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

### 3. Run the API

```bash
uvicorn src.api.main:app --reload
```

* API: http://localhost:8000
* Swagger UI: http://localhost:8000/docs
* Dashboard: http://localhost:8000/dashboard

### 4. Run with Docker

```bash
docker compose up --build
```

---

## Usage

### Option A  Python runner

```python
from src.runner import evaluate

result = evaluate(
    models=["gpt-4o-mini", "claude-3-haiku-20240307"],
    dataset="data/sample_qa.json",
    evaluators=["exact_match", "semantic_similarity"],
    run_name="my_comparison",
)

print(result["comparison"]["interpretation"])
#  "Model A is significantly better (p=0.0120)"
```

### Option B  REST API

```bash
# Run an evaluation
curl -X POST http://localhost:8000/api/run-eval \
  -H "Content-Type: application/json" \
  -d '{"models": ["gpt-4o-mini", "claude-3-haiku-20240307"],
       "dataset": "data/sample_qa.json",
       "evaluators": ["exact_match", "semantic_similarity"]}'

# Inspect results
curl http://localhost:8000/api/results/1

# Statistical comparison
curl http://localhost:8000/api/compare/1
```

### Option C  Local Ollama demo (no API key required)

```bash
# Pull models
ollama pull llama3.2:1b
ollama pull llama3.2:3b

# Run comparison
python run_comparison.py
```

Sample output:

```
Primary metric:   exact_match
Samples:          10
CI method:        bootstrap (2000 resamples)
Comparison:       paired bootstrap test

Warning: sample size is small; confidence intervals may be wide.

  ollama/llama3.2:1b
    Exact Match:          30.0%  (95% CI: 10.050.0%)
    Semantic Similarity:  62.5%  (95% CI: 48.076.0%)

  Difference (AB): +5.0%
  p-value:          0.5800
  Significant:      No
```

### Option D  Example scripts

```bash
# Compare cloud models
python -m examples.compare_two_models

# Regression test  fail if candidate drops > 5% vs baseline
python -m examples.regerssion_test_suite
```

---

## Dataset Format

JSON array where each item has `input` and `expected_output`:

```json
[
  { "input": "What is the capital of France?", "expected_output": "Paris" },
  { "input": "What is 2 + 2?",                "expected_output": "4"     }
]
```

---

## Running Tests

```bash
pytest tests/ -v
```

60 tests covering evaluators (including all normalization cases), statistics, tracking, and datasets.

---

## Tech Stack

| Concern | Library |
|---|---|
| LLM access | [LiteLLM](https://github.com/BerriAI/litellm)  unified interface to 100+ models |
| API | FastAPI + Uvicorn |
| Statistics | NumPy |
| Similarity | scikit-learn (TF-IDF cosine) |
| Storage | SQLite (built-in, zero infrastructure) |
| Config | python-dotenv |
| Testing | pytest + pytest-asyncio |
| Deployment | Docker + Docker Compose |

---

## What makes this different

* **Normalized scoring**  exact match is robust to whitespace, punctuation, numeric formatting, and case  so `"4.0"` matches `"4"` and `" Pacific "` matches `"Pacific"`
* **Statistical validity**  results come with confidence intervals and p-values, not just averages
* **Inspectable storage**  every sample result, normalized output, and model summary is persisted and queryable via API
* **Production structure**  proper module layout, typed schemas, migration-safe DB, Docker support

---

## Not yet implemented

The following are intentionally deferred:

* LLM-as-judge scoring
* Hallucination / faithfulness detection
* Calibration analysis
* Sample size estimation
* Stratified datasets (by category / difficulty)
* Auth / multi-user support

---

## Contributing

Contributions welcome. Open an issue to discuss major changes before submitting a PR. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT  see [LICENSE](LICENSE).
