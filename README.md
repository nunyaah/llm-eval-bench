# llm-eval-bench

**Statistically rigorous evaluation framework for LLMs**

Most LLM evaluations rely on simple averages and small test sets. This leads to misleading conclusions and unstable results.

`llm-eval-bench` is a production-oriented evaluation harness that applies **proper statistical methods**—bootstrap, hypothesis testing, confidence intervals, and calibration—to measure model performance reliably.

---

## Why this exists

Typical evaluation:

* Model A: 82%
* Model B: 79%
  → “A is better”

Reality:

* That difference may be **noise**, not a real improvement.

This project answers:

* Is one model actually better?
* How confident are we?
* Is the improvement meaningful?
* What is the cost vs performance tradeoff?

---

## Core Features

### Evaluation Methods

* Exact match / regex scoring
* Semantic similarity (embeddings-based)
* LLM-as-a-judge scoring
* Faithfulness detection (RAG grounding)
* Hallucination detection

### Statistical Analysis (key differentiator)

* Bootstrap confidence intervals
* Paired model comparison
* Hypothesis testing (significance)
* Effect size (Cohen’s d)
* Calibration analysis (confidence vs accuracy)
* Sample size estimation

### Dataset Management

* Versioned evaluation datasets
* Golden test sets
* Regression testing (detect performance drops)
* Stratification by category/difficulty

### Cost & Latency Tracking

* Token usage
* Cost per run / per sample
* Latency (p50, p95, p99)
* Cost–quality tradeoff analysis

### API & Dashboard

* FastAPI endpoints to run and compare evaluations
* Simple dashboard for results and comparisons

---

## Example Use Case

Compare two models on a QA dataset:

```python
run = evaluate(
    models=["gpt-4", "claude-3"],
    dataset="qa_dataset.json",
    evaluators=["exact_match", "semantic_similarity"]
)

compare(run)
```

Output:

* Accuracy with confidence intervals
* Statistical significance (p-value)
* Effect size
* Cost per correct answer

---

## Tech Stack

* Python
* FastAPI
* LiteLLM (multi-provider model access)
* NumPy / SciPy (statistics)
* scikit-learn (similarity)
* SQLite (storage)
* Docker (deployment)
* pytest (testing)

---

## Project Structure

```
llm-eval-bench/
├── src/
│   ├── evaluators/
│   ├── statistics/
│   ├── datasets/
│   ├── tracking/
│   ├── api/
│   └── dashboard/
├── tests/
├── examples/
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/llm-eval-bench.git
cd llm-eval-bench
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

```bash
cp .env.example .env
```

Add your API keys.

### 4. Run the API

```bash
uvicorn src.api.main:app --reload
```

---

## Roadmap

### Phase 1

* Evaluation runner
* Exact match + similarity evaluators
* Basic metrics + storage

### Phase 2

* Bootstrap confidence intervals
* Paired comparisons
* Cost & latency tracking

### Phase 3

* Hypothesis testing
* Effect size
* Calibration analysis

### Phase 4

* LLM-as-judge
* Faithfulness / hallucination detection
* Regression testing

### Phase 5

* Dashboard + visualization
* Documentation + examples

---

## What makes this different

* Focus on **statistical validity**, not just metrics
* Designed as a **production system**, not a notebook
* Combines **ML evaluation + MLOps + software engineering**
* Targets real-world needs: cost, latency, reliability

---

## Contributing

Contributions are welcome. Open an issue to discuss major changes before submitting a PR.

---

## License

MIT License

---

## Citation

If you use this project in research or production, please cite it (CITATION.cff coming soon).
