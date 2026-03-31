"""Evaluate and compare two Ollama-hosted open source LLMs on the sample QA dataset.

Uses LiteLLM's Ollama integration — no API keys required.
Requires Ollama to be running: https://ollama.com
"""

import time
import litellm
from dotenv import load_dotenv

load_dotenv()

from src.datasets.loader import load_dataset
from src.evaluators.exact_match import ExactMatchEvaluator
from src.evaluators.semantic_similarity import SemanticSimilarityEvaluator
from src.statistics.bootstrap import bootstrap_confidence_interval
from src.statistics.comparison import paired_bootstrap_test
from src.tracking.database import Database
from src.tracking.tracker import CostLatencyTracker

# Ollama models — prefixed with "ollama/" for LiteLLM routing
MODELS = [
    "ollama/llama3.2:1b",
    "ollama/llama3.2:3b",
]
DATASET = "data/sample_qa.json"
EVALUATORS = [ExactMatchEvaluator(), SemanticSimilarityEvaluator()]

# Instruct models to respond with just the answer — improves exact match scoring
SYSTEM_PROMPT = (
    "You are a concise factual assistant. "
    "Answer each question with ONLY the answer itself — no explanation, no punctuation, no extra words. "
    "For example: if asked 'What is the capital of France?' reply only 'Paris'."
)


def call_model(model: str, prompt: str) -> dict:
    start = time.perf_counter()
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=64,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    output = response.choices[0].message.content or ""
    usage = response.usage
    tokens_used = ((usage.prompt_tokens or 0) + (usage.completion_tokens or 0)) if usage else 0
    cost = 0.0
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        pass
    return {"output": output, "latency_ms": latency_ms, "tokens_used": tokens_used, "cost": cost}


def main():
    data = load_dataset(DATASET)
    db = Database()
    tracker = CostLatencyTracker()

    run_id = db.create_run(
        dataset_path=DATASET,
        models=MODELS,
        evaluators=[ev.name for ev in EVALUATORS],
        name="llama3.2_1b_vs_3b",
    )

    model_scores: dict[str, dict[str, list[float]]] = {
        m: {ev.name: [] for ev in EVALUATORS} for m in MODELS
    }

    print(f"\nRun #{run_id}: Evaluating {len(data)} samples across {len(MODELS)} models...\n")

    for i, item in enumerate(data, 1):
        input_text = item["input"]
        expected = item["expected_output"]
        print(f"[{i}/{len(data)}] {input_text}")

        for model in MODELS:
            try:
                r = call_model(model, input_text)
                actual = r["output"].strip()
                latency_ms = r["latency_ms"]
                tokens_used = r["tokens_used"]
                cost = r["cost"]
            except Exception as e:
                print(f"  !! {model} error: {e}")
                actual = None
                latency_ms, tokens_used, cost = 0.0, 0, 0.0

            scores = {}
            for ev in EVALUATORS:
                s = ev.score(expected, actual or "") if actual is not None else 0.0
                scores[ev.name] = s
                model_scores[model][ev.name].append(s)

            tracker.record(model, latency_ms, tokens_used, cost)
            db.insert_result(
                run_id=run_id,
                model=model,
                input_text=input_text,
                expected_output=expected,
                actual_output=actual,
                scores=scores,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                cost=cost,
            )

            print(f"  {model:<35} actual={actual!r:<30} em={scores.get('exact_match', 0):.1f}  sim={scores.get('semantic_similarity', 0):.2f}")

    db.complete_run(run_id)

    # ---- Summary ----
    print(f"\n{'='*65}")
    print("RESULTS SUMMARY")
    print(f"{'='*65}\n")

    for model in MODELS:
        stats = model_scores[model]
        em = bootstrap_confidence_interval(stats["exact_match"], seed=42)
        sim = bootstrap_confidence_interval(stats["semantic_similarity"], seed=42)
        tr = tracker.summary(model)
        print(f"  {model}")
        print(f"    Exact Match:          {em['mean']*100:5.1f}%  (95% CI: {em['lower']*100:.1f}–{em['upper']*100:.1f}%)")
        print(f"    Semantic Similarity:  {sim['mean']*100:5.1f}%  (95% CI: {sim['lower']*100:.1f}–{sim['upper']*100:.1f}%)")
        print(f"    Avg latency:          {tr['avg_latency_ms']:.0f} ms")
        print(f"    Total cost:           ${tr['total_cost']:.5f}")
        print()

    # ---- Statistical comparison ----
    print(f"{'='*65}")
    print("STATISTICAL COMPARISON  (primary metric: Exact Match)")
    print(f"{'='*65}\n")

    scores_a = model_scores[MODELS[0]]["exact_match"]
    scores_b = model_scores[MODELS[1]]["exact_match"]
    comp = paired_bootstrap_test(scores_a, scores_b, seed=42)

    print(f"  {MODELS[0]}: {comp['mean_a']*100:.1f}%")
    print(f"  {MODELS[1]}: {comp['mean_b']*100:.1f}%")
    print(f"  Difference (A−B): {comp['mean_diff']*100:+.1f}%")
    print(f"  p-value:          {comp['p_value']:.4f}")
    print(f"  Significant:      {'Yes' if comp['is_significant'] else 'No'}")
    print(f"\n  → {comp['interpretation']}")
    print(f"\nResults stored in DB as run #{run_id}")


if __name__ == "__main__":
    main()
