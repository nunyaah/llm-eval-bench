"""Example: Compare two models on a QA dataset.

Usage:
    python -m examples.compare_two_models

Requires API keys set in .env (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
or Ollama running locally for ollama/ models.
"""

from src.config import DATASET, EVALUATORS, MODELS, PRIMARY_METRIC, SYSTEM_PROMPT
from src.runner import evaluate


def main():
    result = evaluate(
        models=MODELS,
        dataset=DATASET,
        evaluators=EVALUATORS,
        run_name="example_comparison",
        system_prompt=SYSTEM_PROMPT,
        primary_metric=PRIMARY_METRIC,
    )

    print(f"\n{'='*60}")
    print(f"Evaluation Run #{result['run_id']}")
    print(f"Dataset: {result['dataset']} ({result['num_samples']} samples)")
    print(f"{'='*60}\n")

    for model, stats in result["model_stats"].items():
        print(f"--- {model} ---")
        for ev_name in result["evaluators"]:
            if ev_name in stats:
                s = stats[ev_name]
                print(
                    f"  {ev_name}: {s['mean']*100:.1f}% "
                    f"(±{s['std']*100:.1f}%, "
                    f"CI: {s['lower']*100:.1f}–{s['upper']*100:.1f}%)"
                )
        tracking = stats.get("tracking", {})
        print(f"  Avg latency: {tracking.get('avg_latency_ms', 0):.0f}ms")
        print(f"  Total cost:  ${tracking.get('total_cost', 0):.4f}")
        print()

    if result["comparison"]:
        c = result["comparison"]
        print(f"{'='*60}")
        print("Statistical Comparison")
        print(f"{'='*60}")
        print(f"  {c['model_a']}: {c['mean_a']*100:.1f}%")
        print(f"  {c['model_b']}: {c['mean_b']*100:.1f}%")
        print(f"  p-value: {c['p_value']:.4f}")
        print(f"  → {c['interpretation']}")


if __name__ == "__main__":
    main()
