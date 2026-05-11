"""Example: Regression testing – detect performance drops between model versions.

Usage:
    python -m examples.regerssion_test_suite

Compares a baseline model against a candidate to ensure the candidate
doesn't regress below the baseline's performance.
"""

from src.config import DATASET, EVALUATORS, MODELS, PRIMARY_METRIC, SYSTEM_PROMPT
from src.runner import evaluate


REGRESSION_THRESHOLD = 0.05  # Max acceptable drop in accuracy


def main():
    baseline_model = MODELS[0]
    candidate_model = MODELS[1]

    result = evaluate(
        models=[baseline_model, candidate_model],
        dataset=DATASET,
        evaluators=EVALUATORS,
        run_name="regression_test",
        system_prompt=SYSTEM_PROMPT,
        primary_metric=PRIMARY_METRIC,
    )

    baseline_stats = result["model_stats"][baseline_model]
    candidate_stats = result["model_stats"][candidate_model]

    baseline_exact = baseline_stats["exact_match"]["mean"]
    candidate_exact = candidate_stats["exact_match"]["mean"]

    baseline_sem = baseline_stats.get("semantic_similarity", {}).get("mean", 0.0)
    candidate_sem = candidate_stats.get("semantic_similarity", {}).get("mean", 0.0)

    print(f"\n{'='*60}")
    print("Regression Test Results")
    print(f"{'='*60}\n")
    print(f"  {'Model':<40} {'Exact Match':>12} {'Semantic Sim':>14}")
    print(f"  {'-'*66}")
    print(f"  Baseline  ({baseline_model})  {baseline_exact*100:>10.1f}%  {baseline_sem*100:>12.1f}%")
    print(f"  Candidate ({candidate_model})  {candidate_exact*100:>10.1f}%  {candidate_sem*100:>12.1f}%")

    diff = candidate_exact - baseline_exact
    print(f"\n  Exact-match difference: {diff*100:+.1f}%")

    if diff < -REGRESSION_THRESHOLD:
        print(f"\n  ❌ REGRESSION DETECTED: Candidate dropped by {abs(diff)*100:.1f}%")
        print(f"     (threshold: {REGRESSION_THRESHOLD*100:.1f}%)")
    else:
        print(f"\n  ✅ No regression detected")

    if result["comparison"]:
        c = result["comparison"]
        print(f"\n  Statistical significance: p={c['p_value']:.4f}")
        print(f"  → {c['interpretation']}")


if __name__ == "__main__":
    main()
