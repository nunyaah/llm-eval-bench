"""Example: Regression testing – detect performance drops between model versions.

Usage:
    python -m examples.regerssion_test_suite

Compares a baseline model against a candidate to ensure the candidate
doesn't regress below the baseline's performance.
"""

from src.runner import evaluate


REGRESSION_THRESHOLD = 0.05  # Max acceptable drop in accuracy


def main():
    baseline_model = "gpt-4o-mini"
    candidate_model = "gpt-4o"

    result = evaluate(
        models=[baseline_model, candidate_model],
        dataset="data/sample_qa.json",
        evaluators=["exact_match"],
        run_name="regression_test",
    )

    baseline_stats = result["model_stats"][baseline_model]
    candidate_stats = result["model_stats"][candidate_model]

    baseline_score = baseline_stats["exact_match"]["mean"]
    candidate_score = candidate_stats["exact_match"]["mean"]

    print(f"\n{'='*60}")
    print("Regression Test Results")
    print(f"{'='*60}\n")
    print(f"  Baseline ({baseline_model}):   {baseline_score*100:.1f}%")
    print(f"  Candidate ({candidate_model}): {candidate_score*100:.1f}%")

    diff = candidate_score - baseline_score
    print(f"  Difference: {diff*100:+.1f}%")

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
