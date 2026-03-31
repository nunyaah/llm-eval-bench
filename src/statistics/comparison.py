import numpy as np


def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 10000,
    significance_level: float = 0.05,
    seed: int | None = None,
) -> dict:
    """Paired bootstrap test to compare two models.

    Tests whether model A is significantly different from model B
    by bootstrapping the difference in means.

    Returns:
        dict with keys: mean_a, mean_b, mean_diff, p_value, is_significant, interpretation
    """
    a = np.array(scores_a)
    b = np.array(scores_b)

    if len(a) != len(b):
        raise ValueError("Score arrays must have the same length")

    rng = np.random.default_rng(seed)
    n = len(a)
    observed_diff = float(np.mean(a) - np.mean(b))

    # Bootstrap the difference under the null hypothesis (shift to zero-mean diff)
    diffs = a - b
    centered_diffs = diffs - np.mean(diffs)

    count_extreme = 0
    for _ in range(n_bootstrap):
        sample = rng.choice(centered_diffs, size=n, replace=True)
        boot_diff = np.mean(sample)
        if abs(boot_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = float(count_extreme / n_bootstrap)
    is_significant = p_value < significance_level

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))

    if not is_significant:
        interpretation = "No statistically significant difference between models"
    elif observed_diff > 0:
        interpretation = f"Model A is significantly better (p={p_value:.4f})"
    else:
        interpretation = f"Model B is significantly better (p={p_value:.4f})"

    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": observed_diff,
        "p_value": p_value,
        "is_significant": is_significant,
        "significance_level": significance_level,
        "interpretation": interpretation,
    }
