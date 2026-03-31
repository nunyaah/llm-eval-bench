import numpy as np


def bootstrap_confidence_interval(
    scores: list[float],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> dict:
    """Compute bootstrap confidence interval for a set of scores.

    Returns:
        dict with keys: mean, lower, upper, std, confidence_level
    """
    scores_arr = np.array(scores)
    rng = np.random.default_rng(seed)
    n = len(scores_arr)

    bootstrap_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(scores_arr, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)

    alpha = 1 - confidence_level
    lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    return {
        "mean": float(np.mean(scores_arr)),
        "lower": lower,
        "upper": upper,
        "std": float(np.std(bootstrap_means)),
        "confidence_level": confidence_level,
    }
