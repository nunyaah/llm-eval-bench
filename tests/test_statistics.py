import pytest

from src.statistics.bootstrap import bootstrap_confidence_interval
from src.statistics.comparison import paired_bootstrap_test


class TestBootstrapConfidenceInterval:
    def test_basic(self):
        scores = [0.8, 0.9, 0.7, 0.85, 0.95]
        result = bootstrap_confidence_interval(scores, seed=42)
        assert "mean" in result
        assert "lower" in result
        assert "upper" in result
        assert result["lower"] <= result["mean"] <= result["upper"]
        assert result["confidence_level"] == 0.95

    def test_all_same_scores(self):
        scores = [1.0] * 10
        result = bootstrap_confidence_interval(scores, seed=42)
        assert result["mean"] == 1.0
        assert result["lower"] == 1.0
        assert result["upper"] == 1.0

    def test_custom_confidence_level(self):
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = bootstrap_confidence_interval(scores, confidence_level=0.90, seed=42)
        assert result["confidence_level"] == 0.90

    def test_ci_width_increases_with_variance(self):
        low_var = [0.80, 0.81, 0.79, 0.80, 0.80]
        high_var = [0.2, 0.5, 0.8, 1.0, 0.3]
        r1 = bootstrap_confidence_interval(low_var, seed=42)
        r2 = bootstrap_confidence_interval(high_var, seed=42)
        width1 = r1["upper"] - r1["lower"]
        width2 = r2["upper"] - r2["lower"]
        assert width2 > width1


class TestPairedBootstrapTest:
    def test_identical_scores_not_significant(self):
        scores = [0.8, 0.9, 0.7, 0.85, 0.95]
        result = paired_bootstrap_test(scores, scores, seed=42)
        assert result["is_significant"] is False
        assert result["mean_diff"] == 0.0

    def test_clearly_different_scores(self):
        scores_a = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        scores_b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = paired_bootstrap_test(scores_a, scores_b, seed=42)
        assert result["is_significant"] is True
        assert result["mean_a"] == 1.0
        assert result["mean_b"] == 0.0

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            paired_bootstrap_test([1.0], [1.0, 0.5])

    def test_interpretation_a_better(self):
        scores_a = [1.0] * 20
        scores_b = [0.0] * 20
        result = paired_bootstrap_test(scores_a, scores_b, seed=42)
        assert "Model A" in result["interpretation"]
        assert "better" in result["interpretation"]

    def test_interpretation_b_better(self):
        scores_a = [0.0] * 20
        scores_b = [1.0] * 20
        result = paired_bootstrap_test(scores_a, scores_b, seed=42)
        assert "Model B" in result["interpretation"]
        assert "better" in result["interpretation"]
