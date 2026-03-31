from src.evaluators.exact_match import ExactMatchEvaluator
from src.evaluators.semantic_similarity import SemanticSimilarityEvaluator


class TestExactMatchEvaluator:
    def test_exact_match(self):
        ev = ExactMatchEvaluator()
        assert ev.score("hello", "hello") == 1.0

    def test_no_match(self):
        ev = ExactMatchEvaluator()
        assert ev.score("hello", "world") == 0.0

    def test_case_insensitive_default(self):
        ev = ExactMatchEvaluator()
        assert ev.score("Hello", "hello") == 1.0

    def test_case_sensitive(self):
        ev = ExactMatchEvaluator(case_sensitive=True)
        assert ev.score("Hello", "hello") == 0.0

    def test_strip_whitespace(self):
        ev = ExactMatchEvaluator()
        assert ev.score("  hello  ", "hello") == 1.0

    def test_no_strip_whitespace(self):
        ev = ExactMatchEvaluator(strip_whitespace=False)
        assert ev.score("  hello  ", "hello") == 0.0

    def test_name(self):
        ev = ExactMatchEvaluator()
        assert ev.name == "exact_match"


class TestSemanticSimilarityEvaluator:
    def test_identical_strings(self):
        ev = SemanticSimilarityEvaluator()
        score = ev.score("the cat sat on the mat", "the cat sat on the mat")
        assert score > 0.99

    def test_similar_strings(self):
        ev = SemanticSimilarityEvaluator()
        score = ev.score(
            "the cat sat on the mat",
            "a cat was sitting on a mat",
        )
        assert 0.0 < score < 1.0

    def test_unrelated_strings(self):
        ev = SemanticSimilarityEvaluator()
        score = ev.score(
            "quantum physics experiments",
            "baking chocolate cake recipe",
        )
        assert score < 0.5

    def test_empty_string(self):
        ev = SemanticSimilarityEvaluator()
        assert ev.score("hello", "") == 0.0
        assert ev.score("", "hello") == 0.0

    def test_name(self):
        ev = SemanticSimilarityEvaluator()
        assert ev.name == "semantic_similarity"
