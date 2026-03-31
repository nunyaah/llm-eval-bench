from src.evaluators.exact_match import ExactMatchEvaluator
from src.evaluators.normalization import normalize_answer, normalize_numeric, normalize_text
from src.evaluators.semantic_similarity import SemanticSimilarityEvaluator


class TestNormalization:
    # --- normalize_text ---

    def test_strip_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_lowercase(self):
        assert normalize_text("Pacific") == "pacific"

    def test_collapse_spaces(self):
        assert normalize_text("hello   world") == "hello world"

    def test_collapse_newlines(self):
        assert normalize_text("2\n2") == "2 2"

    def test_remove_punctuation(self):
        assert normalize_text("William Shakespeare.") == "william shakespeare"

    def test_remove_comma(self):
        assert normalize_text("hello, world!") == "hello world"

    def test_remove_quotes(self):
        assert normalize_text('"answer"') == "answer"

    def test_mixed(self):
        assert normalize_text("  Pacific Ocean.  ") == "pacific ocean"

    # --- normalize_numeric ---

    def test_float_to_int(self):
        assert normalize_numeric("4.0") == "4"

    def test_trailing_zeros(self):
        assert normalize_numeric("3.50") == "3.5"

    def test_non_numeric_passthrough(self):
        assert normalize_numeric("Paris") == "Paris"

    def test_integer_string(self):
        assert normalize_numeric("42") == "42"

    def test_thousands_separator(self):
        assert normalize_numeric("1,000") == "1000"

    # --- normalize_answer (full pipeline) ---

    def test_leading_whitespace_and_case(self):
        assert normalize_answer(" Pacific ") == "pacific"

    def test_case_match(self):
        assert normalize_answer("Pacific Ocean") == normalize_answer("pacific ocean")

    def test_numeric_normalization(self):
        assert normalize_answer("4.0") == normalize_answer("4")

    def test_punctuation_end(self):
        assert normalize_answer("William Shakespeare.") == "william shakespeare"

    def test_remove_articles(self):
        assert normalize_answer("The Pacific", remove_articles=True) == "pacific"
        assert normalize_answer("A cat", remove_articles=True) == "cat"
        assert normalize_answer("An apple", remove_articles=True) == "apple"

    def test_no_article_removal_by_default(self):
        # articles not removed unless requested
        assert normalize_answer("the answer") == "the answer"

    def test_newline_in_answer(self):
        assert normalize_answer("2\n2") == "2 2"


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

    def test_case_sensitive_without_normalization(self):
        ev = ExactMatchEvaluator(case_sensitive=True, normalize=False)
        assert ev.score("Hello", "hello") == 0.0

    def test_strip_whitespace_without_normalization(self):
        ev = ExactMatchEvaluator(normalize=False)
        assert ev.score("  hello  ", "hello") == 1.0

    def test_no_strip_whitespace_without_normalization(self):
        ev = ExactMatchEvaluator(strip_whitespace=False, normalize=False)
        assert ev.score("  hello  ", "hello") == 0.0

    def test_name(self):
        ev = ExactMatchEvaluator()
        assert ev.name == "exact_match"

    # --- normalized exact match ---

    def test_normalized_whitespace(self):
        ev = ExactMatchEvaluator()
        assert ev.score(" Pacific ", "Pacific") == 1.0

    def test_normalized_punctuation(self):
        ev = ExactMatchEvaluator()
        assert ev.score("William Shakespeare.", "william shakespeare") == 1.0

    def test_normalized_numeric(self):
        ev = ExactMatchEvaluator()
        assert ev.score("4.0", "4") == 1.0

    def test_normalized_newline(self):
        ev = ExactMatchEvaluator()
        # "2\n2" should NOT equal "2" — they are different answers
        assert ev.score("2\n2", "22") == 0.0
        # but "2\n2" normalized to "2 2" still differs from "2"
        assert ev.score("answer\n", "answer") == 1.0

    def test_normalized_output(self):
        ev = ExactMatchEvaluator()
        assert ev.normalized_output(" Pacific ") == "pacific"
        assert ev.normalized_output("4.0") == "4"


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
