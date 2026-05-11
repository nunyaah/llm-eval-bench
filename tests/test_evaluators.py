from unittest.mock import MagicMock, patch

from src.evaluators.exact_match import ExactMatchEvaluator
from src.evaluators.faithfulness import FaithfulnessEvaluator
from src.evaluators.llm_judge import LLMJudgeEvaluator
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


def _make_litellm_response(content: str):
    """Create a mock litellm completion response."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


class TestLLMJudgeEvaluator:
    def test_name(self):
        ev = LLMJudgeEvaluator()
        assert ev.name == "llm_judge"

    def test_empty_actual_returns_zero(self):
        ev = LLMJudgeEvaluator()
        assert ev.score("expected", "") == 0.0
        assert ev.score("expected", "  ") == 0.0

    @patch("src.evaluators.llm_judge.litellm.completion")
    def test_perfect_score(self, mock_completion):
        mock_completion.return_value = _make_litellm_response(
            '{"score": 10, "reasoning": "Perfect answer"}'
        )
        ev = LLMJudgeEvaluator()
        result = ev.score("Paris", "Paris", input_text="What is the capital of France?")
        assert result == 1.0

    @patch("src.evaluators.llm_judge.litellm.completion")
    def test_partial_score(self, mock_completion):
        mock_completion.return_value = _make_litellm_response(
            '{"score": 7, "reasoning": "Mostly correct"}'
        )
        ev = LLMJudgeEvaluator()
        result = ev.score("Paris", "paris, france", input_text="Capital of France?")
        assert result == 0.7

    @patch("src.evaluators.llm_judge.litellm.completion")
    def test_zero_score(self, mock_completion):
        mock_completion.return_value = _make_litellm_response(
            '{"score": 0, "reasoning": "Completely wrong"}'
        )
        ev = LLMJudgeEvaluator()
        result = ev.score("Paris", "Tokyo")
        assert result == 0.0

    @patch("src.evaluators.llm_judge.litellm.completion")
    def test_api_error_returns_zero(self, mock_completion):
        mock_completion.side_effect = Exception("API error")
        ev = LLMJudgeEvaluator()
        result = ev.score("Paris", "Paris")
        assert result == 0.0

    def test_parse_score_valid_json(self):
        assert LLMJudgeEvaluator._parse_score('{"score": 8, "reasoning": "Good"}') == 0.8

    def test_parse_score_fallback_regex(self):
        assert LLMJudgeEvaluator._parse_score("score: 6") == 0.6

    def test_parse_score_bare_number(self):
        assert LLMJudgeEvaluator._parse_score("7") == 0.7

    def test_parse_score_clamps_to_one(self):
        assert LLMJudgeEvaluator._parse_score('{"score": 15, "reasoning": ""}') == 1.0

    def test_parse_score_unparseable(self):
        assert LLMJudgeEvaluator._parse_score("no score here at all") == 0.0

    @patch("src.evaluators.llm_judge.litellm.completion")
    def test_custom_judge_model(self, mock_completion):
        mock_completion.return_value = _make_litellm_response('{"score": 9, "reasoning": ""}')
        ev = LLMJudgeEvaluator(judge_model="claude-3-haiku-20240307")
        ev.score("a", "b")
        call_kwargs = mock_completion.call_args
        assert call_kwargs[1]["model"] == "claude-3-haiku-20240307"


class TestFaithfulnessEvaluator:
    def test_name(self):
        ev = FaithfulnessEvaluator()
        assert ev.name == "faithfulness"

    def test_empty_actual_returns_zero(self):
        ev = FaithfulnessEvaluator()
        assert ev.score("expected", "") == 0.0
        assert ev.score("expected", "  ") == 0.0

    @patch("src.evaluators.faithfulness.litellm.completion")
    def test_fully_faithful(self, mock_completion):
        mock_completion.return_value = _make_litellm_response(
            '{"score": 10, "hallucinations": [], "reasoning": "Fully faithful"}'
        )
        ev = FaithfulnessEvaluator()
        result = ev.score("Paris", "Paris", input_text="Capital of France?")
        assert result == 1.0

    @patch("src.evaluators.faithfulness.litellm.completion")
    def test_partial_hallucination(self, mock_completion):
        mock_completion.return_value = _make_litellm_response(
            '{"score": 5, "hallucinations": ["Paris is in Germany"], "reasoning": "Mixed"}'
        )
        ev = FaithfulnessEvaluator()
        result = ev.score("Paris", "Paris is in Germany")
        assert result == 0.5

    @patch("src.evaluators.faithfulness.litellm.completion")
    def test_full_hallucination(self, mock_completion):
        mock_completion.return_value = _make_litellm_response(
            '{"score": 0, "hallucinations": ["Everything is wrong"], "reasoning": "Fabricated"}'
        )
        ev = FaithfulnessEvaluator()
        result = ev.score("Paris", "The moon is made of cheese")
        assert result == 0.0

    @patch("src.evaluators.faithfulness.litellm.completion")
    def test_api_error_returns_zero(self, mock_completion):
        mock_completion.side_effect = Exception("API error")
        ev = FaithfulnessEvaluator()
        result = ev.score("Paris", "Paris")
        assert result == 0.0

    def test_parse_score_valid_json(self):
        text = '{"score": 8, "hallucinations": [], "reasoning": "Good"}'
        assert FaithfulnessEvaluator._parse_score(text) == 0.8

    def test_parse_score_fallback_regex(self):
        assert FaithfulnessEvaluator._parse_score("score: 4") == 0.4

    def test_parse_score_unparseable(self):
        assert FaithfulnessEvaluator._parse_score("no score here at all") == 0.0

    @patch("src.evaluators.faithfulness.litellm.completion")
    def test_custom_judge_model(self, mock_completion):
        mock_completion.return_value = _make_litellm_response(
            '{"score": 10, "hallucinations": [], "reasoning": ""}'
        )
        ev = FaithfulnessEvaluator(judge_model="claude-3-haiku-20240307")
        ev.score("a", "b")
        call_kwargs = mock_completion.call_args
        assert call_kwargs[1]["model"] == "claude-3-haiku-20240307"
