from src.evaluators.base import BaseEvaluator
from src.evaluators.normalization import normalize_answer


class ExactMatchEvaluator(BaseEvaluator):
    """Normalized exact match evaluator.

    By default applies full normalization (case, whitespace, punctuation, numerics)
    so that surface-level formatting differences do not affect scores.
    Set ``normalize=False`` to fall back to the original strip+lowercase behaviour.
    """

    name = "exact_match"

    def __init__(
        self,
        case_sensitive: bool = False,
        strip_whitespace: bool = True,
        normalize: bool = True,
        remove_articles: bool = False,
    ):
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace
        self.normalize = normalize
        self.remove_articles = remove_articles

    def _prepare(self, text: str) -> str:
        """Return the comparison form of *text*."""
        if self.normalize:
            return normalize_answer(text, remove_articles=self.remove_articles)
        e = text
        if self.strip_whitespace:
            e = e.strip()
        if not self.case_sensitive:
            e = e.lower()
        return e

    def score(self, expected: str, actual: str, input_text: str | None = None) -> float:
        return 1.0 if self._prepare(expected) == self._prepare(actual) else 0.0

    def normalized_output(self, text: str) -> str:
        """Return the normalized form of *text* for display / logging."""
        return self._prepare(text)
