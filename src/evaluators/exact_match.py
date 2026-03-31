from src.evaluators.base import BaseEvaluator


class ExactMatchEvaluator(BaseEvaluator):
    """Scores 1.0 if the expected and actual outputs match exactly, 0.0 otherwise."""

    name = "exact_match"

    def __init__(self, case_sensitive: bool = False, strip_whitespace: bool = True):
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    def score(self, expected: str, actual: str) -> float:
        e = expected
        a = actual

        if self.strip_whitespace:
            e = e.strip()
            a = a.strip()

        if not self.case_sensitive:
            e = e.lower()
            a = a.lower()

        return 1.0 if e == a else 0.0
