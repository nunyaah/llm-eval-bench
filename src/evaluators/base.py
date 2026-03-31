from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    """Base class for all evaluators."""

    name: str = "base"

    @abstractmethod
    def score(self, expected: str, actual: str) -> float:
        """Score the actual output against the expected output.

        Returns a float between 0.0 and 1.0.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
