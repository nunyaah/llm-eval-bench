"""LLM-as-judge evaluator.

Uses a configurable judge LLM (via LiteLLM) to score the quality of an
actual answer against the expected answer, given the original question.
"""

import json
import re

import litellm

from src.config import JUDGE_MODEL as _DEFAULT_JUDGE_MODEL
from src.evaluators.base import BaseEvaluator

_JUDGE_SYSTEM_PROMPT = """\
You are an impartial judge evaluating an AI assistant's answer.
You will receive a QUESTION, an EXPECTED answer, and the ACTUAL answer produced by the assistant.

Score the ACTUAL answer on a scale from 0 to 10 where:
  0  = completely wrong, irrelevant, or harmful
  5  = partially correct or incomplete
  10 = fully correct, complete, and well-expressed

Respond with ONLY a JSON object: {"score": <integer 0-10>, "reasoning": "<brief explanation>"}
Do NOT include any other text outside the JSON object."""

_JUDGE_USER_TEMPLATE = """\
QUESTION:
{input_text}

EXPECTED ANSWER:
{expected}

ACTUAL ANSWER:
{actual}"""


class LLMJudgeEvaluator(BaseEvaluator):
    """Scores answers using a separate LLM as an impartial judge.

    The judge receives the question, expected answer, and actual answer,
    then returns a quality score normalised to [0, 1].

    Parameters
    ----------
    judge_model : str
        LiteLLM model identifier for the judge (default from ``config.JUDGE_MODEL``).
    """

    name = "llm_judge"

    def __init__(self, judge_model: str | None = None):
        self.judge_model = judge_model or _DEFAULT_JUDGE_MODEL

    def score(self, expected: str, actual: str, input_text: str | None = None) -> float:
        if not actual or not actual.strip():
            return 0.0

        user_msg = _JUDGE_USER_TEMPLATE.format(
            input_text=input_text or "(not provided)",
            expected=expected,
            actual=actual,
        )

        try:
            response = litellm.completion(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            content = response.choices[0].message.content or ""
            return self._parse_score(content)
        except Exception:
            return 0.0

    @staticmethod
    def _parse_score(text: str) -> float:
        """Extract a 0-10 score from the judge response and normalise to [0, 1]."""
        # Try JSON parse first
        try:
            data = json.loads(text)
            raw = float(data["score"])
            return max(0.0, min(1.0, raw / 10.0))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

        # Fallback: look for a bare number after "score"
        match = re.search(r'"?score"?\s*[:=]\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if match:
            raw = float(match.group(1))
            return max(0.0, min(1.0, raw / 10.0))

        # Last resort: first number in string
        match = re.search(r'\b(\d+(?:\.\d+)?)\b', text)
        if match:
            raw = float(match.group(1))
            if raw <= 10:
                return max(0.0, min(1.0, raw / 10.0))

        return 0.0
