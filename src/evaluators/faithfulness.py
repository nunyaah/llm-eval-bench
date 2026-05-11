"""Hallucination / faithfulness evaluator.

Uses a configurable judge LLM (via LiteLLM) to assess whether the actual
answer is faithful to the expected answer and the question, or whether it
contains hallucinated, fabricated, or unsupported claims.
"""

import json
import re

import litellm

from src.config import JUDGE_MODEL as _DEFAULT_JUDGE_MODEL
from src.evaluators.base import BaseEvaluator

_FAITHFULNESS_SYSTEM_PROMPT = """\
You are a factual-accuracy auditor.  You will receive a QUESTION, an EXPECTED
answer (the ground truth), and the ACTUAL answer produced by an AI assistant.

Your job is to judge **faithfulness**: does the ACTUAL answer stick to verifiable
facts consistent with the EXPECTED answer, or does it introduce hallucinated,
fabricated, or unsupported claims?

Score on a scale from 0 to 10:
  0  = entirely hallucinated / fabricated — no overlap with truth
  5  = mix of correct and hallucinated content
  10 = fully faithful, no hallucinations

Respond with ONLY a JSON object:
{"score": <integer 0-10>, "hallucinations": ["<quoted claim>", ...], "reasoning": "<brief explanation>"}
Do NOT include any other text outside the JSON object.
If there are no hallucinations, return an empty list for "hallucinations"."""

_FAITHFULNESS_USER_TEMPLATE = """\
QUESTION:
{input_text}

EXPECTED ANSWER (ground truth):
{expected}

ACTUAL ANSWER:
{actual}"""


class FaithfulnessEvaluator(BaseEvaluator):
    """Detects hallucinations by checking faithfulness of the actual answer.

    A judge LLM compares the actual answer against the expected ground truth
    and flags any fabricated or unsupported claims.  Returns a score
    normalised to [0, 1] where 1.0 means fully faithful (no hallucinations).

    Parameters
    ----------
    judge_model : str
        LiteLLM model identifier for the judge (default from ``config.JUDGE_MODEL``).
    """

    name = "faithfulness"

    def __init__(self, judge_model: str | None = None):
        self.judge_model = judge_model or _DEFAULT_JUDGE_MODEL

    def score(self, expected: str, actual: str, input_text: str | None = None) -> float:
        if not actual or not actual.strip():
            return 0.0

        user_msg = _FAITHFULNESS_USER_TEMPLATE.format(
            input_text=input_text or "(not provided)",
            expected=expected,
            actual=actual,
        )

        try:
            response = litellm.completion(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": _FAITHFULNESS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            content = response.choices[0].message.content or ""
            return self._parse_score(content)
        except Exception:
            return 0.0

    @staticmethod
    def _parse_score(text: str) -> float:
        """Extract a 0-10 faithfulness score and normalise to [0, 1]."""
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
