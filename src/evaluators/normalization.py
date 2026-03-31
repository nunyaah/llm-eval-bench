"""Text normalization utilities for answer comparison.

Provides normalization functions that make exact match scoring robust to
surface-level formatting differences (whitespace, case, punctuation, numerics).
"""

import re


def normalize_text(text: str) -> str:
    """Normalize text for answer comparison.

    Applies: strip, lowercase, collapse whitespace/newlines, remove punctuation.
    """
    text = text.strip()
    text = text.lower()
    # Collapse all whitespace (including newlines, tabs) to a single space
    text = re.sub(r"\s+", " ", text)
    # Remove trivial punctuation that doesn't change meaning
    text = re.sub(r"[.,!?;:'\"()\[\]{}]", "", text)
    text = text.strip()
    return text


def normalize_numeric(text: str) -> str:
    """Normalize numeric representations.

    '4.0' -> '4', '3.50' -> '3.5', '1,000' -> '1000'
    Returns the original string unchanged if it is not a plain number.
    """
    # Remove thousands separators before trying to parse
    candidate = text.strip().replace(",", "")
    try:
        val = float(candidate)
        # Represent as integer if there is no fractional part
        if val == int(val):
            return str(int(val))
        # Otherwise strip trailing zeros
        return str(val)
    except (ValueError, OverflowError):
        return text


def normalize_answer(text: str, remove_articles: bool = False) -> str:
    """Full normalization pipeline for QA-style answers.

    Steps applied in order:
    1. Strip, lowercase, collapse whitespace  (no punctuation removal yet)
    2. Optional article removal  (the / a / an at the start)
    3. normalize_numeric  — if the whole string parses as a number, return its
       canonical form (e.g. "4.0" → "4") *before* punctuation is stripped so
       the decimal point is still present for parsing.
    4. Remove trivial punctuation.

    Example::

        normalize_answer(" Pacific ")          # -> "pacific"
        normalize_answer("4.0")                # -> "4"
        normalize_answer("William Shakespeare.") # -> "william shakespeare"
    """
    # Step 1: strip, lowercase, collapse whitespace (preserve punctuation for now)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)

    # Step 2: optional article removal
    if remove_articles:
        text = re.sub(r"^(the|a|an)\s+", "", text)

    # Step 3: numeric normalization BEFORE punctuation stripping
    # (so that "4.0" still has its "." when we try float())
    numeric_attempt = normalize_numeric(text)
    if numeric_attempt != text:
        return numeric_attempt

    # Step 4: remove trivial punctuation
    text = re.sub(r"[.,!?;:'\"()\[\]{}]", "", text)
    text = text.strip()
    return text
