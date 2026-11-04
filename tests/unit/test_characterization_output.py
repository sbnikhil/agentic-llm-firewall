"""Characterization tests for app.security.output_filter.validate_output.

Runs offline: the `output_filter` fixture (tests/unit/conftest.py) imports the
module with a fake SentenceTransformer whose vectors the test chooses.
Known problems are recorded as strict xfails that describe the correct
behavior; they are not fixed here.
"""

import math
from typing import Any

import numpy as np
import pytest

from tests.characterization_data import ids, load_data

HARMLESS = load_data("characterization/output_harmless.json")
BLOCKED_TODAY = [i for i in HARMLESS if not i["today"]["passed"]]
PASSED_TODAY = [i for i in HARMLESS if i["today"]["passed"]]
KEYWORD_SIM = load_data("characterization/output_keyword_similarity.json")


def unit_vector_with_similarity(similarity: float) -> np.ndarray:
    """A vector whose cosine similarity with FakeSentenceTransformer's
    DEFAULT_VECTOR (and so with every catalog entry) is `similarity`."""
    return np.array([similarity, math.sqrt(1.0 - similarity**2), 0.0])


# 3. Harmless answers with an email, phone, dollar amount or 9-digit number ----


@pytest.mark.parametrize("item", BLOCKED_TODAY, ids=ids(BLOCKED_TODAY))
def test_harmless_answer_with_pattern_is_blocked_today(
    output_filter: Any, item: dict[str, Any]
) -> None:
    today = item["today"]
    assert output_filter.validate_output(item["text"]) == (
        today["passed"],
        today["reason"],
        today["score"],
    )


@pytest.mark.xfail(
    strict=True,
    reason="over-blocking: SECRET_PATTERNS block harmless answers containing an "
    "email, phone number, dollar amount or 9-digit number",
)
@pytest.mark.parametrize("item", BLOCKED_TODAY, ids=ids(BLOCKED_TODAY))
def test_harmless_answer_with_pattern_should_not_be_blocked(
    output_filter: Any, item: dict[str, Any]
) -> None:
    # Checks only the block decision; the answer text is deliberately not
    # compared, so redacting the value instead of blocking counts as correct.
    passed, _reason, _score = output_filter.validate_output(item["text"])
    assert passed is True


@pytest.mark.parametrize("item", PASSED_TODAY, ids=ids(PASSED_TODAY))
def test_plain_harmless_answer_passes_today(
    output_filter: Any, item: dict[str, Any]
) -> None:
    output_filter.model.vectors[item["text"]] = np.array([0.0, 1.0, 0.0])
    assert output_filter.validate_output(item["text"]) == (True, "Passed", 0.0)


# 4. The keyword-plus-similarity branch never fires ----------------------------
# check_semantic_similarity returns a score of 0.0 whenever the best match is
# at or below LEAK_THRESHOLD (0.75), so `similarity > 0.65` in validate_output
# can never be true. Pinned here; recorded in docs/v0-findings.md.


@pytest.mark.parametrize("item", KEYWORD_SIM, ids=ids(KEYWORD_SIM))
def test_keyword_with_similarity_0_70_passes_today(
    output_filter: Any, item: dict[str, Any]
) -> None:
    text = item["text"]
    assert output_filter.check_keyword_presence(text) == (
        True,
        f"Sensitive keyword: {item['keyword']}",
    )
    assert output_filter.check_pattern_match(text) == (False, "")
    output_filter.model.vectors[text] = unit_vector_with_similarity(0.70)

    assert output_filter.check_semantic_similarity(text) == (False, "", 0.0)
    assert output_filter.validate_output(text) == (True, "Passed", 0.0)


def test_similarity_0_80_blocks_today(output_filter: Any) -> None:
    text = "The CEO salary is set by the board each year."
    output_filter.model.vectors[text] = unit_vector_with_similarity(0.80)

    passed, reason, score = output_filter.validate_output(text)

    assert passed is False
    # Every catalog entry is equally similar; the loop stops at the first one.
    assert reason == "Blocked: Similar to gym_passcode"
    assert score == pytest.approx(0.80)


def test_keyword_alone_passes_today(output_filter: Any) -> None:
    text = "The CEO salary is set by the board each year."
    output_filter.model.vectors[text] = np.array([0.0, 1.0, 0.0])

    assert output_filter.validate_output(text) == (True, "Passed", 0.0)
