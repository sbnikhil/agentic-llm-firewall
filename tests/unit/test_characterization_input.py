"""Characterization tests for app.security.input_validator.validate_input.

They pin what the input check does today, so a later restructure can be
shown not to change it. Known problems are recorded as strict xfails that
describe the correct behavior; they are not fixed here.
"""

from typing import Any

import pytest

from app.security.input_validator import validate_input
from tests.characterization_data import ids, load_data

GOLDEN = load_data("golden_dataset.json")
GOLDEN_INPUT_BLOCK = [c for c in GOLDEN if c["expected_type"] == "input_block"]
GOLDEN_REASONS = load_data("characterization/golden_input_block_reasons.json")

HARMLESS = load_data("characterization/input_harmless.json")
BLOCKED_TODAY = [i for i in HARMLESS if not i["today"]["valid"]]
ALLOWED_TODAY = [i for i in HARMLESS if i["today"]["valid"]]
CLEARLY_HARMLESS_BLOCKED = [i for i in BLOCKED_TODAY if i["clearly_harmless"]]

ROLES = ["employee", "manager", "executive", "", "not-a-real-role"]


# 1. Golden input_block cases -------------------------------------------------


def test_golden_has_16_input_block_cases() -> None:
    assert len(GOLDEN_INPUT_BLOCK) == 16
    assert [c["test_name"] for c in GOLDEN_INPUT_BLOCK] == [
        r["test_name"] for r in GOLDEN_REASONS
    ]


@pytest.mark.parametrize(
    "pinned", GOLDEN_REASONS, ids=[r["test_name"] for r in GOLDEN_REASONS]
)
def test_golden_input_block_case_is_blocked(pinned: dict[str, Any]) -> None:
    case = GOLDEN[pinned["golden_index"]]
    assert case["test_name"] == pinned["test_name"]
    assert case["expected_type"] == "input_block"

    valid, reason = validate_input(case["input"], "employee")

    assert valid is False
    assert reason == pinned["today"]["reason"]


# 2. Over-blocking of harmless sentences ---------------------------------------


@pytest.mark.parametrize("item", BLOCKED_TODAY, ids=ids(BLOCKED_TODAY))
def test_harmless_phrase_is_blocked_today(item: dict[str, Any]) -> None:
    assert validate_input(item["text"], "employee") == (
        item["today"]["valid"],
        item["today"]["reason"],
    )


@pytest.mark.xfail(
    strict=True,
    reason="over-blocking: JAILBREAK_PATTERNS substring match blocks harmless "
    "sentences that merely contain a listed phrase",
)
@pytest.mark.parametrize(
    "item", CLEARLY_HARMLESS_BLOCKED, ids=ids(CLEARLY_HARMLESS_BLOCKED)
)
def test_clearly_harmless_phrase_should_be_allowed(item: dict[str, Any]) -> None:
    valid, _reason = validate_input(item["text"], "employee")
    assert valid is True


@pytest.mark.parametrize("item", ALLOWED_TODAY, ids=ids(ALLOWED_TODAY))
def test_harmless_words_without_full_phrase_are_allowed_today(
    item: dict[str, Any],
) -> None:
    assert validate_input(item["text"], "employee") == (True, "")


# 5. user_role has no effect ---------------------------------------------------


ROLE_INPUTS = (
    [c["input"] for c in GOLDEN_INPUT_BLOCK]
    + [i["text"] for i in HARMLESS]
    + ["", "   ", "x" * 2001, "!!!???***", "What is the remote work policy?"]
)


@pytest.mark.parametrize(
    "text", ROLE_INPUTS, ids=[f"input{i}" for i in range(len(ROLE_INPUTS))]
)
def test_user_role_does_not_change_result(text: str) -> None:
    results = {role: validate_input(text, role) for role in ROLES}
    assert len(set(results.values())) == 1, results
