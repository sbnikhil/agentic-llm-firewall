"""Characterization of the golden cases the other tests do not pin.

With the real models, pins today's exact detector output for:
- the 9 `redaction` cases: the full redact_pii output;
- the 20 `pass` / `audit_fail` cases: the full validate_output tuple.

The `input_block` cases are pinned in tests/unit/test_characterization_input.py.
Expected values are in tests/data/characterization/golden_detector_outputs.json.
"""

from collections.abc import Callable
from typing import Any

import pytest

from tests.characterization_data import load_data

pytestmark = pytest.mark.integration

GOLDEN = load_data("golden_dataset.json")
PINNED = load_data("characterization/golden_detector_outputs.json")
REDACTION = PINNED["redaction"]
OUTPUT = PINNED["output"]

# MiniLM scores can drift slightly across platforms and torch builds.
SIMILARITY_TOLERANCE = 0.01


def golden_case(pinned: dict[str, Any], expected_type: str) -> dict[str, Any]:
    case = GOLDEN[pinned["golden_index"]]
    assert case["test_name"] == pinned["test_name"]
    assert case["expected_type"] == expected_type
    return case


def test_every_redaction_pass_and_audit_fail_case_is_pinned() -> None:
    def indices(*types: str) -> list[int]:
        return [i for i, c in enumerate(GOLDEN) if c["expected_type"] in types]

    assert [p["golden_index"] for p in REDACTION] == indices("redaction")
    assert [p["golden_index"] for p in OUTPUT] == indices("pass", "audit_fail")
    assert (len(REDACTION), len(OUTPUT)) == (9, 20)


@pytest.mark.parametrize("pinned", REDACTION, ids=[p["test_name"] for p in REDACTION])
def test_golden_redaction_output_today(
    redact_pii: Callable[[str], str], pinned: dict[str, Any]
) -> None:
    case = golden_case(pinned, "redaction")
    assert redact_pii(case["input"]) == pinned["redact_pii_today"]


@pytest.mark.parametrize("pinned", OUTPUT, ids=[p["test_name"] for p in OUTPUT])
def test_golden_validate_output_today(
    validate_output: Callable[[str], Any], pinned: dict[str, Any]
) -> None:
    case = golden_case(pinned, pinned["expected_type"])
    today = pinned["today"]

    passed, reason, score = validate_output(case["input"])

    assert passed == today["passed"]
    assert reason == today["reason"]
    assert score == pytest.approx(today["score"], abs=SIMILARITY_TOLERANCE)
