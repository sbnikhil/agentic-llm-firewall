"""Characterization tests against the real models (Presidio/spaCy, MiniLM).

Pins the same behaviors as the unit tests, but with the models the app
actually loads, plus what Presidio redaction does to the same answers.
The models come from the local cache (or are downloaded on first run).
"""

from typing import Any

import numpy as np
import pytest

from llmfirewall.engine import models
from llmfirewall.engine.settings import get_settings
from tests.characterization_data import ids, load_data

pytestmark = pytest.mark.integration

HARMLESS = load_data("characterization/output_harmless.json")
BLOCKED_TODAY = [i for i in HARMLESS if not i["today"]["passed"]]
PASSED_TODAY = [i for i in HARMLESS if i["today"]["passed"]]
KEYWORD_SIM = load_data("characterization/output_keyword_similarity.json")

# MiniLM scores can drift slightly across platforms and torch builds.
SIMILARITY_TOLERANCE = 0.01


@pytest.fixture(scope="module")
def output_filter() -> Any:
    from llmfirewall.engine.detectors import output_filter as module

    assert type(models.sentence_transformer()).__name__ == "SentenceTransformer"
    return module


@pytest.fixture(scope="module")
def redactor() -> Any:
    from llmfirewall.engine.detectors import pii as module

    return module


def max_catalog_similarity(output_filter: Any, text: str) -> tuple[float, str]:
    """Best cosine similarity of `text` against secrets_catalog.json, computed
    the same way check_semantic_similarity does, but returning the score even
    when it is below LEAK_THRESHOLD."""
    model = models.sentence_transformer()
    response = model.encode(text)
    best, best_type = -1.0, ""
    for entry in models.load_secrets_catalog():
        secret = model.encode(entry["text"])
        sim = float(
            np.dot(response, secret)
            / (np.linalg.norm(response) * np.linalg.norm(secret))
        )
        if sim > best:
            best, best_type = sim, entry["type"]
    return best, best_type


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


@pytest.mark.parametrize("item", PASSED_TODAY, ids=ids(PASSED_TODAY))
def test_plain_harmless_answer_passes_today(
    output_filter: Any, item: dict[str, Any]
) -> None:
    similarity, _ = max_catalog_similarity(output_filter, item["text"])
    assert similarity <= get_settings().leak_threshold
    assert output_filter.validate_output(item["text"]) == (True, "Passed", 0.0)


# 4. The keyword-plus-similarity branch never fires (real MiniLM) -------------


@pytest.mark.parametrize("item", KEYWORD_SIM, ids=ids(KEYWORD_SIM))
def test_keyword_with_similarity_in_dead_window_passes_today(
    output_filter: Any, item: dict[str, Any]
) -> None:
    text = item["text"]
    similarity, nearest = max_catalog_similarity(output_filter, text)

    assert nearest == item["nearest_catalog_type"]
    assert similarity == pytest.approx(
        item["observed_max_similarity"], abs=SIMILARITY_TOLERANCE
    )
    # Inside the window where the keyword-plus-similarity branch was meant
    # to block: above 0.65, at or below LEAK_THRESHOLD.
    assert 0.65 < similarity <= get_settings().leak_threshold
    assert output_filter.check_keyword_presence(text)[0] is True
    assert output_filter.validate_output(text) == (True, "Passed", 0.0)


# Presidio: what redact_pii does today to the same harmless answers ----------


@pytest.mark.parametrize("item", HARMLESS, ids=ids(HARMLESS))
def test_redact_pii_output_today(redactor: Any, item: dict[str, Any]) -> None:
    assert redactor.redact_pii(item["text"]) == item["redact_pii_today"]
