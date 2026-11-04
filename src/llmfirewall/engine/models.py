"""Models and model-derived data, each created once on first use and cached.

Nothing here loads at import: the heavy libraries are imported inside the
loader functions. Call clear_caches() to drop everything (tests do this).
"""

from __future__ import annotations

import json
from functools import cache
from typing import TYPE_CHECKING, Any

from llmfirewall.engine.settings import get_settings

if TYPE_CHECKING:
    import numpy as np
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from sentence_transformers import SentenceTransformer


@cache
def sentence_transformer() -> SentenceTransformer:
    """The one SentenceTransformer shared by every detector."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().sentence_model)


@cache
def presidio_analyzer() -> AnalyzerEngine:
    """Presidio's default analyzer (spaCy NLP engine from Presidio's default
    configuration, with the model name from settings) plus the custom
    recognizers in detectors.pii."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    from llmfirewall.engine.detectors.pii import custom_recognizers

    provider = NlpEngineProvider()  # reads Presidio's default.yaml
    provider.nlp_configuration["models"] = [
        {"lang_code": "en", "model_name": get_settings().spacy_model}
    ]
    analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
    for recognizer in custom_recognizers():
        analyzer.registry.add_recognizer(recognizer)
    return analyzer


@cache
def presidio_anonymizer() -> AnonymizerEngine:
    from presidio_anonymizer import AnonymizerEngine

    return AnonymizerEngine()


def load_secrets_catalog() -> list[dict[str, Any]]:
    """Read the secrets catalog file; an empty list if it does not exist."""
    try:
        with open(get_settings().secrets_catalog_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


@cache
def secrets_catalog_embeddings() -> tuple[tuple[dict[str, Any], np.ndarray], ...]:
    """Each catalog entry with its embedding, encoded once per process.

    Entries are encoded one at a time, as v0 did on every call, so the
    vectors are the same."""
    model = sentence_transformer()
    return tuple(
        (entry, model.encode(entry.get("text", ""))) for entry in load_secrets_catalog()
    )


def clear_caches() -> None:
    for loader in (
        sentence_transformer,
        presidio_analyzer,
        presidio_anonymizer,
        secrets_catalog_embeddings,
    ):
        loader.cache_clear()
    get_settings.cache_clear()
