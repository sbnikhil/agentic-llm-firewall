"""Offline guards and fakes for unit tests.

Unit tests must never load a real model or touch the network:
- importing sentence_transformers, presidio, spaCy, torch or transformers fails
  (the output_filter fixture swaps in a fake SentenceTransformer instead);
- opening a socket fails;
- every test runs with tmp_path as the working directory, so any relative
  file write lands there.
"""

import socket
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from llmfirewall.engine import models

HEAVY_MODULES = (
    "sentence_transformers",
    "presidio_analyzer",
    "presidio_anonymizer",
    "spacy",
    "torch",
    "transformers",
)


@pytest.fixture(autouse=True)
def _offline_and_isolated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[None]:
    def refuse(*args: object, **kwargs: object) -> None:
        raise RuntimeError("unit tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    # A None entry in sys.modules makes `import <name>` raise ImportError.
    for name in HEAVY_MODULES:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.chdir(tmp_path)
    yield


class FakeSentenceTransformer:
    """Stands in for SentenceTransformer: returns vectors chosen by the test.

    Any text not in `vectors` encodes to DEFAULT_VECTOR, so every
    secrets_catalog.json entry encodes to DEFAULT_VECTOR. A response vector v
    then has cosine similarity v[0] / |v| with every catalog entry.
    """

    DEFAULT_VECTOR = np.array([1.0, 0.0, 0.0])

    def __init__(self, model_name_or_path: str, *args: object, **kwargs: object):
        self.model_name_or_path = model_name_or_path
        self.vectors: dict[str, np.ndarray] = {}

    def encode(self, text: str) -> np.ndarray:
        return self.vectors.get(text, self.DEFAULT_VECTOR)


@pytest.fixture
def output_filter(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """llmfirewall.engine.detectors.output_filter backed by the fake model.

    The engine's model caches are cleared before and after, so the fake is
    created fresh for this test and never leaks into other tests.
    """
    fake_module = types.ModuleType("sentence_transformers")
    setattr(fake_module, "SentenceTransformer", FakeSentenceTransformer)  # noqa: B010
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    from llmfirewall.engine.detectors import output_filter as module

    models.clear_caches()
    try:
        assert isinstance(models.sentence_transformer(), FakeSentenceTransformer)
        yield module
    finally:
        models.clear_caches()


@pytest.fixture
def fake_model(output_filter: Any) -> FakeSentenceTransformer:
    """The fake SentenceTransformer the output_filter fixture installed."""
    model = models.sentence_transformer()
    assert isinstance(model, FakeSentenceTransformer)
    return model
