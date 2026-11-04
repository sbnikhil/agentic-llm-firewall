"""Offline guards and fakes for unit tests.

Unit tests must never load a real model or touch the network:
- importing sentence_transformers, presidio, spaCy, torch or transformers fails
  (the output_filter fixture swaps in a fake SentenceTransformer instead);
- opening a socket fails;
- every test runs with tmp_path as the working directory, so any relative
  file write lands there.
"""

import importlib
import socket
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import app.security as security_pkg

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
    """A fresh import of app.security.output_filter backed by the fake model.

    The real module (if an integration test already imported it) is put back
    afterwards, so tests in other directories still see the real model.
    """
    fake_module = types.ModuleType("sentence_transformers")
    setattr(fake_module, "SentenceTransformer", FakeSentenceTransformer)  # noqa: B010
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    name = "app.security.output_filter"
    saved_module = sys.modules.pop(name, None)
    had_attr = hasattr(security_pkg, "output_filter")
    saved_attr = getattr(security_pkg, "output_filter", None)
    try:
        module = importlib.import_module(name)
        assert isinstance(module.model, FakeSentenceTransformer)
        yield module
    finally:
        sys.modules.pop(name, None)
        if saved_module is not None:
            sys.modules[name] = saved_module
        if had_attr:
            setattr(security_pkg, "output_filter", saved_attr)  # noqa: B010
        elif hasattr(security_pkg, "output_filter"):
            delattr(security_pkg, "output_filter")
