"""Importing the engine must not load a model or open a connection.

Every module under llmfirewall is imported fresh while:
- the model libraries (sentence_transformers, presidio_*, spacy, torch,
  transformers, huggingface_hub) are replaced by modules that raise on any
  attribute access, so constructing or even naming a loader at import fails;
- opening a socket raises (autouse fixture in conftest.py).
Afterwards no model cache in llmfirewall.engine.models may hold anything.
"""

import importlib
import pkgutil
import sys
import types

import pytest

import llmfirewall

MODEL_LIBRARIES = (
    "sentence_transformers",
    "presidio_analyzer",
    "presidio_analyzer.nlp_engine",
    "presidio_anonymizer",
    "spacy",
    "torch",
    "transformers",
    "huggingface_hub",
)


class RaisingModule(types.ModuleType):
    def __getattr__(self, attr: str) -> object:
        raise AssertionError(f"{self.__name__}.{attr} used at import time")


def all_module_names() -> list[str]:
    names = [
        info.name
        for info in pkgutil.walk_packages(llmfirewall.__path__, "llmfirewall.")
    ]
    return ["llmfirewall", *sorted(names)]


def test_engine_modules_are_discovered() -> None:
    names = all_module_names()
    for expected in (
        "llmfirewall.engine.settings",
        "llmfirewall.engine.models",
        "llmfirewall.engine.detectors.input_rules",
        "llmfirewall.engine.detectors.pii",
        "llmfirewall.engine.detectors.output_filter",
    ):
        assert expected in names


def test_importing_every_module_loads_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in MODEL_LIBRARIES:
        monkeypatch.setitem(sys.modules, name, RaisingModule(name))
    # Forget the already-imported llmfirewall modules (restored afterwards)
    # so each one really runs its module body again here.
    for name in [n for n in sys.modules if n.split(".")[0] == "llmfirewall"]:
        monkeypatch.delitem(sys.modules, name)

    names = all_module_names()
    for name in names:
        importlib.import_module(name)

    models = sys.modules["llmfirewall.engine.models"]
    for loader in (
        models.sentence_transformer,
        models.presidio_analyzer,
        models.presidio_anonymizer,
        models.secrets_catalog_embeddings,
    ):
        assert loader.cache_info().currsize == 0, loader.__name__
