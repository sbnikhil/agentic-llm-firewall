"""Baseline ("before") latency for the v0 firewall components.

Times three functions directly (not the LangGraph nodes):
  input_check  = app.security.input_validator.validate_input(text, "employee")
  pii_redact   = app.redactor.redact_pii(text)
  output_filter= app.security.output_filter.validate_output(text)

Inputs: every "input" string in tests/data/golden_dataset.json (n=45). The
output filter is fed these user inputs, not model answers.

Two measurements:
  warm  - one process; import (models load), one discarded warm-up call per
          component, then PASSES passes over all 45 strings. Model loading excluded.
  cold  - COLD_RUNS fresh subprocesses per component; each times
          import (incl. model load) + the first call. Model loading included.
          Python interpreter startup itself is not included.

All torch work is forced onto CPU: before output_filter is imported, the
SentenceTransformer constructor it uses is wrapped to pass device="cpu"
(otherwise it picks MPS on Apple silicon), and the model's device is asserted.
No app code is modified.
Hugging Face is put in offline mode so loading is read from the local cache.

Usage (from repo root):  .venv/bin/python bench/baseline_v0.py
"""

import time

_T0 = time.perf_counter()  # cold-mode clock starts before any heavy import

import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "tests" / "data" / "golden_dataset.json"
PASSES = 5
COLD_RUNS = 10
COMPONENTS = ("input_check", "pii_redact", "output_filter")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.chdir(REPO)  # app code reads some files by relative path
sys.path.insert(0, str(REPO))


def force_cpu():
    """Make app code's `SentenceTransformer(name)` behave as `SentenceTransformer(name, device="cpu")`.

    output_filter.py does `from sentence_transformers import SentenceTransformer`
    at import time, so this must run before that import.
    """
    import functools

    import sentence_transformers
    import torch

    if not isinstance(sentence_transformers.SentenceTransformer, functools.partial):
        sentence_transformers.SentenceTransformer = functools.partial(
            sentence_transformers.SentenceTransformer, device="cpu"
        )
    return torch


def load(component):
    """Import a component (this is where its models load) and return its callable."""
    if component == "input_check":
        from app.security.input_validator import validate_input

        return lambda t: validate_input(t, "employee")
    if component == "pii_redact":
        from app.redactor import redact_pii

        return redact_pii
    if component == "output_filter":
        force_cpu()
        from app.security import output_filter

        dev = str(output_filter.model.device)
        assert dev == "cpu", f"model on {dev}, expected cpu"
        return output_filter.validate_output
    raise ValueError(component)


def inputs():
    return [c["input"] for c in json.loads(DATASET.read_text())]


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summary(ms):
    return {
        "n": len(ms),
        "p50_ms": round(pct(ms, 50), 3),
        "p95_ms": round(pct(ms, 95), 3),
        "mean_ms": round(statistics.fmean(ms), 3),
        "min_ms": round(min(ms), 3),
        "max_ms": round(max(ms), 3),
    }


def cold_child(component, idx):
    fn = load(component)
    fn(inputs()[idx % 45])
    print(json.dumps({"ms": (time.perf_counter() - _T0) * 1000}))


def warm():
    texts = inputs()
    fns = {c: load(c) for c in COMPONENTS}
    for fn in fns.values():
        fn(texts[0])  # discarded warm-up
    out = {}
    for c, fn in fns.items():
        ms = []
        for _ in range(PASSES):
            for t in texts:
                s = time.perf_counter()
                fn(t)
                ms.append((time.perf_counter() - s) * 1000)
        out[c] = summary(ms)
    return out


def cold():
    out = {}
    for c in COMPONENTS:
        ms = []
        for i in range(COLD_RUNS):
            r = subprocess.run(
                [sys.executable, __file__, "--cold-child", c, str(i)],
                capture_output=True,
                text=True,
                check=True,
            )
            ms.append(json.loads(r.stdout.strip().splitlines()[-1])["ms"])
        out[c] = summary(ms)
    return out


def environment():
    torch = force_cpu()
    from importlib.metadata import version

    pkgs = [
        "presidio-analyzer",
        "presidio-anonymizer",
        "spacy",
        "en-core-web-lg",
        "sentence-transformers",
        "transformers",
        "torch",
        "numpy",
        "langchain",
        "langgraph",
        "fastapi",
        "pytest",
    ]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_device": "cpu (SentenceTransformer device='cpu', CUDA hidden)",
        "mps_available_but_unused": torch.backends.mps.is_available(),
        "torch_num_threads": torch.get_num_threads(),
        "packages": {p: version(p) for p in pkgs},
        "n_inputs": len(inputs()),
        "passes": PASSES,
        "cold_runs": COLD_RUNS,
    }


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--cold-child":
        cold_child(sys.argv[2], int(sys.argv[3]))
    else:
        print(
            json.dumps({"env": environment(), "warm": warm(), "cold": cold()}, indent=2)
        )
