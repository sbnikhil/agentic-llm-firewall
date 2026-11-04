"""Shared fixtures for integration tests: the real detector functions.

Tests that only call detector functions get them from here, so the code
under test can be re-pointed in one place without editing the tests.
"""

from collections.abc import Callable
from typing import Any

import pytest


@pytest.fixture(scope="session")
def redact_pii() -> Callable[[str], str]:
    from llmfirewall.engine.detectors.pii import redact_pii

    return redact_pii


@pytest.fixture(scope="session")
def validate_output() -> Callable[[str], Any]:
    from llmfirewall.engine.detectors.output_filter import validate_output

    return validate_output
