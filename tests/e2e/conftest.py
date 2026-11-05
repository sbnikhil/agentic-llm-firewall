"""Fixtures for the e2e suite: the compose stack in deploy/compose.

    cd deploy/compose && docker compose -f docker-compose.yml \
        -f docker-compose.e2e.yml up -d --build --wait
    set -a && . deploy/compose/.env && set +a
    LLMFW_E2E_BASE_URL=http://127.0.0.1:4000 uv run pytest tests/e2e

The suite is skipped only when LLMFW_E2E_BASE_URL is unset. When it is set,
an unreachable proxy is an error, not a skip (CI sets it).

Keys are created per session through LiteLLM's /key/generate with the
master key, for the users and teams bootstrap/bootstrap.py created, and
deleted afterwards. The master key is read from the environment and never
printed.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
from openai import OpenAI

BASE_URL = os.environ.get("LLMFW_E2E_BASE_URL", "").rstrip("/")
MOCK_URL = os.environ.get("LLMFW_E2E_MOCK_URL", "http://127.0.0.1:8001").rstrip("/")
MODEL = "chat"


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    for item in items:
        if "tests/e2e" in str(item.path).replace(os.sep, "/"):
            item.add_marker(pytest.mark.e2e)
            if not BASE_URL:
                item.add_marker(
                    pytest.mark.skip(reason="LLMFW_E2E_BASE_URL is not set")
                )


@pytest.fixture(scope="session")
def admin() -> Iterator[httpx.Client]:
    master = os.environ.get("LITELLM_MASTER_KEY")
    if not master:
        pytest.fail("LITELLM_MASTER_KEY is not set (source deploy/compose/.env)")
    with httpx.Client(
        base_url=BASE_URL, headers={"Authorization": f"Bearer {master}"}, timeout=60
    ) as client:
        assert client.get("/health/readiness").status_code == 200
        yield client


# What bootstrap/bootstrap.py gives every key (kept in sync by
# test_bootstrap_keys_are_restricted_to_the_chat_routes).
CHAT_ROUTES = [
    "/v1/chat/completions",
    "/chat/completions",
    f"/engines/{MODEL}/chat/completions",
    f"/openai/deployments/{MODEL}/chat/completions",
]
UNRESTRICTED = object()


@pytest.fixture(scope="session")
def make_key(admin: httpx.Client) -> Iterator[Callable[..., str]]:
    """Test keys, restricted to CHAT_ROUTES unless the test passes its own
    allowed_routes, or allowed_routes=UNRESTRICTED to create a key without
    any (which the guardrail must block)."""
    created: list[str] = []

    def make(
        user_id: str, team_id: str, allowed_routes: Any = CHAT_ROUTES, **extra: Any
    ) -> str:
        body: dict[str, Any] = {"user_id": user_id, "team_id": team_id, **extra}
        if allowed_routes is not UNRESTRICTED:
            body["allowed_routes"] = allowed_routes
        response = admin.post("/key/generate", json=body)
        assert response.status_code == 200, response.status_code
        key = response.json()["key"]
        created.append(key)
        return key

    yield make
    if created:
        admin.post("/key/delete", json={"keys": created})


def client_for(key: str) -> OpenAI:
    return OpenAI(base_url=f"{BASE_URL}/v1", api_key=key, max_retries=0, timeout=60)


def dev_key(name: str) -> str:
    key = os.environ.get(name)
    if not key:
        pytest.fail(f"{name} is not set (source deploy/compose/.env)")
    return key


@pytest.fixture(scope="session")
def alice_key() -> str:
    """The key bootstrap created for alice (team-standard -> policy default)."""
    return dev_key("LLMFW_DEV_ALICE_KEY")


@pytest.fixture(scope="session")
def bob_key() -> str:
    """The key bootstrap created for bob (team-strict -> policy strict)."""
    return dev_key("LLMFW_DEV_BOB_KEY")


@pytest.fixture(scope="session")
def alice(alice_key: str) -> OpenAI:
    return client_for(alice_key)


@pytest.fixture(scope="session")
def bob(bob_key: str) -> OpenAI:
    return client_for(bob_key)


@pytest.fixture
def seed() -> int:
    """Correlates one request with what the mock upstream received. Sent as
    the OpenAI `seed` parameter, which LiteLLM forwards and no detector
    reads (an id in message text could be redacted as PII)."""
    return secrets.randbits(31)


@pytest.fixture(scope="session")
def mock() -> Iterator[httpx.Client]:
    with httpx.Client(base_url=MOCK_URL, timeout=30) as client:
        yield client


@pytest.fixture
def upstream(mock: httpx.Client) -> Callable[[int], list[dict[str, Any]]]:
    """The request bodies the mocked upstream received for a seed."""

    def received(seed: int) -> list[dict[str, Any]]:
        response = mock.get(f"/_requests/{seed}")
        assert response.status_code == 200
        return response.json()

    return received


@pytest.fixture
def upstream_total(mock: httpx.Client) -> Callable[[], int]:
    """How many requests reached the mocked upstream, on any route."""

    def total() -> int:
        return mock.get("/_stats").json()["requests"]

    return total
