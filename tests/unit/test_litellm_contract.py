"""The parts of LiteLLM the adapter depends on, pinned.

Checked against litellm==1.108.3 (ADR 0003). If an upgrade changes any of
these, this file fails in CI before the adapter silently stops being called
or starts reading the wrong argument. Fix the adapter, then update the
expectation here on purpose.

Parameter names, order, kinds and defaults are pinned; annotations are not,
because LiteLLM types several of them under TYPE_CHECKING only.
"""

from __future__ import annotations

import inspect
from typing import Any

import litellm
import pytest
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth

from llmfirewall.adapters.litellm_guardrail import LLMFirewallGuardrail

POS = inspect.Parameter.POSITIONAL_OR_KEYWORD
EMPTY = inspect.Parameter.empty


def shape(fn: Any) -> list[tuple[str, Any, Any]]:
    return [
        (p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()
    ]


def test_litellm_version_is_the_pinned_one() -> None:
    from importlib.metadata import version

    assert version("litellm") == "1.108.3"
    assert litellm is not None


HOOKS = {
    "async_pre_call_hook": [
        ("self", POS, EMPTY),
        ("user_api_key_dict", POS, EMPTY),
        ("cache", POS, EMPTY),
        ("data", POS, EMPTY),
        ("call_type", POS, EMPTY),
    ],
    "async_post_call_success_hook": [
        ("self", POS, EMPTY),
        ("data", POS, EMPTY),
        ("user_api_key_dict", POS, EMPTY),
        ("response", POS, EMPTY),
    ],
    "async_post_call_streaming_iterator_hook": [
        ("self", POS, EMPTY),
        ("user_api_key_dict", POS, EMPTY),
        ("response", POS, EMPTY),
        ("request_data", POS, EMPTY),
    ],
    "apply_guardrail": [
        ("self", POS, EMPTY),
        ("inputs", POS, EMPTY),
        ("request_data", POS, EMPTY),
        ("input_type", POS, EMPTY),
        ("logging_obj", POS, None),
    ],
    "should_run_guardrail": [
        ("self", POS, EMPTY),
        ("data", POS, EMPTY),
        ("event_type", POS, EMPTY),
    ],
}


@pytest.mark.parametrize("name", sorted(HOOKS))
def test_litellm_hook_signature(name: str) -> None:
    assert shape(getattr(CustomGuardrail, name)) == HOOKS[name]


@pytest.mark.parametrize(
    "name",
    [
        "async_pre_call_hook",
        "async_post_call_success_hook",
        "async_post_call_streaming_iterator_hook",
        "should_run_guardrail",
    ],
)
def test_adapter_overrides_match_litellm(name: str) -> None:
    """The proxy calls these with keyword arguments: the names must match."""
    ours = [(n, k) for n, k, _ in shape(getattr(LLMFirewallGuardrail, name))]
    theirs = [(n, k) for n, k, _ in HOOKS[name]]
    assert ours == theirs


def test_hooks_are_defined_on_the_adapter_class_itself() -> None:
    """LiteLLM looks hooks up in the leaf class's own __dict__
    (ProxyLogging.callback_capabilities), not through inheritance."""
    for name in (
        "async_pre_call_hook",
        "async_post_call_success_hook",
        "async_post_call_streaming_iterator_hook",
        "should_run_guardrail",
    ):
        assert name in LLMFirewallGuardrail.__dict__, name


def test_adapter_does_not_define_apply_guardrail() -> None:
    """A subclass that defines apply_guardrail is routed through LiteLLM's
    unified guardrail instead of the hooks above (proxy/utils.py)."""
    assert "apply_guardrail" not in LLMFirewallGuardrail.__dict__
    assert LLMFirewallGuardrail.apply_guardrail is CustomGuardrail.apply_guardrail


def test_custom_guardrail_still_inherits_the_logger_hooks() -> None:
    assert issubclass(CustomGuardrail, CustomLogger)


def test_custom_guardrail_init_accepts_the_registry_arguments() -> None:
    params = inspect.signature(CustomGuardrail.__init__).parameters
    for name in ("guardrail_name", "event_hook", "default_on"):
        assert name in params
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def test_user_api_key_auth_carries_the_verified_ids() -> None:
    """The only two fields identity.py reads."""
    fields = UserAPIKeyAuth.model_fields
    assert "user_id" in fields
    assert "team_id" in fields
    auth = UserAPIKeyAuth(user_id="u1", team_id="t1")
    assert (auth.user_id, auth.team_id) == ("u1", "t1")


def test_master_key_admin_user_id_is_the_known_sentinel() -> None:
    """Master-key calls get this user_id and no team: unmapped, so they get
    the default policy (docs/design/policy.md section 3)."""
    from litellm.constants import LITELLM_PROXY_ADMIN_NAME

    assert LITELLM_PROXY_ADMIN_NAME == "default_user_id"
