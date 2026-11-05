"""The LiteLLM adapter with fake detectors: identity, gating, call types,
error bodies, audit events and the streaming hook. Offline.

The integration suite (tests/integration/test_litellm_adapter.py) runs the
same hooks with the real detectors and LiteLLM's own stream wrapper.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import HTTPException
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import (
    Choices,
    Delta,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)
from pydantic import SecretStr

from llmfirewall.adapters.litellm_guardrail import (
    CHAT_CALL_TYPES,
    UNRESTRICTED_KEY,
    UNSUPPORTED_CALL_TYPE,
    LLMFirewallGuardrail,
    key_is_route_restricted,
)
from llmfirewall.engine.audit import AuditEvent, MissingAuditKey, user_id_hmac
from llmfirewall.engine.engine import Engine
from llmfirewall.engine.identity import PolicyResolver, parse_identity_map
from llmfirewall.engine.settings import Settings
from tests.fake_secrets import GUARDRAIL_API_KEY
from tests.unit.engine_fakes import (
    EXPENSIVE,
    FakeDetector,
    detector_config,
    finds,
    flags,
    make_policy,
)

KEY = "test-hmac-key"
SECRET = GUARDRAIL_API_KEY
BOMB = "detonate-now"


class ListSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def emit(self, event: AuditEvent) -> None:
        self.events.append(event)


def detectors() -> list[FakeDetector]:
    return [
        FakeDetector("secret", EXPENSIVE, finds(SECRET, "API_KEY")),
        FakeDetector("bomb", EXPENSIVE, flags(BOMB, "BOMB")),
    ]


def build(
    *, sink: ListSink | None = None, holdback: int = 16, failing: bool = False
) -> tuple[LLMFirewallGuardrail, ListSink]:
    dets = detectors()
    if failing:
        dets[0] = FakeDetector("secret", EXPENSIVE, error=RuntimeError("boom"))
    roles = ["user", "response"]
    default = make_policy(
        dets,
        {
            "secret": detector_config("redact", roles),
            "bomb": detector_config("block", roles),
        },
        name="default",
    )
    strict = make_policy(
        dets,
        {
            "secret": detector_config("block", roles),
            "bomb": detector_config("block", roles),
        },
        name="strict",
    )
    policies = {"default": default, "strict": strict}
    identity_map = parse_identity_map(
        {"version": 1, "default": "default", "teams": {"team-strict": "strict"}},
        policies,
    )
    sink = sink or ListSink()
    guard = LLMFirewallGuardrail(
        engine=Engine(dets),
        resolver=PolicyResolver(identity_map, policies),
        settings=Settings(
            audit_hmac_key=SecretStr(KEY),
            stream_holdback_chars=holdback,
            stream_context_chars=8,
            stream_scan_step_chars=1,
        ),
        audit_sink=sink,
        guardrail_name="llmfirewall",
        default_on=True,
    )
    return guard, sink


CHAT_ROUTES = ["/v1/chat/completions", "/chat/completions"]


def auth(
    user: str | None = "alice",
    team: str | None = "team-default",
    allowed_routes: Any = CHAT_ROUTES,
) -> UserAPIKeyAuth:
    """A verified key. Restricted to the chat routes by default, as every
    key must be (the adapter blocks unrestricted keys)."""
    return UserAPIKeyAuth(user_id=user, team_id=team, allowed_routes=allowed_routes)


def chat(text: str, **extra: Any) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": text}], **extra}


def pre(
    guard: LLMFirewallGuardrail,
    data: dict[str, Any],
    who: UserAPIKeyAuth,
    call_type: str = "completion",
) -> Any:
    return asyncio.run(
        guard.async_pre_call_hook(
            user_api_key_dict=who, cache=None, data=data, call_type=call_type
        )
    )


def detail(e: pytest.ExceptionInfo[HTTPException]) -> dict[str, Any]:
    body: Any = e.value.detail
    assert isinstance(body, dict)
    return body["error"]


# -- setup -------------------------------------------------------------------


def test_refuses_to_start_without_the_audit_key() -> None:
    dets = detectors()
    policy = make_policy(
        dets,
        {
            "secret": detector_config("redact", ["user"]),
            "bomb": detector_config("block", ["user"]),
        },
        name="default",
    )
    resolver = PolicyResolver(
        parse_identity_map({"version": 1, "default": "default"}, {"default": policy}),
        {"default": policy},
    )
    for key in (None, SecretStr("")):
        with pytest.raises(MissingAuditKey):
            LLMFirewallGuardrail(
                engine=Engine(dets),
                resolver=resolver,
                settings=Settings(audit_hmac_key=key),
                audit_sink=ListSink(),
            )


# -- gating ------------------------------------------------------------------

BYPASS_ATTEMPTS: list[dict[str, Any]] = [
    {},
    {"guardrails": []},
    {"guardrails": ["something-else"]},
    {"metadata": {"guardrails": []}},
    {"litellm_metadata": {"guardrails": []}},
    {"metadata": {"user_api_key_metadata": {"disable_global_guardrails": True}}},
    {"metadata": {"user_api_key_team_metadata": {"disable_global_guardrails": True}}},
    {
        "metadata": {
            "user_api_key_metadata": {"opted_out_global_guardrails": ["llmfirewall"]}
        }
    },
    {
        "litellm_metadata": {
            "user_api_key_metadata": {"disable_global_guardrails": True}
        }
    },
]


@pytest.mark.parametrize("data", BYPASS_ATTEMPTS)
@pytest.mark.parametrize("event", list(GuardrailEventHooks))
def test_nothing_in_the_request_or_metadata_turns_the_guardrail_off(
    data: dict[str, Any], event: GuardrailEventHooks
) -> None:
    guard, _ = build()
    assert guard.should_run_guardrail(data, event) is True


def test_litellm_default_would_have_been_switched_off_by_metadata() -> None:
    """Why the override exists: the inherited method honours these fields."""
    from litellm.integrations.custom_guardrail import CustomGuardrail

    guard, _ = build()
    data = {"metadata": {"user_api_key_metadata": {"disable_global_guardrails": True}}}
    assert (
        CustomGuardrail.should_run_guardrail(guard, data, GuardrailEventHooks.pre_call)
        is False
    )


# -- call types --------------------------------------------------------------


def all_call_types() -> list[str]:
    from litellm.types.utils import CallTypes

    return sorted(c.value for c in CallTypes)


@pytest.mark.parametrize(
    "call_type", [c for c in all_call_types() if c not in CHAT_CALL_TYPES]
)
def test_every_non_chat_call_type_is_blocked(call_type: str) -> None:
    guard, sink = build()
    with pytest.raises(HTTPException) as e:
        pre(guard, chat("hello"), auth(), call_type)
    assert e.value.status_code == 400
    assert detail(e)["message"] == UNSUPPORTED_CALL_TYPE
    assert sink.events[-1].reason == "unsupported_call_type"
    assert sink.events[-1].action == "block"


@pytest.mark.parametrize("call_type", sorted(CHAT_CALL_TYPES))
def test_chat_call_types_are_checked(call_type: str) -> None:
    guard, sink = build()
    assert (
        pre(guard, chat("hello"), auth(), call_type)["messages"][0]["content"]
        == "hello"
    )
    assert sink.events[-1].reason == "decision"


# -- identity ----------------------------------------------------------------

SPOOFS: list[dict[str, Any]] = [
    {"metadata": {"policy": "default"}},
    {"metadata": {"llmfirewall_policy": "default", "policy_name": "default"}},
    {"litellm_metadata": {"policy": "default"}},
    {"user": "default"},
    {
        "metadata": {
            "user_api_key_team_id": "team-default",
            "user_api_key_user_id": "nobody",
        }
    },
    {"metadata": {"user_api_key_team_metadata": {"policy": "default"}}},
    {
        "proxy_server_request": {
            "headers": {
                "x-litellm-policy": "default",
                "x-litellm-team-id": "team-default",
            }
        }
    },
    {"policy": "default"},
]


@pytest.mark.parametrize("spoof", SPOOFS)
def test_client_set_fields_cannot_choose_the_policy(spoof: dict[str, Any]) -> None:
    """team-strict blocks the secret; no field in the request makes it the
    redacting default policy."""
    guard, sink = build()
    with pytest.raises(HTTPException) as e:
        pre(guard, chat(f"my key is {SECRET}", **spoof), auth("bob", "team-strict"))
    assert e.value.status_code == 400
    assert sink.events[-1].policy_name == "strict"


def test_default_team_gets_the_default_policy() -> None:
    guard, sink = build()
    data = pre(guard, chat(f"my key is {SECRET}"), auth("alice", "team-default"))
    assert data["messages"][0]["content"] == "my key is [REDACTED:API_KEY]"
    assert sink.events[-1].policy_name == "default"
    assert sink.events[-1].action == "redact"


def test_master_key_identity_gets_the_default_policy() -> None:
    guard, sink = build()
    pre(guard, chat("hello"), auth("default_user_id", None))
    assert sink.events[-1].policy_name == "default"


# -- request decisions -------------------------------------------------------


def test_block_error_names_detectors_but_contains_no_matched_text() -> None:
    guard, _ = build()
    with pytest.raises(HTTPException) as e:
        pre(guard, chat(f"please {BOMB} {SECRET}"), auth())
    body = json.dumps(e.value.detail)
    assert BOMB not in body and SECRET not in body
    assert detail(e)["detectors"] == ["bomb", "secret"]
    assert detail(e)["entity_types"] == ["API_KEY", "BOMB"]


def test_messages_that_are_not_a_list_are_blocked() -> None:
    guard, sink = build()
    with pytest.raises(HTTPException) as e:
        pre(guard, {"messages": "hello"}, auth())
    assert e.value.status_code == 500
    assert sink.events[-1].reason == "internal_error"


def test_detector_error_fails_closed() -> None:
    guard, _ = build(failing=True)
    with pytest.raises(HTTPException) as e:
        pre(guard, chat("hello"), auth())
    assert e.value.status_code == 400
    assert detail(e)["failures"] == ["error"]


# -- audit -------------------------------------------------------------------


def test_audit_event_has_team_and_user_hmac_but_no_raw_user_or_text() -> None:
    guard, sink = build()
    pre(
        guard,
        chat(f"my key is {SECRET}", litellm_call_id="not-a-uuid"),
        auth("alice@example.com", "team-default"),
    )
    event = sink.events[-1]
    raw = event.model_dump_json()
    assert event.team_id == "team-default"
    assert event.user_id_hmac == user_id_hmac(KEY.encode(), "alice@example.com")
    assert "alice" not in raw
    assert SECRET not in raw and "my key is" not in raw
    assert event.call_id is None  # not a UUID: dropped
    assert event.counts == {"API_KEY": 1}
    assert set(event.detector_ms) == {"bomb", "secret"}
    assert event.findings[0].start == len("my key is ")


def test_audit_keeps_a_uuid_call_id() -> None:
    guard, sink = build()
    call_id = "1b4e28ba-2fa1-11d2-883f-0016d3cca427"
    pre(guard, chat("hi", litellm_call_id=call_id), auth())
    assert sink.events[-1].call_id == call_id


# -- non-streaming responses -------------------------------------------------


def response(text: str | None) -> ModelResponse:
    return ModelResponse(
        choices=[
            Choices(
                index=0,
                message=Message(role="assistant", content=text),
                finish_reason="stop",
            )
        ]
    )


def post(
    guard: LLMFirewallGuardrail, resp: Any, who: UserAPIKeyAuth | None = None
) -> Any:
    return asyncio.run(
        guard.async_post_call_success_hook(
            data={}, user_api_key_dict=who or auth(), response=resp
        )
    )


def test_response_redaction() -> None:
    guard, sink = build()
    original = response(f"your key: {SECRET}")
    out = post(guard, original)
    assert out.choices[0].message.content == "your key: [REDACTED:API_KEY]"
    assert original.choices[0].message.content == f"your key: {SECRET}"
    assert sink.events[-1].direction == "response"


def test_response_block() -> None:
    guard, _ = build()
    with pytest.raises(HTTPException) as e:
        post(guard, response(f"ok {BOMB}"))
    assert BOMB not in json.dumps(e.value.detail)
    assert detail(e)["direction"] == "response"


def test_response_without_text_passes() -> None:
    guard, _ = build()
    assert post(guard, response(None)).choices[0].message.content is None


def test_unknown_response_type_is_blocked() -> None:
    guard, sink = build()
    with pytest.raises(HTTPException):
        post(guard, {"choices": []})
    assert sink.events[-1].reason == "unsupported_response"


# -- streaming ---------------------------------------------------------------


def stream_chunks(parts: list[str]) -> list[ModelResponseStream]:
    chunks = [
        ModelResponseStream(
            id="c1",
            model="m",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content=p, role="assistant" if i == 0 else None),
                )
            ],
        )
        for i, p in enumerate(parts)
    ]
    chunks.append(
        ModelResponseStream(
            id="c1",
            model="m",
            choices=[
                StreamingChoices(
                    index=0, delta=Delta(content=None), finish_reason="stop"
                )
            ],
        )
    )
    return chunks


async def agen(items: list[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


def stream(
    guard: LLMFirewallGuardrail, parts: list[str], who: UserAPIKeyAuth | None = None
) -> list[ModelResponseStream]:
    async def collect() -> list[ModelResponseStream]:
        return [
            c
            async for c in guard.async_post_call_streaming_iterator_hook(
                user_api_key_dict=who or auth(),
                response=agen(stream_chunks(parts)),
                request_data={},
            )
        ]

    return asyncio.run(collect())


def text_of(chunks: list[ModelResponseStream]) -> str:
    return "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)


@pytest.mark.parametrize("cut", range(1, len(SECRET)))
def test_stream_secret_split_across_chunks_is_redacted(cut: int) -> None:
    guard, sink = build()
    text = f"here: {SECRET} done, and more text."
    i = text.index(SECRET) + cut
    chunks = stream(guard, [text[:i], text[i:]])
    assert text_of(chunks) == "here: [REDACTED:API_KEY] done, and more text."
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert sink.events[-1].direction == "stream"
    assert sink.events[-1].action == "redact"


def test_stream_block_ends_with_content_filter_and_sends_nothing_held() -> None:
    guard, sink = build()  # N=16 >= len(BOMB)
    text = f"a fairly long harmless opening sentence. {BOMB} secret tail"
    chunks = stream(guard, list(text))
    received = text_of(chunks)
    assert BOMB[:3] not in received
    assert chunks[-1].choices[0].finish_reason == "content_filter"
    assert chunks[-1].choices[0].delta.content is None
    event = sink.events[-1]
    assert event.action == "block"
    assert event.stream_stop is not None
    assert event.stream_stop.reason == "content_filter"
    assert event.stream_stop.released_chars == len(received)
    assert event.stream_stop.withheld_chars > 0
    assert BOMB not in event.model_dump_json()


def test_stream_strict_policy_blocks_a_split_secret() -> None:
    guard, _ = build()
    text = f"here: {SECRET} done"
    chunks = stream(guard, [text[:9], text[9:]], auth("bob", "team-strict"))
    assert chunks[-1].choices[0].finish_reason == "content_filter"
    assert "sk-" not in text_of(chunks)


def test_stream_detector_error_ends_with_content_filter() -> None:
    guard, sink = build(failing=True)
    chunks = stream(guard, list("hello there, this is a stream"))
    assert chunks[-1].choices[0].finish_reason == "content_filter"
    assert sink.events[-1].action == "block"


def test_stream_unknown_chunk_type_is_blocked() -> None:
    guard, sink = build()

    async def collect() -> list[Any]:
        return [
            c
            async for c in guard.async_post_call_streaming_iterator_hook(
                user_api_key_dict=auth(), response=agen([b"raw bytes"]), request_data={}
            )
        ]

    chunks = asyncio.run(collect())
    assert chunks[-1].choices[0].finish_reason == "content_filter"
    assert sink.events[-1].reason == "unsupported_response"


def test_stream_without_finish_reason_still_releases_after_the_final_check() -> None:
    guard, _ = build()

    async def collect() -> list[Any]:
        parts = stream_chunks(["hello ", "world"])[:-1]  # no finish chunk
        return [
            c
            async for c in guard.async_post_call_streaming_iterator_hook(
                user_api_key_dict=auth(), response=agen(parts), request_data={}
            )
        ]

    assert text_of(asyncio.run(collect())) == "hello world"


def test_holdback_shorter_than_the_trigger_length_leaks_a_prefix() -> None:
    """Why N must be at least the length of the shortest text that makes a
    detector fire: this fake only fires on all 15 characters, so with N=8
    the first characters are released before it can (ADR 0004)."""
    guard, _ = build(holdback=8)
    text = f"here: {SECRET} done, and more text."
    i = text.index(SECRET) + 9
    assert text_of(stream(guard, [text[:i], text[i:]])).startswith("here: s")


# -- keys without allowed_routes -----------------------------------------------


@pytest.mark.parametrize("routes", [None, [], "", "/v1/chat/completions", {}])
def test_unrestricted_key_is_blocked_before_any_check(routes: Any) -> None:
    """None, an empty list or a non-list all mean "every route" to LiteLLM."""
    guard, sink = build()
    who = UserAPIKeyAuth(user_id="alice", team_id="team-default")
    object.__setattr__(who, "allowed_routes", routes)
    with pytest.raises(HTTPException) as e:
        pre(guard, chat("hello"), who)
    assert e.value.status_code == 403
    assert detail(e)["message"] == UNRESTRICTED_KEY
    assert sink.events[-1].reason == "unrestricted_key"
    assert sink.events[-1].action == "block"
    assert sink.events[-1].checks == 0


def test_unrestricted_key_is_blocked_even_for_non_chat_call_types() -> None:
    guard, sink = build()
    with pytest.raises(HTTPException) as e:
        pre(guard, chat("hello"), auth(allowed_routes=[]), "pass_through_endpoint")
    assert detail(e)["message"] == UNRESTRICTED_KEY


def test_master_key_identity_without_routes_is_blocked() -> None:
    guard, _ = build()
    with pytest.raises(HTTPException) as e:
        pre(guard, chat("hello"), auth("default_user_id", None, allowed_routes=[]))
    assert e.value.status_code == 403


@pytest.mark.parametrize(
    ("routes", "restricted"),
    [
        (["/v1/chat/completions"], True),
        (["llm_api_routes"], True),  # a group still counts as a restriction
        ([], False),
        (None, False),
    ],
)
def test_key_is_route_restricted(routes: Any, restricted: bool) -> None:
    who = UserAPIKeyAuth(user_id="u")
    object.__setattr__(who, "allowed_routes", routes)
    assert key_is_route_restricted(who) is restricted


def test_default_user_api_key_auth_is_unrestricted() -> None:
    """LiteLLM's default for allowed_routes is []: unrestricted."""
    assert key_is_route_restricted(UserAPIKeyAuth(user_id="u")) is False
