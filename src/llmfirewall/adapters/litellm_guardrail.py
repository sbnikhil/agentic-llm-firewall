"""LiteLLM CustomGuardrail: the engine on every chat completion.

Hooks (verified against litellm==1.108.3, pinned by
tests/unit/test_litellm_contract.py; ADR 0003):
- async_pre_call_hook: resolve the policy from the identity LiteLLM verified
  (UserAPIKeyAuth.user_id / .team_id only), run Engine.check_request, then
  forward the request, forward the redacted messages, or block.
- async_post_call_success_hook: non-streaming responses, Engine.check_response
  on each choice.
- async_post_call_streaming_iterator_hook: streamed responses, through
  engine.stream.StreamWindow (ADR 0004). post_call is not used for streams:
  it runs only after the client has the text.

apply_guardrail is deliberately not defined: in LiteLLM a subclass that
defines it is routed through LiteLLM's "unified guardrail" instead of these
hooks.

should_run_guardrail is overridden to always return True, so nothing in the
request (a "guardrails" list) or in key/team metadata
(disable_global_guardrails, opted_out_global_guardrails) can turn this
guardrail off. Only the proxy config can.

Fail closed:
- a key whose allowed_routes is unrestricted is blocked with
  "key has no allowed_routes" (UNRESTRICTED_KEY). LiteLLM treats None, a
  non-list or an empty list as "every route" (RouteChecks.
  is_virtual_key_allowed_to_call_route), and so does this check. Calls made
  with the master key are therefore blocked too. allowed_routes is read only
  for this check, never to choose a policy;
- any call type other than a chat completion is blocked with
  "unsupported call type";
- any exception in this adapter blocks.

Error bodies and audit events name detectors and entity types, never text.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import sys
from collections.abc import AsyncGenerator, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Final, TypeVar

from fastapi import HTTPException
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import (
    Delta,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)

from llmfirewall.engine import audit
from llmfirewall.engine.audit import (
    AuditEvent,
    AuditFinding,
    AuditSink,
    Direction,
    LoggingSink,
    MissingAuditKey,
    Reason,
    StreamStop,
)
from llmfirewall.engine.decision import Action, Decision
from llmfirewall.engine.engine import Engine, builtin_detectors
from llmfirewall.engine.identity import (
    PolicyResolver,
    VerifiedIdentity,
    load_identity_map,
)
from llmfirewall.engine.policy import Policy, load_policies
from llmfirewall.engine.settings import Settings, get_settings
from llmfirewall.engine.stream import StreamWindow, WindowConfig

T = TypeVar("T")

# LiteLLM's call types for /chat/completions. Everything else is blocked.
CHAT_CALL_TYPES: Final = frozenset({"completion", "acompletion"})

UNSUPPORTED_CALL_TYPE: Final = "unsupported call type"
UNRESTRICTED_KEY: Final = "key has no allowed_routes"
CONTENT_FILTER: Final = "content_filter"

_log = logging.getLogger(__name__)


def _blocked(
    status: int, direction: str, message: str, decision: Decision | None = None
) -> HTTPException:
    """The error the client gets. Detector names, entity types and failure
    kinds only: no matched text, no exception message."""
    detail: dict[str, Any] = {
        "type": "llmfirewall_blocked",
        "direction": direction,
        "message": message,
    }
    if decision is not None:
        detail["detectors"] = sorted(
            {f.detector for f in decision.findings}
            | {f.detector for f in decision.failures}
        )
        detail["entity_types"] = sorted({f.entity_type for f in decision.findings})
        detail["failures"] = sorted({f.kind for f in decision.failures})
    return HTTPException(status_code=status, detail={"error": detail})


def key_is_route_restricted(user_api_key_dict: Any) -> bool:
    """True when the key has a non-empty allowed_routes list: the same test
    LiteLLM applies before it restricts routes (route_checks.py,
    is_virtual_key_allowed_to_call_route, litellm==1.108.3)."""
    routes = getattr(user_api_key_dict, "allowed_routes", None)
    return isinstance(routes, list) and len(routes) > 0


class LLMFirewallGuardrail(CustomGuardrail):
    def __init__(
        self,
        *,
        engine: Engine | None = None,
        resolver: PolicyResolver | None = None,
        settings: Settings | None = None,
        audit_sink: AuditSink | None = None,
        warm_up: bool = True,
        **kwargs: Any,
    ) -> None:
        # kwargs are what LiteLLM's guardrail registry passes: guardrail_name,
        # event_hook, default_on and the other litellm_params.
        super().__init__(**kwargs)
        settings = settings or get_settings()
        if (
            settings.audit_hmac_key is None
            or not settings.audit_hmac_key.get_secret_value()
        ):
            raise MissingAuditKey("LLMFW_AUDIT_HMAC_KEY is not set")
        self._hmac_key = settings.audit_hmac_key.get_secret_value().encode("utf-8")
        self._engine = engine or Engine()
        if resolver is None:
            policies = load_policies(
                settings.policies_dir, detectors=builtin_detectors()
            )
            resolver = PolicyResolver(
                load_identity_map(settings.identity_map_path, policies), policies
            )
        self._resolver = resolver
        self._window = WindowConfig.from_settings(settings)
        self._audit = audit_sink or _stdout_sink()
        # One worker: the engine and its models run off the event loop, one
        # call at a time (spaCy and the sentence model are not guaranteed
        # thread-safe). The latency task may revisit this.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llmfw")
        if warm_up:
            self._warm_up()

    # -- gating ---------------------------------------------------------

    def should_run_guardrail(self, data: Any, event_type: GuardrailEventHooks) -> bool:
        """Always run. LiteLLM's version lets a request's "guardrails" list
        and key/team metadata switch a guardrail off."""
        return True

    # -- helpers ----------------------------------------------------------

    def _warm_up(self) -> None:
        """Load every model and run each one once, so the first request does
        not pay the load time (which would exceed the latency budget and
        fail closed)."""
        default = self._resolver.resolve(VerifiedIdentity(user_id=None, team_id=None))
        self._engine.check_request([{"role": "user", "content": "warm up"}], default)
        self._engine.check_response("warm up", default)

    async def _run(self, fn: Callable[..., T], *args: Any) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    def _identity(self, user_api_key_dict: Any) -> tuple[VerifiedIdentity, str | None]:
        identity = VerifiedIdentity.from_litellm_auth(user_api_key_dict)
        return identity, audit.user_id_hmac(self._hmac_key, identity.user_id)

    def _emit(
        self,
        *,
        direction: Direction,
        reason: Reason,
        action: Action,
        identity: VerifiedIdentity,
        user_hmac: str | None,
        data: Any,
        call_type: str | None,
        policy: Policy | None = None,
        decisions: Sequence[Decision] = (),
        findings: Sequence[AuditFinding] | None = None,
        stream_stop: StreamStop | None = None,
    ) -> None:
        try:
            if findings is None:
                findings = [f for d in decisions for f in audit.findings_of(d)]
            event = audit.build_event(
                direction=direction,
                reason=reason,
                action=action,
                policy_name=policy.name if policy else None,
                team_id=identity.team_id,
                user_hmac=user_hmac,
                call_id=data.get("litellm_call_id") if isinstance(data, dict) else None,
                call_type=call_type,
                findings=findings,
                failures=[f for d in decisions for f in audit.failures_of(d)],
                skipped=[s for d in decisions for s in d.skipped],
                decisions=decisions,
                stream_stop=stream_stop,
            )
            self._audit.emit(event)
        except Exception as e:  # an audit failure must not change the decision
            _log.error("llmfirewall audit event failed: %s", type(e).__name__)

    # -- request ------------------------------------------------------------

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: Any,
    ) -> Exception | str | dict | None:
        identity, user_hmac = self._identity(user_api_key_dict)
        call_type_name = call_type if isinstance(call_type, str) else None
        common: dict[str, Any] = {
            "direction": "request",
            "identity": identity,
            "user_hmac": user_hmac,
            "data": data,
            "call_type": call_type_name,
        }

        if not key_is_route_restricted(user_api_key_dict):
            self._emit(reason="unrestricted_key", action=Action.BLOCK, **common)
            raise _blocked(403, "request", UNRESTRICTED_KEY)

        if call_type_name not in CHAT_CALL_TYPES:
            self._emit(reason="unsupported_call_type", action=Action.BLOCK, **common)
            raise _blocked(400, "request", UNSUPPORTED_CALL_TYPE)

        policy: Policy | None = None
        try:
            policy = self._resolver.resolve(identity)
            messages = data.get("messages")
            if not isinstance(messages, list):
                raise TypeError("messages is not a list")
            decision = await self._run(self._engine.check_request, messages, policy)
        except Exception:
            self._emit(
                reason="internal_error", action=Action.BLOCK, policy=policy, **common
            )
            raise _blocked(500, "request", "request could not be checked") from None

        self._emit(
            reason="decision",
            action=decision.action,
            policy=policy,
            decisions=[decision],
            **common,
        )
        if decision.action is Action.BLOCK:
            raise _blocked(400, "request", "request blocked", decision)
        if decision.action is Action.REDACT and decision.redacted_messages is not None:
            data["messages"] = [dict(m) for m in decision.redacted_messages]
        return data

    # -- non-streaming response -----------------------------------------------

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: Any,
        response: Any,
    ) -> Any:
        identity, user_hmac = self._identity(user_api_key_dict)
        common: dict[str, Any] = {
            "direction": "response",
            "identity": identity,
            "user_hmac": user_hmac,
            "data": data,
            "call_type": None,
        }
        if not isinstance(response, ModelResponse):
            self._emit(reason="unsupported_response", action=Action.BLOCK, **common)
            raise _blocked(500, "response", "response could not be checked")

        policy: Policy | None = None
        decisions: list[tuple[int, Decision]] = []
        try:
            policy = self._resolver.resolve(identity)
            for position, choice in enumerate(response.choices):
                message = getattr(choice, "message", None)
                content = getattr(message, "content", None)
                if content is None:
                    continue  # e.g. only tool_calls: not checked (v2)
                if not isinstance(content, str):
                    raise TypeError("response content is not text")
                index = getattr(choice, "index", position)
                decision = await self._run(self._engine.check_response, content, policy)
                decisions.append((index, decision))
        except Exception:
            self._emit(
                reason="internal_error", action=Action.BLOCK, policy=policy, **common
            )
            raise _blocked(500, "response", "response could not be checked") from None

        action = max((d.action for _, d in decisions), default=Action.ALLOW)
        findings = [
            f.model_copy(update={"message_index": index})
            for index, d in decisions
            for f in audit.findings_of(d)
        ]
        self._emit(
            reason="decision",
            action=action,
            policy=policy,
            decisions=[d for _, d in decisions],
            findings=findings,
            **common,
        )
        if action is Action.BLOCK:
            blocking = next(d for _, d in decisions if d.action is Action.BLOCK)
            raise _blocked(400, "response", "response blocked", blocking)
        if action is Action.REDACT:
            response = response.model_copy(deep=True)
            redacted = {
                i: d.redacted_text for i, d in decisions if d.redacted_text is not None
            }
            for position, choice in enumerate(response.choices):
                index = getattr(choice, "index", position)
                if index in redacted:
                    choice.message.content = redacted[index]  # type: ignore[union-attr]
        return response

    # -- streaming response ---------------------------------------------------

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: Any,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        identity, user_hmac = self._identity(user_api_key_dict)
        windows: dict[int, StreamWindow] = {}
        policy: Policy | None = None
        stop: StreamStop | None = None
        reason: Reason = "decision"
        last: ModelResponseStream | None = None

        def window(index: int) -> StreamWindow:
            if index not in windows:
                if policy is None:
                    raise RuntimeError("no policy")
                windows[index] = StreamWindow(self._engine, policy, self._window)
            return windows[index]

        def stopped(index: int, why: Reason) -> ModelResponseStream:
            nonlocal stop, reason
            w = windows.get(index)
            stop = StreamStop(
                reason="content_filter",
                choice_index=index,
                released_chars=w.released_chars if w else 0,
                withheld_chars=w.withheld_chars if w else 0,
            )
            reason = why
            return _final_chunk(last, index)

        try:
            try:
                policy = self._resolver.resolve(identity)
            except Exception:
                yield stopped(0, "internal_error")
                return

            async for chunk in response:
                if not isinstance(chunk, ModelResponseStream):
                    yield stopped(0, "unsupported_response")
                    return
                last = chunk
                out = chunk.model_copy(deep=True)
                keep = False
                for position, choice in enumerate(out.choices):
                    index = getattr(choice, "index", position)
                    delta = choice.delta
                    content = getattr(delta, "content", None)
                    released = ""
                    try:
                        if content is not None and not isinstance(content, str):
                            raise TypeError("delta content is not text")
                        w = window(index)
                        if content:
                            result = await self._run(w.feed, content)
                            if result.stopped:
                                yield stopped(index, "decision")
                                return
                            released = result.text
                        if choice.finish_reason is not None:
                            result = await self._run(w.finish)
                            if result.stopped:
                                yield stopped(index, "decision")
                                return
                            released += result.text
                    except Exception:
                        yield stopped(index, "internal_error")
                        return
                    if content is not None or released:
                        delta.content = released
                    if (
                        released
                        or choice.finish_reason is not None
                        or getattr(delta, "role", None)
                        or getattr(delta, "tool_calls", None)
                        or getattr(delta, "function_call", None)
                    ):
                        keep = True
                if keep or not out.choices:
                    yield out

            # The upstream ended without a finish_reason on some choice:
            # check and release what is still held.
            for index, w in windows.items():
                if w.finished or w.stopped:
                    continue
                try:
                    result = await self._run(w.finish)
                except Exception:
                    yield stopped(index, "internal_error")
                    return
                if result.stopped:
                    yield stopped(index, "decision")
                    return
                if result.text:
                    yield _text_chunk(last, index, result.text)
        finally:
            if stop is not None:
                await _close(response)
            decisions = [d for w in windows.values() for d in w.decisions]
            findings = [
                f.model_copy(update={"message_index": i})
                for i, w in windows.items()
                for f in w.findings
            ]
            action = (
                Action.BLOCK
                if stop is not None
                else max((d.action for d in decisions), default=Action.ALLOW)
            )
            self._emit(
                direction="stream",
                reason=reason,
                action=action,
                identity=identity,
                user_hmac=user_hmac,
                data=request_data,
                call_type=None,
                policy=policy,
                decisions=decisions,
                findings=findings,
                stream_stop=stop,
            )


async def _close(response: Any) -> None:
    """Stop reading from the upstream after a block."""
    aclose: Any = getattr(response, "aclose", None)
    if callable(aclose):
        try:
            result = aclose()
            if inspect.isawaitable(result):
                await result
        except Exception as e:
            _log.debug("closing the upstream stream failed: %s", type(e).__name__)


def _final_chunk(
    template: ModelResponseStream | None, index: int
) -> ModelResponseStream:
    """The chunk that ends a blocked stream: no content, content_filter."""
    return _chunk(template, index, Delta(content=None), CONTENT_FILTER)


def _text_chunk(
    template: ModelResponseStream | None, index: int, text: str
) -> ModelResponseStream:
    return _chunk(template, index, Delta(content=text), None)


def _chunk(
    template: ModelResponseStream | None,
    index: int,
    delta: Delta,
    finish_reason: str | None,
) -> ModelResponseStream:
    kwargs: dict[str, Any] = {}
    if template is not None:
        kwargs = {
            "id": template.id,
            "model": template.model,
            "created": template.created,
        }
    return ModelResponseStream(
        choices=[
            StreamingChoices(index=index, delta=delta, finish_reason=finish_reason)
        ],
        **kwargs,
    )


def _stdout_sink() -> AuditSink:
    """Audit events as JSON lines on stdout, unless the operator already
    configured the llmfirewall.audit logger."""
    logger = logging.getLogger(audit.AUDIT_LOGGER)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return LoggingSink(logger)


__all__ = [
    "CHAT_CALL_TYPES",
    "UNRESTRICTED_KEY",
    "UNSUPPORTED_CALL_TYPE",
    "AuditEvent",
    "LLMFirewallGuardrail",
    "key_is_route_restricted",
]
