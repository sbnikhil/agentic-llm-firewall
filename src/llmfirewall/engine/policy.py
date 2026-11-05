"""Policy schema, YAML loading, and merging two policies into the stricter.

Design: docs/design/policy.md section 2. A policy is validated against the
detectors it will run with: every registered detector must be listed, and
each detector's params are checked by its own params model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llmfirewall.engine.decision import Action

if TYPE_CHECKING:
    from llmfirewall.engine.pipeline import Detector

IDENTITY_FILE = "identity.yaml"


class PolicyError(ValueError):
    """A policy or identity map is invalid. Raised at load time, so a bad
    file stops the service from starting instead of being skipped."""


class Role(StrEnum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"
    RESPONSE = "response"  # the model's answer, in check_response only


REQUEST_ROLES = frozenset(Role) - {Role.RESPONSE}


class FailureMode(StrEnum):
    FAIL_CLOSED = "fail_closed"
    FAIL_OPEN = "fail_open"


class DetectorParams(BaseModel):
    """Base for a detector's params model.

    stricter() merges two values into the stricter one. The default only
    accepts equal values; a model with fields overrides it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def stricter(self, other: Self) -> Self:
        if self == other:
            return self
        raise PolicyError(
            f"{type(self).__name__} has no rule for merging different values"
        )


class NoParams(DetectorParams):
    pass


# The YAML shape, validated strictly (no coercion: "true" is not a bool).

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)

_RoleName = Literal[
    "system", "developer", "user", "assistant", "tool", "function", "response"
]
_ModeName = Literal["fail_closed", "fail_open"]


class _RawDetectorConfig(BaseModel):
    model_config = _STRICT

    enabled: bool
    action: Literal["allow", "redact", "block"]
    apply_to: list[_RoleName]
    params: dict[str, Any] = Field(default_factory=dict)


class _RawPolicy(BaseModel):
    model_config = _STRICT

    version: Literal[1]
    on_detector_error: _ModeName = "fail_closed"
    on_timeout: _ModeName = "fail_closed"
    latency_budget_ms: int = Field(gt=0)
    detectors: dict[str, _RawDetectorConfig]


# The parsed policy.


class DetectorConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool
    action: Action
    apply_to: frozenset[Role]
    params: DetectorParams


class Policy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: Literal[1] = 1
    on_detector_error: FailureMode
    on_timeout: FailureMode
    latency_budget_ms: int
    detectors: dict[str, DetectorConfig]


def _registry(detectors: Sequence[Detector]) -> dict[str, Detector]:
    registry: dict[str, Detector] = {}
    for detector in detectors:
        if detector.name in registry:
            raise PolicyError(f"two detectors are named {detector.name!r}")
        registry[detector.name] = detector
    return registry


def parse_policy(
    data: Mapping[str, Any], *, name: str, detectors: Sequence[Detector]
) -> Policy:
    """Validate policy data (already loaded from YAML) against `detectors`."""
    registry = _registry(detectors)
    if not isinstance(data, Mapping):
        raise PolicyError(f"policy {name!r}: expected a mapping at the top level")
    try:
        raw = _RawPolicy.model_validate(dict(data))
    except ValidationError as e:
        raise PolicyError(f"policy {name!r} is invalid: {e}") from e

    unknown = sorted(set(raw.detectors) - set(registry))
    missing = sorted(set(registry) - set(raw.detectors))
    if unknown:
        raise PolicyError(f"policy {name!r}: unknown detectors {unknown}")
    if missing:
        raise PolicyError(
            f"policy {name!r}: detectors {missing} are not listed; every "
            "detector must be listed, with enabled: false to turn it off"
        )

    configs: dict[str, DetectorConfig] = {}
    for det_name, raw_config in raw.detectors.items():
        where = f"policy {name!r}: detectors.{det_name}"
        if raw_config.enabled and not raw_config.apply_to:
            raise PolicyError(f"{where}.apply_to must not be empty when enabled")
        params_model = registry[det_name].params_model
        if not issubclass(params_model, DetectorParams):
            raise PolicyError(f"{where}: params model must be a DetectorParams")
        try:
            params = params_model.model_validate(raw_config.params)
        except ValidationError as e:
            raise PolicyError(f"{where}.params is invalid: {e}") from e
        configs[det_name] = DetectorConfig(
            enabled=raw_config.enabled,
            action=Action[raw_config.action.upper()],
            apply_to=frozenset(Role(r) for r in raw_config.apply_to),
            params=params,
        )

    return Policy(
        name=name,
        on_detector_error=FailureMode(raw.on_detector_error),
        on_timeout=FailureMode(raw.on_timeout),
        latency_budget_ms=raw.latency_budget_ms,
        detectors=configs,
    )


# YAML loading.


def _reject_duplicate_keys(node: yaml.Node | None, source: str) -> None:
    """yaml.safe_load keeps the last of two equal keys silently. Walk the
    composed node tree (no objects are constructed) and refuse duplicates."""
    if isinstance(node, yaml.MappingNode):
        seen: set[tuple[str, str]] = set()
        for key, value in node.value:
            if isinstance(key, yaml.ScalarNode):
                ident = (key.tag, key.value)
                if ident in seen:
                    raise PolicyError(
                        f"{source}: duplicate key {key.value!r} "
                        f"(line {key.start_mark.line + 1})"
                    )
                seen.add(ident)
            _reject_duplicate_keys(value, source)
    elif isinstance(node, yaml.SequenceNode):
        for item in node.value:
            _reject_duplicate_keys(item, source)


def load_yaml(path: Path) -> Any:
    """Read a YAML file with yaml.safe_load, refusing duplicate keys."""
    try:
        text = path.read_text()
        _reject_duplicate_keys(yaml.compose(text, Loader=yaml.SafeLoader), path.name)
        return yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise PolicyError(f"{path.name}: not valid YAML: {e}") from e
    except OSError as e:
        raise PolicyError(f"{path.name}: cannot be read: {e}") from e


def load_policy_file(path: Path, *, detectors: Sequence[Detector]) -> Policy:
    """Load one policy; its name is the file name without .yaml."""
    data = load_yaml(path)
    if not isinstance(data, Mapping):
        raise PolicyError(f"{path.name}: expected a mapping at the top level")
    return parse_policy(data, name=path.stem, detectors=detectors)


def load_policies(
    directory: Path, *, detectors: Sequence[Detector]
) -> dict[str, Policy]:
    """Every *.yaml in `directory` except identity.yaml. Any invalid file
    fails the whole load."""
    return {
        path.stem: load_policy_file(path, detectors=detectors)
        for path in sorted(directory.glob("*.yaml"))
        if path.name != IDENTITY_FILE
    }


# Merging.


def _stricter_mode(a: FailureMode, b: FailureMode) -> FailureMode:
    if FailureMode.FAIL_CLOSED in (a, b):
        return FailureMode.FAIL_CLOSED
    return FailureMode.FAIL_OPEN


def _merge_detector(a: DetectorConfig, b: DetectorConfig) -> DetectorConfig:
    enabled = [c for c in (a, b) if c.enabled]
    if not enabled:
        # Runs on neither side: disabled, with neutral values that have no
        # effect (and do not depend on which side came first).
        return DetectorConfig(
            enabled=False,
            action=Action.ALLOW,
            apply_to=frozenset(),
            params=type(a.params)(),
        )
    if len(enabled) == 1:
        return enabled[0]
    return DetectorConfig(
        enabled=True,
        action=max(a.action, b.action),
        apply_to=a.apply_to | b.apply_to,
        params=a.params.stricter(b.params),
    )


def merge_stricter(a: Policy, b: Policy) -> Policy:
    """One policy whose effective behavior is at least as strict as both.

    Per detector, only the sides where it is enabled count: a disabled
    detector's action, apply_to and params have no effect, so they are not
    merged. Enabled on both sides: the stricter action, the union of
    apply_to, and params merged by the params model's stricter(). Enabled on
    one side: that side's config. Enabled on neither: disabled.
    Fail closed if either fails closed; the lower latency budget.
    """
    if a == b:
        return a
    if set(a.detectors) != set(b.detectors):
        raise PolicyError(
            f"cannot merge {a.name!r} and {b.name!r}: different detectors"
        )
    configs: dict[str, DetectorConfig] = {}
    for det_name, ca in a.detectors.items():
        cb = b.detectors[det_name]
        if type(ca.params) is not type(cb.params):
            raise PolicyError(
                f"cannot merge {a.name!r} and {b.name!r}: detector "
                f"{det_name!r} has different params models"
            )
        configs[det_name] = _merge_detector(ca, cb)
    return Policy(
        name=f"{a.name}+{b.name}",
        on_detector_error=_stricter_mode(a.on_detector_error, b.on_detector_error),
        on_timeout=_stricter_mode(a.on_timeout, b.on_timeout),
        latency_budget_ms=min(a.latency_budget_ms, b.latency_budget_ms),
        detectors=configs,
    )
