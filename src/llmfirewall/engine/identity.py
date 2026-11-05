"""Verified LiteLLM identity -> policy, from a server-side mapping file only.

Design: docs/design/policy.md section 3. The policy is never taken from
request content, request metadata or headers: resolution takes only a
VerifiedIdentity, built from the two fields LiteLLM sets after checking the
caller's key (UserAPIKeyAuth.user_id and .team_id).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from llmfirewall.engine.policy import Policy, PolicyError, load_yaml, merge_stricter


class VerifiedIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    user_id: str | None
    team_id: str | None

    @classmethod
    def from_litellm_auth(cls, auth: Any) -> VerifiedIdentity:
        """Read user_id and team_id from LiteLLM's UserAPIKeyAuth, nothing
        else. Missing, empty or non-string ids count as no id."""

        def verified(field: str) -> str | None:
            value = getattr(auth, field, None)
            return value if isinstance(value, str) and value else None

        return cls(user_id=verified("user_id"), team_id=verified("team_id"))


class IdentityMap(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    version: Literal[1]
    default: str = Field(min_length=1)
    users: dict[str, str] = Field(default_factory=dict)
    teams: dict[str, str] = Field(default_factory=dict)

    @field_validator("users", "teams")
    @classmethod
    def _non_empty_ids(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not key for key in value):
            raise ValueError("ids must be non-empty strings")
        return value

    def policy_names(self) -> set[str]:
        return {self.default, *self.users.values(), *self.teams.values()}


def parse_identity_map(
    data: Mapping[str, Any], policies: Mapping[str, Policy]
) -> IdentityMap:
    if not isinstance(data, Mapping):
        raise PolicyError("identity map: expected a mapping at the top level")
    try:
        identity_map = IdentityMap.model_validate(dict(data))
    except ValidationError as e:
        raise PolicyError(f"identity map is invalid: {e}") from e
    _check_names(identity_map, policies)
    return identity_map


def load_identity_map(path: Path, policies: Mapping[str, Policy]) -> IdentityMap:
    return parse_identity_map(load_yaml(path), policies)


def _check_names(identity_map: IdentityMap, policies: Mapping[str, Policy]) -> None:
    unknown = sorted(identity_map.policy_names() - set(policies))
    if unknown:
        raise PolicyError(f"identity map names unknown policies {unknown}")


class PolicyResolver:
    """Order: if both the user and the team are mapped, the stricter merge of
    the two policies (so neither mapping can weaken the other); otherwise the
    one that is mapped; otherwise the default."""

    def __init__(
        self, identity_map: IdentityMap, policies: Mapping[str, Policy]
    ) -> None:
        _check_names(identity_map, policies)
        self._map = identity_map
        self._policies = dict(policies)
        # Every user/team combination is merged now, so a merge problem
        # surfaces at startup rather than on a request.
        self._merged: dict[tuple[str, str], Policy] = {
            (u, t): merge_stricter(self._policies[u], self._policies[t])
            for u in set(identity_map.users.values())
            for t in set(identity_map.teams.values())
        }

    def resolve(self, identity: VerifiedIdentity) -> Policy:
        if not isinstance(identity, VerifiedIdentity):
            raise TypeError("resolve() takes a VerifiedIdentity")
        user_policy = self._map.users.get(identity.user_id or "")
        team_policy = self._map.teams.get(identity.team_id or "")
        if user_policy and team_policy:
            return self._merged[(user_policy, team_policy)]
        name = user_policy or team_policy or self._map.default
        return self._policies[name]
