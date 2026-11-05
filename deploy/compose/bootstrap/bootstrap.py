"""Create the demo teams, users and keys through LiteLLM's own management API.

Runs once at `docker compose up`, after the proxy is healthy. Idempotent: a
team or user that already exists is left as it is; an existing key gets its
allowed_routes set again.

    team-standard  -> policy "default"  (mapped in policies/identity.yaml)
    team-strict    -> policy "strict"
    alice          -> member of team-standard, key LLMFW_DEV_ALICE_KEY
    bob            -> member of team-strict,   key LLMFW_DEV_BOB_KEY

Every key is created with allowed_routes = CHAT_ROUTES, so LiteLLM rejects
it on every other route (/v1/messages, /v1/responses, provider pass-through
routes, ...). The guardrail blocks any key whose allowed_routes is
unrestricted. Each run checks that the stored keys carry exactly CHAT_ROUTES.

LiteLLM matches an explicit allowed_routes entry by exact path or path
prefix (RouteChecks._route_matches_allowed_route, litellm==1.108.3); a
"{model}" placeholder is not expanded for explicit entries. So the two
deployment-style aliases are listed with this stack's model name, "chat".

The policy mapping lives only in the server-side identity map; nothing here
(or in LiteLLM's team/user metadata) chooses a policy.

Stdlib only, so it runs with the proxy image's Python. Never prints a key.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("LITELLM_URL", "http://litellm:4000").rstrip("/")
MASTER_KEY = os.environ["LITELLM_MASTER_KEY"]

TEAMS = [
    {"team_id": "team-standard", "team_alias": "standard"},
    {"team_id": "team-strict", "team_alias": "strict"},
]
USERS = [
    {"user_id": "alice", "team_id": "team-standard", "key_env": "LLMFW_DEV_ALICE_KEY"},
    {"user_id": "bob", "team_id": "team-strict", "key_env": "LLMFW_DEV_BOB_KEY"},
]

MODEL = "chat"
CHAT_ROUTES = [
    "/v1/chat/completions",
    "/chat/completions",
    f"/engines/{MODEL}/chat/completions",
    f"/openai/deployments/{MODEL}/chat/completions",
]


def call(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    request = urllib.request.Request(  # noqa: S310 (fixed http URL in compose)
        f"{BASE}{path}",
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {MASTER_KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read() or b"{}")
        except ValueError:
            payload = {}
        return e.code, payload


def wait_ready(timeout_s: float = 300) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            status, _ = call("GET", "/health/readiness")
            if status == 200:
                return
        except OSError:
            pass
        time.sleep(2)
    sys.exit("proxy did not become ready")


def ensure_team(team: dict) -> None:
    status, _ = call("GET", f"/team/info?team_id={team['team_id']}")
    if status == 200:
        print(f"team {team['team_id']}: exists")
        return
    status, payload = call("POST", "/team/new", team)
    if status != 200:
        sys.exit(f"team {team['team_id']}: /team/new returned {status}: {payload}")
    print(f"team {team['team_id']}: created")


def ensure_user(user: dict) -> None:
    status, payload = call("GET", f"/user/info?user_id={user['user_id']}")
    if status == 200 and payload.get("user_info"):
        print(f"user {user['user_id']}: exists")
    else:
        status, payload = call("POST", "/user/new", {"user_id": user["user_id"]})
        if status != 200:
            sys.exit(f"user {user['user_id']}: /user/new returned {status}: {payload}")
        print(f"user {user['user_id']}: created")
    status, payload = call(
        "POST",
        "/team/member_add",
        {
            "team_id": user["team_id"],
            "member": {"role": "user", "user_id": user["user_id"]},
        },
    )
    if status == 200:
        print(f"user {user['user_id']}: added to {user['team_id']}")
    elif "already" in json.dumps(payload).lower():
        print(f"user {user['user_id']}: already in {user['team_id']}")
    else:
        sys.exit(
            f"user {user['user_id']}: /team/member_add returned {status}: {payload}"
        )


def ensure_key(user: dict) -> None:
    key = os.environ[user["key_env"]]
    if not key.startswith("sk-"):
        sys.exit(f"{user['key_env']} must start with sk-")
    status, _ = call("GET", f"/key/info?key={key}")
    if status == 200:
        status, payload = call(
            "POST", "/key/update", {"key": key, "allowed_routes": CHAT_ROUTES}
        )
        if status != 200:
            sys.exit(f"key for {user['user_id']}: /key/update returned {status}")
        print(f"key for {user['user_id']}: exists, allowed_routes set")
    else:
        status, payload = call(
            "POST",
            "/key/generate",
            {
                "key": key,
                "user_id": user["user_id"],
                "team_id": user["team_id"],
                "key_alias": f"{user['user_id']}-dev",
                "allowed_routes": CHAT_ROUTES,
            },
        )
        if status != 200:
            sys.exit(f"key for {user['user_id']}: /key/generate returned {status}")
        print(f"key for {user['user_id']}: created")
    # Check what LiteLLM stored, not what was sent.
    status, payload = call("GET", f"/key/info?key={key}")
    info = payload.get("info", {}) if status == 200 else {}
    if (
        info.get("allowed_routes") != CHAT_ROUTES
        or info.get("user_id") != user["user_id"]
        or info.get("team_id") != user["team_id"]
    ):
        sys.exit(f"key for {user['user_id']}: stored key does not match")


def main() -> None:
    wait_ready()
    for team in TEAMS:
        ensure_team(team)
    for user in USERS:
        ensure_user(user)
        ensure_key(user)
    print("bootstrap done")


if __name__ == "__main__":
    main()
