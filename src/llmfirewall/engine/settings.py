"""Engine settings: model names and thresholds.

Defaults are the values hardcoded in the v0 app. Each can be overridden with
an environment variable prefixed LLMFW_, e.g. LLMFW_LEAK_THRESHOLD=0.8.
Settings are read on first call to get_settings(), not at import.
"""

from functools import cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/llmfirewall/engine/settings.py -> repository root
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLMFW_", frozen=True)

    # Models
    sentence_model: str = "all-MiniLM-L6-v2"
    spacy_model: str = "en_core_web_lg"

    # Input rules
    max_query_length: int = 2000
    max_special_char_ratio: float = 0.3

    # Output filter
    leak_threshold: float = 0.75
    keyword_similarity_threshold: float = 0.65
    secrets_catalog_path: Path = REPO_ROOT / "secrets_catalog.json"

    # Policies (docs/design/policy.md)
    policies_dir: Path = REPO_ROOT / "policies"
    identity_map_path: Path = REPO_ROOT / "policies" / "identity.yaml"

    # Streaming window (docs/decisions/0004-streaming.md). The defaults are
    # not final: the final hold-back comes from the latency task.
    stream_holdback_chars: int = Field(default=128, ge=0)  # N
    stream_context_chars: int = Field(default=128, ge=0)  # C
    stream_scan_step_chars: int = Field(default=32, ge=1)  # S

    # Audit (engine/audit.py): key for the HMAC-SHA256 of the user id.
    # SecretStr so the value never appears in a repr or a log line.
    audit_hmac_key: SecretStr | None = None


@cache
def get_settings() -> Settings:
    return Settings()
