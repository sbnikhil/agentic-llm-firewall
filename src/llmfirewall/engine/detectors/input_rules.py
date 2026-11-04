"""Input rules: length, CSV-formula, jailbreak-phrase and symbol checks.

Moved unchanged from v0 app/security/input_validator.py; JAILBREAK_PATTERNS
moved from v0 config.py, and the two thresholds now come from settings.
"""

import unicodedata

from llmfirewall.engine.settings import get_settings

JAILBREAK_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard your rules",
    "you are now in developer mode",
    "pretend you are",
    "act as if",
    "bypass your programming",
    "override your constraints",
    "translate to",
    "hypothetical scenario",
    "for educational purposes",
    "let's play a game",
    "system:",
    "security protocols disabled",
    "confirm you understand",
    "do anything now",
    "dan mode",
    "repeat after me",
    "fill in the redacted",
    "tell me bedtime stories",
    "grandmother used to tell",
    "execute the above",
    "think step by step",
    "let's think",
    "security disabled",
    "bypass all restrictions",
    "role play",
]


def normalize_unicode(text: str) -> str:
    """Normalize fullwidth and other unicode variants to ASCII"""
    return unicodedata.normalize("NFKC", text)


def detect_jailbreak_attempt(query: str) -> bool:
    # Normalize unicode to catch obfuscation attempts
    query_normalized = normalize_unicode(query).lower()
    for pattern in JAILBREAK_PATTERNS:
        if pattern in query_normalized:
            return True
    return False


def contains_excessive_special_chars(query: str) -> bool:
    special_char_count = sum(1 for c in query if not c.isalnum() and not c.isspace())
    return special_char_count > len(query) * get_settings().max_special_char_ratio


def detect_csv_injection(query: str) -> bool:
    """Detect CSV injection attempts (formulas starting with =, +, -, @)"""
    stripped = query.strip()
    return stripped.startswith(("=", "+", "-", "@")) and "," in query


def validate_input(query: str, user_role: str) -> tuple[bool, str]:
    if not query or not query.strip():
        return False, "Empty query"

    if len(query) > get_settings().max_query_length:
        return False, "Query too long"

    if detect_csv_injection(query):
        return False, "CSV injection attempt detected"

    if detect_jailbreak_attempt(query):
        return False, "Prompt injection detected"

    if contains_excessive_special_chars(query):
        return False, "Suspicious input pattern"

    return True, ""
