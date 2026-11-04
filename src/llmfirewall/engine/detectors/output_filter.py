"""Output leak filter: regex patterns, keywords, similarity to a secrets catalog.

Moved unchanged from v0 app/security/output_filter.py. Changes: the model is
the shared one from llmfirewall.engine.models, the catalog embeddings are
computed once per process instead of on every call, and the thresholds come
from settings.
"""

import re

import numpy as np

from llmfirewall.engine import models
from llmfirewall.engine.settings import get_settings

SECRET_PATTERNS = {
    "numeric_code": r"(?:passcode|password|code|pin)[\s:is]*\d{4,6}\b",
    "api_key": r"\b(sk-[a-zA-Z0-9]{20,}|[A-Z0-9]{32,})\b",
    "password": r"password\s*[:=]\s*\S+",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b",
    "ssn": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
    "salary": r"\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?",
    "budget": r"budget.*\$\d+",
}

SENSITIVE_KEYWORDS = [
    "salary",
    "compensation",
    "budget",
    "revenue",
    "profit",
    "passcode",
    "password",
    "credential",
    "api key",
    "secret",
    "confidential",
    "restricted",
    "internal only",
]


def check_keyword_presence(response: str) -> tuple[bool, str]:
    """Check for sensitive keywords that shouldn't appear in responses"""
    response_lower = response.lower()
    for keyword in SENSITIVE_KEYWORDS:
        if keyword in response_lower:
            return True, f"Sensitive keyword: {keyword}"
    return False, ""


def check_pattern_match(response: str) -> tuple[bool, str]:
    for pattern_name, pattern in SECRET_PATTERNS.items():
        if re.search(pattern, response, re.IGNORECASE):
            return True, f"Pattern detected: {pattern_name}"
    return False, ""


def check_semantic_similarity(response: str) -> tuple[bool, str, float]:
    secrets_catalog = models.secrets_catalog_embeddings()

    if not secrets_catalog:
        return False, "", 0.0

    response_embedding = models.sentence_transformer().encode(response)

    for secret_entry, secret_embedding in secrets_catalog:
        similarity = np.dot(response_embedding, secret_embedding) / (
            np.linalg.norm(response_embedding) * np.linalg.norm(secret_embedding)
        )

        if similarity > get_settings().leak_threshold:
            return True, secret_entry.get("type", "unknown"), float(similarity)

    return False, "", 0.0


def validate_output(response: str) -> tuple[bool, str, float]:
    """
    Ensemble approach to leak detection:
    1. Pattern matching (regex)
    2. Keyword presence
    3. Semantic similarity
    """
    # Layer 1: Pattern matching
    is_pattern_leak, pattern_type = check_pattern_match(response)
    if is_pattern_leak:
        return False, f"Blocked: {pattern_type}", 1.0

    # Layer 2: Keyword detection
    has_sensitive_keyword, keyword = check_keyword_presence(response)

    # Layer 3: Semantic similarity
    is_semantic_leak, leak_type, similarity = check_semantic_similarity(response)

    # Decision logic: Block if semantic leak OR (keyword + high similarity)
    if is_semantic_leak:
        return False, f"Blocked: Similar to {leak_type}", similarity

    if (
        has_sensitive_keyword
        and similarity > get_settings().keyword_similarity_threshold
    ):
        return False, f"Blocked: {keyword} with similarity {similarity:.2f}", similarity

    return True, "Passed", 0.0
