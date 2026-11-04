"""PII redaction with Presidio.

Moved unchanged from v0 app/redactor.py. The recognizers are built on demand
by custom_recognizers(), and the analyzer and anonymizer come from
llmfirewall.engine.models, so importing this module loads nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmfirewall.engine import models

if TYPE_CHECKING:
    from presidio_analyzer import PatternRecognizer

REDACT_ENTITIES = [
    "PERSON",
    "LOCATION",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "API_KEY",
    "US_SSN",
    "CREDIT_CARD",
    "IP_ADDRESS",
]


def custom_recognizers() -> list[PatternRecognizer]:
    """The recognizers v0 added to Presidio's defaults, in the same order."""
    from presidio_analyzer import Pattern, PatternRecognizer

    local_phone_pattern = Pattern(
        name="local_phone_pattern",
        regex=r"(\b\d{3}[-.\s]\d{4}\b)",
        score=0.5,  # Lower score because 7 digits can sometimes be other data
    )
    local_phone_recognizer = PatternRecognizer(
        supported_entity="PHONE_NUMBER",
        patterns=[local_phone_pattern],
        context=["phone", "call", "cell", "contact", "mobile"],
    )

    api_key_pattern = Pattern(
        name="api_key_pattern", regex=r"(sk-[a-zA-Z0-9]{15,})", score=1.0
    )
    api_key_recognizer = PatternRecognizer(
        supported_entity="API_KEY", patterns=[api_key_pattern]
    )

    ssn_pattern = Pattern(
        name="ssn_pattern", regex=r"(\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b)", score=1.0
    )
    ssn_recognizer = PatternRecognizer(
        supported_entity="US_SSN", patterns=[ssn_pattern]
    )

    credit_card_pattern = Pattern(
        name="credit_card_pattern",
        regex=r"(\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b)",
        score=0.9,
    )
    credit_card_recognizer = PatternRecognizer(
        supported_entity="CREDIT_CARD", patterns=[credit_card_pattern]
    )

    ip_address_pattern = Pattern(
        name="ip_address_pattern", regex=r"(\b(?:\d{1,3}\.){3}\d{1,3}\b)", score=0.8
    )
    ip_address_recognizer = PatternRecognizer(
        supported_entity="IP_ADDRESS",
        patterns=[ip_address_pattern],
        context=["server", "network", "ip", "address", "located at"],
    )

    return [
        local_phone_recognizer,
        api_key_recognizer,
        ssn_recognizer,
        credit_card_recognizer,
        ip_address_recognizer,
    ]


def redact_pii(text: str) -> str:
    """
    Analyzes text for PII (Names, Locations, Emails, etc.)
    and returns a redacted version.
    """

    analysis_results = models.presidio_analyzer().analyze(
        text=text,
        language="en",
        entities=REDACT_ENTITIES,
    )

    anonymized_result = models.presidio_anonymizer().anonymize(
        text=text, analyzer_results=analysis_results
    )

    return anonymized_result.text
