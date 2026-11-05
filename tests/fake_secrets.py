"""Synthetic secret-shaped values for the tests, in one place.

None of these is a real credential: each is a made-up string in the `sk-...`
shape the detectors match. Each value sits on its own line, marked for
gitleaks, so the scanner keeps checking the rest of the repository.
"""

# Synthetic test value, not a real credential (test_audit.py).
AUDIT_API_KEY = "sk-ABCDEF0123456789"  # gitleaks:allow

# Synthetic test value, not a real credential (test_stream_window.py).
STREAM_API_KEY = "sk-ABCDEFGH12345678"  # gitleaks:allow

# Synthetic test value, not a real credential (test_litellm_guardrail.py).
GUARDRAIL_API_KEY = "sk-SECRET123456"  # gitleaks:allow
