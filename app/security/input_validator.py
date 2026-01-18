import re
import unicodedata
from config import SecurityConfig

def normalize_unicode(text: str) -> str:
    """Normalize fullwidth and other unicode variants to ASCII"""
    return unicodedata.normalize('NFKC', text)

def detect_jailbreak_attempt(query: str) -> bool:
    # Normalize unicode to catch obfuscation attempts
    query_normalized = normalize_unicode(query).lower()
    for pattern in SecurityConfig.JAILBREAK_PATTERNS:
        if pattern in query_normalized:
            return True
    return False

def contains_excessive_special_chars(query: str) -> bool:
    special_char_count = sum(1 for c in query if not c.isalnum() and not c.isspace())
    return special_char_count > len(query) * 0.3

def detect_csv_injection(query: str) -> bool:
    """Detect CSV injection attempts (formulas starting with =, +, -, @)"""
    stripped = query.strip()
    return stripped.startswith(('=', '+', '-', '@')) and ',' in query

def validate_input(query: str, user_role: str) -> tuple[bool, str]:
    if not query or not query.strip():
        return False, "Empty query"
    
    if len(query) > 2000:
        return False, "Query too long"
    
    if detect_csv_injection(query):
        return False, "CSV injection attempt detected"
    
    if detect_jailbreak_attempt(query):
        return False, "Prompt injection detected"
    
    if contains_excessive_special_chars(query):
        return False, "Suspicious input pattern"
    
    return True, ""
