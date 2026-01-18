class SecurityConfig:
    LEAK_THRESHOLD = 0.75  # Lowered from 0.80 to catch more paraphrased leaks
    MAX_REWRITE_ATTEMPTS = 3
    RAG_TOP_K = 2
    
    REDACT_ENTITIES = [
        "PERSON", 
        "LOCATION", 
        "EMAIL_ADDRESS", 
        "PHONE_NUMBER", 
        "API_KEY"
    ]
    
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
        "role play"
    ]
    
    RATE_LIMIT_QUERIES_PER_MINUTE = 10
    
    ROLE_PERMISSIONS = {
        "employee": ["public", "internal"],
        "manager": ["public", "internal", "confidential"],
        "executive": ["public", "internal", "confidential", "secret"]
    }
    
    COLLECTION_NAMES = {
        "public": "company_knowledge",
        "internal": "company_knowledge",
        "confidential": "company_knowledge"
    }
