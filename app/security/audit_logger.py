import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path(__file__).parent.parent.parent / "security_audit.log"

def log_security_event(event_data: dict):
    event_data["timestamp"] = datetime.now().isoformat()
    
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(event_data) + "\n")

def log_interaction(user_id: str, user_role: str, query: str, redacted_query: str, 
                   topic_level: str, response: str, blocked: bool, block_reason: str = None):
    event = {
        "user_id": user_id,
        "user_role": user_role,
        "query": query,
        "redacted_query": redacted_query,
        "topic_level": topic_level,
        "response": response if not blocked else "[BLOCKED]",
        "blocked": blocked,
        "block_reason": block_reason
    }
    log_security_event(event)
