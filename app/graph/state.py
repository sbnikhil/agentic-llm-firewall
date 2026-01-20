from typing import TypedDict

class AgentState(TypedDict):
    user_id: str
    user_role: str
    user_input: str
    redacted_input: str
    topic_level: str
    blocked: bool
    block_reason: str
    rag_context: str
    llm_response: str
    passed_output_security: bool
    leak_score: float
    loop_count: int