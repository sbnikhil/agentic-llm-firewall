import os
import cassio 
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
# from langgraph_checkpoint_cassandra import CassandraSaver  # TODO: Fix serialization issue
from langgraph.graph import StateGraph, END 
from app.graph.state import AgentState
from app.graph.nodes import (
    input_security_node, 
    controlled_rag_node, 
    llm_node, 
    output_security_node, 
    audit_node
)

load_dotenv()

# TODO: Re-enable Cassandra checkpointing once serialization is fixed
# token = os.getenv("ASTRA_DB_APPLICATION_TOKEN").strip()
# db_id = os.getenv("ASTRA_DB_ID").strip()
# keyspace = os.getenv("ASTRA_DB_KEYSPACE", "logs").strip()

# if not db_id or not token:
#     raise ValueError("Missing ASTRA_DB_ID or ASTRA_DB_APPLICATION_TOKEN in .env")

# cassio.init(
#     token=token,
#     database_id=db_id
# )

# from cassio.config import resolve_session
# session = resolve_session()

workflow = StateGraph(AgentState)

workflow.add_node("input_security", input_security_node)
workflow.add_node("controlled_rag", controlled_rag_node)
workflow.add_node("llm", llm_node)
workflow.add_node("output_security", output_security_node)
workflow.add_node("audit", audit_node)

workflow.set_entry_point("input_security")

def route_after_input_security(state: AgentState):
    if state.get("blocked"):
        return "audit"
    return "controlled_rag"

workflow.add_conditional_edges(
    "input_security",
    route_after_input_security,
    {
        "audit": "audit",
        "controlled_rag": "controlled_rag"
    }
)

workflow.add_edge("controlled_rag", "llm")
workflow.add_edge("llm", "output_security")

def route_after_output_security(state: AgentState):
    if state.get("blocked"):
        return "audit"
    
    if not state.get("passed_output_security"):
        return "llm"
    
    return "audit"

workflow.add_conditional_edges(
    "output_security",
    route_after_output_security,
    {
        "audit": "audit",
        "llm": "llm"
    }
)

workflow.add_edge("audit", END)

# Using MemorySaver for conversation history (maintains state during runtime)
# TODO: Switch to CassandraSaver once serialization issue is resolved
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)