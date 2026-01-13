import os
import cassio 
from dotenv import load_dotenv
from langgraph_checkpoint_cassandra import CassandraSaver
from langgraph.graph import StateGraph, END 
from app.graph.state import AgentState
from app.graph.nodes import redactor_node, llm_node, auditor_node

load_dotenv()

token = os.getenv("ASTRA_DB_APPLICATION_TOKEN").strip()
db_id = os.getenv("ASTRA_DB_ID").strip()
keyspace = os.getenv("ASTRA_DB_KEYSPACE", "logs").strip()

if not db_id or not token:
    raise ValueError("Missing ASTRA_DB_ID or ASTRA_DB_APPLICATION_TOKEN in .env")

cassio.init(
    token=token,
    database_id=db_id
)

from cassio.config import resolve_session
session = resolve_session()

workflow = StateGraph(AgentState)

# nodes
workflow.add_node("redactor", redactor_node)
workflow.add_node("llm", llm_node)
workflow.add_node("auditor", auditor_node)

#entry point
workflow.set_entry_point("redactor")

# edges
workflow.add_edge("redactor", "llm")
workflow.add_edge("llm", "auditor")

def decide_to_finish(state: AgentState):
    if not state.get("is_leak") or state.get("loop_count", 0) >= 3:
        return "end"
    return "rewrite"

workflow.add_conditional_edges(
    "auditor",
    decide_to_finish,
    {
        "end": END,
        "rewrite": "llm"
    }
)

checkpointer = CassandraSaver(session=session, keyspace=keyspace) 
#checkpointer.setup()
app = workflow.compile(checkpointer=checkpointer)