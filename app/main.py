import os
from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from app.graph.workflow import app as app_graph 

app = FastAPI(title="Agentic Semantic Firewall API")

class ChatRequest(BaseModel):
    message: str
    session_id: str

@app.get("/")
async def root():
    return {"status": "online", "engine": "Agentic Semantic Firewall"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not request.session_id or not request.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty")

    config = {"configurable": {"thread_id": str(request.session_id)}, "recursion_limit": 50}
    
    initial_state = {
        "user_input": request.message,
        "messages": [HumanMessage(content=request.message)],
        "is_leak": False,
        "loop_count": 0
    }
    
    try:
        final_state = app_graph.invoke(initial_state, config=config)
        
        if final_state.get("is_leak"):
             raise HTTPException(
                status_code=403, 
                detail="Security Policy Violation: Sensitive data leak detected."
            )

        return {
            "session_id": request.session_id,
            "redacted_query": final_state.get("redacted_input"),
            "ai_response": final_state.get("llm_response"),
            "audit_status": "Passed"
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Graph Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")