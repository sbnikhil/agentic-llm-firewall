import os
import time
from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from app.graph.workflow import app as app_graph
from app.metrics import metrics_collector

app = FastAPI(title="AI Security Gateway API")

class ChatRequest(BaseModel):
    message: str
    session_id: str
    user_id: str
    user_role: str = "employee"

@app.get("/")
async def root():
    return {"status": "online", "engine": "AI Security Gateway"}

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    health_status = {
        "status": "healthy",
        "components": {}
    }
    
    # Check AstraDB connectivity
    try:
        from astrapy import DataAPIClient
        from dotenv import load_dotenv
        load_dotenv()
        
        client = DataAPIClient(os.getenv('ASTRA_DB_APPLICATION_TOKEN'))
        db = client.get_database_by_api_endpoint(
            os.getenv('ASTRA_DB_API_ENDPOINT'),
            keyspace='logs'
        )
        collections = db.list_collection_names()
        health_status["components"]["astradb"] = {
            "status": "healthy",
            "collections": len(collections)
        }
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["astradb"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Check Ollama connectivity
    try:
        import requests
        response = requests.get("http://host.docker.internal:11434/api/tags", timeout=2)
        if response.status_code == 200:
            health_status["components"]["ollama"] = {"status": "healthy"}
        else:
            raise Exception("Ollama not responding")
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["ollama"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    return health_status

@app.get("/metrics")
async def get_metrics():
    return metrics_collector.get_summary_stats()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not request.session_id or not request.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty")
    
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id cannot be empty")
    
    if request.user_role not in ["employee", "manager", "executive"]:
        raise HTTPException(status_code=400, detail="Invalid user_role")

    config = {"configurable": {"thread_id": str(request.session_id)}, "recursion_limit": 50}
    
    initial_state = {
        "user_input": request.message,
        "user_id": request.user_id,
        "user_role": request.user_role,
        "blocked": False,
        "loop_count": 0
    }
    
    start_time = time.time()
    
    try:
        final_state = app_graph.invoke(initial_state, config=config)
        latency_ms = (time.time() - start_time) * 1000
        
        blocked = final_state.get("blocked", False)
        
        metrics_collector.record_request(
            user_id=request.user_id,
            user_role=request.user_role,
            topic_level=final_state.get("topic_level", "unknown"),
            blocked=blocked,
            block_reason=final_state.get("block_reason", None),
            latency_ms=latency_ms,
            leak_score=final_state.get("leak_score", 0.0),
            loop_count=final_state.get("loop_count", 0)
        )
        
        if blocked:
            raise HTTPException(
                status_code=403, 
                detail=f"Security Policy Violation: {final_state.get('block_reason', 'Access denied')}"
            )

        return {
            "session_id": request.session_id,
            "user_id": request.user_id,
            "redacted_query": final_state.get("redacted_input"),
            "ai_response": final_state.get("llm_response"),
            "topic_level": final_state.get("topic_level"),
            "latency_ms": round(latency_ms, 2),
            "audit_status": "Passed"
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")