import os
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
from app.redactor import redact_pii
from app.security.input_validator import validate_input
from app.security.access_control import classify_query_sensitivity, has_permission, get_allowed_collections
from app.security.output_filter import validate_output
from app.security.audit_logger import log_interaction
from config import SecurityConfig

load_dotenv()

llm = ChatOllama(
    model="llama3.2:1b",
    base_url="http://host.docker.internal:11434",
    temperature=0.1
)

def get_retriever(collection_name: str):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vstore = AstraDBVectorStore(
        embedding=embeddings,
        collection_name=collection_name,
        api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    )
    return vstore.as_retriever(search_kwargs={"k": SecurityConfig.RAG_TOP_K})

def input_security_node(state: AgentState):
    user_input = state["user_input"]
    user_role = state.get("user_role", "employee")
    
    is_valid, error_msg = validate_input(user_input, user_role)
    if not is_valid:
        return {
            "blocked": True,
            "block_reason": error_msg,
            "redacted_input": user_input,
            "topic_level": "unknown",
            "loop_count": 0
        }
    
    topic_level = classify_query_sensitivity(user_input)
    
    if not has_permission(user_role, topic_level):
        return {
            "blocked": True,
            "block_reason": f"Insufficient permissions for {topic_level} content",
            "redacted_input": user_input,
            "topic_level": topic_level,
            "loop_count": 0
        }
    
    redacted = redact_pii(user_input)
    
    return {
        "redacted_input": redacted,
        "topic_level": topic_level,
        "blocked": False,
        "loop_count": 0
    }

def controlled_rag_node(state: AgentState):
    user_role = state.get("user_role", "employee")
    query = state.get("redacted_input", "")
    
    # Get allowed access levels for this role
    from config import SecurityConfig
    allowed_levels = SecurityConfig.ROLE_PERMISSIONS.get(user_role, ["public"])
    
    # Retrieve from single collection with metadata filtering
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vstore = AstraDBVectorStore(
            embedding=embeddings,
            collection_name="company_knowledge",
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
            token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
        )
        
        # Search with metadata filter for allowed access levels
        docs = vstore.similarity_search(
            query, 
            k=SecurityConfig.RAG_TOP_K,
            filter={"access_level": {"$in": allowed_levels}}
        )
        
        if not docs:
            return {"rag_context": "No relevant information found for your access level."}
        
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"rag_context": context}
        
    except Exception as e:
        # Log error but don't expose internal details
        return {"rag_context": "Error retrieving information."}

def llm_node(state: AgentState):
    user_query = state.get("redacted_input", "")
    context = state.get("rag_context", "")
    loop_count = state.get("loop_count", 0)
    
    system_prompt = f"You are a secure company assistant. Use only the following approved context to answer questions: {context}"
    
    if loop_count > 0:
        system_prompt += "\n\nIMPORTANT: Your previous response contained sensitive information. Rewrite to avoid revealing specific codes, passwords, budget figures, or confidential details. Provide general guidance instead."
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ]
        response = llm.invoke(messages)
        
        return {
            "llm_response": str(response.content),
            "loop_count": loop_count
        }
    except Exception as e:
        # Log error internally, return user-friendly message
        return {
            "llm_response": "I'm having trouble processing your request right now. Please try again later.",
            "loop_count": loop_count,
            "blocked": True,
            "block_reason": f"LLM service error: {str(e)}"
        }

def output_security_node(state: AgentState):
    response = state.get("llm_response", "")
    
    passed, reason, score = validate_output(response)
    
    if not passed:
        loop_count = state.get("loop_count", 0)
        
        if loop_count >= SecurityConfig.MAX_REWRITE_ATTEMPTS:
            return {
                "blocked": True,
                "block_reason": f"Max rewrite attempts reached. {reason}",
                "passed_output_security": False,
                "leak_score": score,
                "loop_count": loop_count + 1
            }
        
        return {
            "passed_output_security": False,
            "leak_score": score,
            "loop_count": loop_count + 1
        }
    
    return {
        "passed_output_security": True,
        "blocked": False,
        "leak_score": score,
        "loop_count": state.get("loop_count", 0)
    }

def audit_node(state: AgentState):
    log_interaction(
        user_id=state.get("user_id", "unknown"),
        user_role=state.get("user_role", "unknown"),
        query=state.get("user_input", ""),
        redacted_query=state.get("redacted_input", ""),
        topic_level=state.get("topic_level", "unknown"),
        response=state.get("llm_response", ""),
        blocked=state.get("blocked", False),
        block_reason=state.get("block_reason", None)
    )
    
    return {}