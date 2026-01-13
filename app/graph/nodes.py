import os
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from app.graph.state import AgentState
from app.redactor import redact_pii  
from app.database import leak_check 

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1
)

def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vstore = AstraDBVectorStore(
        embedding=embeddings,
        collection_name="company_knowledge",
        api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    )
    return vstore.as_retriever(search_kwargs={"k": 2})

def redactor_node(state: AgentState):
    print("--redacting input---")
    safe_text = redact_pii(state["user_input"])
    return {"redacted_input": safe_text, "loop_count": 0}

def llm_node(state: AgentState):
    print("---searching knowledge base & generating response---")
    user_query = state.get("redacted_input", "")
    retriever = get_retriever()
    relevant_docs = retriever.invoke(user_query)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    response = llm.invoke([
        ("system", f"Secure Assistant. Context: {context}"),
        ("human", user_query)
    ])
    return {"llm_response": str(response.content)}

def auditor_node(state: AgentState):
    print("--auditing output---")
    is_leak, score = leak_check(state.get("llm_response", ""))
    
    return {
        "is_leak": bool(is_leak), 
        "leak_score": float(score),
        "loop_count": state.get("loop_count", 0) + 1
    }