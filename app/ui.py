import streamlit as st
import requests
import uuid

st.title("Agentic Semantic Firewall")

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

current_sid = st.sidebar.text_input("Session ID", value=st.session_state.session_id)
user_input = st.chat_input("Ask a policy question...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    
    payload = {
        "message": user_input, 
        "session_id": current_sid
    }

    try:
        response = requests.post("http://backend:8000/chat", json=payload)
        if response.status_code == 200:
            data = response.json()
            with st.chat_message("assistant"):
              st.write(data["ai_response"])
              st.caption(f"Status: {data['audit_status']} | Query Redacted: {data['redacted_query']}")
        elif response.status_code == 403:
            st.error("Security Policy Violation: Response Blocked.")
    
    except Exception as e:
        st.error("Could not connect to the backend")