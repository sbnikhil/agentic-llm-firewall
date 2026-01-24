import streamlit as st
import requests
import uuid

st.title("AI Security Gateway")

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if 'messages' not in st.session_state:
    st.session_state.messages = []

current_sid = st.sidebar.text_input("Session ID", value=st.session_state.session_id)
user_id = st.sidebar.text_input("User ID", value="demo_user")
user_role = st.sidebar.selectbox("User Role", ["employee", "manager", "executive"])

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "caption" in message:
            st.caption(message["caption"])

user_input = st.chat_input("Ask a policy question...")

if user_input:
    payload = {
        "message": user_input, 
        "session_id": current_sid,
        "user_id": user_id,
        "user_role": user_role
    }

    try:
        response = requests.post("http://backend:8000/chat", json=payload)
        if response.status_code == 200:
            data = response.json()
            
            # Store and display redacted user message
            redacted_msg = data.get("redacted_query", user_input)
            st.session_state.messages.append({"role": "user", "content": redacted_msg})
            with st.chat_message("user"):
                st.write(redacted_msg)
            
            # Store and display assistant response
            ai_msg = data["ai_response"]
            caption = f"Status: {data['audit_status']} | Topic: {data.get('topic_level', 'N/A')}"
            st.session_state.messages.append({"role": "assistant", "content": ai_msg, "caption": caption})
            with st.chat_message("assistant"):
                st.write(ai_msg)
                st.caption(caption)
        elif response.status_code == 403:
            error_data = response.json()
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
            st.error(f"Security Policy Violation: {error_data.get('detail', 'Access Denied')}")
    
    except Exception as e:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        st.error("Could not connect to the backend")