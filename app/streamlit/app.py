import os
import json
import requests
import streamlit as st
from typing import List, Dict, Any, Optional
import time

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="KG-Chat",
    page_icon="💬",
    layout="wide",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .response-container {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        border-left: 5px solid #3498DB;
    }
    .user-message {
        background-color: #E8F8F5;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 3px solid #1ABC9C;
    }
    .assistant-message {
        background-color: #F5EEF8;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 3px solid #8E44AD;
    }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    .stChatMessage {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "processing" not in st.session_state:
    st.session_state["processing"] = False

st.markdown('<h1 class="main-title">KG-Chat</h1>', unsafe_allow_html=True)

response_type = st.sidebar.radio(
    "Response Type",
    ["concise", "detailed"],
    index=0,
    help="Choose whether you want brief answers or more detailed explanations"
)

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question about diabetes", disabled=st.session_state["processing"])

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Processing your query..."):
            st.session_state["processing"] = True
            
            try:
                history = [
                    {"role": msg["role"], "content": msg["content"]} 
                    for msg in st.session_state["messages"][-10:] 
                    if msg["role"] == "user" or msg["role"] == "assistant"
                ]
                
                response = requests.post(
                    f"{API_URL}/api/query",
                    json={
                        "query": prompt,
                        "conversation_history": history,
                        "response_type": response_type
                    },
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    assistant_response = result.get("response", "No response from API.")
                    
                    st.session_state["messages"].append({
                        "role": "assistant", 
                        "content": assistant_response
                    })
                    
                    message_placeholder.markdown(assistant_response)
                    
                else:
                    error_message = f"Error: API returned status code {response.status_code}"
                    message_placeholder.error(error_message)
                    
                    st.session_state["messages"].append({
                        "role": "assistant", 
                        "content": f"{error_message}. Please try again."
                    })
            
            except requests.exceptions.Timeout:
                timeout_message = "The request timed out. The server is taking too long to respond."
                message_placeholder.error(timeout_message)
                
                st.session_state["messages"].append({
                    "role": "assistant", 
                    "content": f"{timeout_message} Please try a simpler query or try again later."
                })
                
            except Exception as e:
                error_message = f"Error processing query: {str(e)}"
                message_placeholder.error(error_message)
                
                st.session_state["messages"].append({
                    "role": "assistant", 
                    "content": f"{error_message} Please try again."
                })
            
            finally:
                st.session_state["processing"] = False

st.sidebar.title("Chat Controls")

if st.sidebar.button("Clear Chat History"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you today?"}]
    st.rerun()

st.markdown("""
<div style="position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; color: #7F8C8D; background-color: #1E1E1E;">
    <p>KG-Chat | Powered by FastAPI and Streamlit</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("System Status")
if st.sidebar.button("Check API Status"):
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            st.sidebar.success("API is online ✅")
        else:
            st.sidebar.error(f"API returned status code {health_response.status_code} ❌")
    except Exception as e:
        st.sidebar.error(f"API is offline or unreachable: {str(e)} ❌")
