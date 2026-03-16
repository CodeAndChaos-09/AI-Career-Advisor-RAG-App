import streamlit as st
import requests

API_URL = "https://ai-career-advisor-rag-app.onrender.com/ask"

st.set_page_config(
    page_title="AI Career Advisor",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------
# Sidebar
# ---------------------------

with st.sidebar:
    st.title("🤖 AI Career Advisor")
    st.write(
        """
This AI assistant helps you explore **career paths in AI, Data Science,
Software Engineering, and Tech**.

Ask questions like:

• How to become a data scientist?  
• Skills needed for AI engineer  
• Career roadmap for machine learning  
• Best programming languages for tech careers
"""
    )

    st.markdown("---")
    st.write("Built with:")
    st.write("• FastAPI")
    st.write("• Streamlit")
    st.write("• LangChain")
    st.write("• FAISS Vector DB")

    st.markdown("---")
    st.caption("🚀 RAG Powered Career Guidance")

# ---------------------------
# Chat Title
# ---------------------------

st.title("💬 AI Career Chat Assistant")

st.write("Ask anything about **AI, Data Science, Programming or Tech Careers**.")

# ---------------------------
# Session State for Chat
# ---------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# Display Chat History
# ---------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------
# Chat Input
# ---------------------------

question = st.chat_input("Ask your career question...")

if question:

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking... 🤔"):

            try:

                response = requests.post(
                    API_URL,
                    json={"question": question}
                )

                result = response.json()

                answer = result["answer"]

                st.markdown(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except:
                st.error("⚠️ Backend server not reachable. Please try again later.")