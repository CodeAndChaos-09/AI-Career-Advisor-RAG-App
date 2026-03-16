import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="AI Career Advisor", page_icon="🤖")

st.title("🤖 AI Career Advisor")
st.write("Ask anything about careers, AI, data science, or tech.")

question = st.text_input("Enter your question")

if st.button("Ask AI"):

    if question.strip() == "":
        st.warning("Please enter a question")
    else:
        try:
            response = requests.post(
                API_URL,
                json={"question": question}
            )

            result = response.json()

            answer = result["answer"]

            st.success("Answer")

            st.write(answer)

        except Exception as e:
            st.error("Backend not running. Please start FastAPI server.")