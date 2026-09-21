import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title("Codebasics Q&A 🌱")
btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    try:
        response = chain.invoke(question)
        st.header("Answer")
        st.write(response)
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            st.error("⚠️ Rate limit reached. Please wait a minute and try again.")
        else:
            st.error(f"Error: {e}")






