import streamlit as st
from src.rag import ask

st.title("Ask my CV")
q = st.text_input("Ask a question about my CV")

if q:
    answer, docs = ask(q)
    st.write(answer)
    with st.expander("Retrieved chunks"):
        for d in docs:
            st.caption(f"Page {d.metadata['page'] + 1}")
            st.write(d.page_content)