import os
import streamlit as st
from typing import List, Dict
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from services.chat_service import ChatService, get_available_vectors
from services.vector_service import VectorService

VECTORS_DIR = "vectors"
os.makedirs(VECTORS_DIR, exist_ok=True)

st.set_page_config(page_title="Gyantra: NCERT Solutions", layout="centered")

if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict[str, str]] = []
if "selected_vector" not in st.session_state:
    st.session_state.selected_vector = None


def format_chat_history(history: List[Dict[str, str]]) -> List[BaseMessage]:
    formatted = []
    for msg in history:
        if msg["role"] == "user":
            formatted.append(HumanMessage(content=msg["content"]))
        else:
            formatted.append(AIMessage(content=msg["content"]))
    return formatted


def main():
    st.title("Gyantra: NCERT Solutions") 
    tab1, tab2 = st.tabs(["Chat", "Manage Vectors"])

    with tab1:
        st.subheader("Select Chapter to Chat")
        vectors = get_available_vectors()
        if not vectors:
            st.warning("No chapters available. Please create vectors first in the 'Manage Vectors' tab.")
        else:
            sel_vector = st.selectbox("Select Chapter to Chat", vectors)
            st.session_state.selected_vector = sel_vector

        if st.session_state.selected_vector:
            st.markdown(f"### Chatting: **{st.session_state.selected_vector.replace('-', ' > ')}**")
            for m in st.session_state.chat_history:
                with st.chat_message(m["role"]):
                    st.write(m["content"])

            user_q = st.chat_input("Ask a question...")
            if user_q:
                st.session_state.chat_history.append({"role": "user", "content": user_q})
                with st.chat_message("user"):
                    st.write(user_q)

                chat_service = ChatService(st.session_state.selected_vector)
                formatted_history = format_chat_history(st.session_state.chat_history)
                response = chat_service.chat(user_q, formatted_history)
                answer = response.get("answer") or str(response)

                with st.chat_message("assistant"):
                    st.write(answer)

                st.session_state.chat_history.append({"role": "assistant", "content": answer})
        elif vectors:
            st.info("Select and load a chapter above to start chatting.")

    with tab2:
        st.header("Manage and Create Chapter Vectors")

        st.subheader("Create New Chapter")
        with st.form("create_vector_form"):
            std = st.text_input("Standard", "11")
            sub = st.text_input("Subject", "Mathematics")
            chap = st.text_input("Chapter", "Sets")
            upload = st.file_uploader("Upload PDF file", type="pdf")
            submit = st.form_submit_button("Process PDF")

        if submit:
            if not upload:
                st.error("Please upload a PDF file.")
            else:
                vector_service = VectorService(std.strip(), sub.strip(), chap.strip())
                with st.spinner("Processing PDF and building vectors..."):
                    result = vector_service.process_pdf(upload.getvalue())
                    if result.get("success"):
                        st.success(result.get("message"))
                    else:
                        st.error(result.get("message"))

        st.subheader("Existing Chapters")
        vectors = get_available_vectors()
        if vectors:
            with st.expander("Show all chapters", expanded=True):
                for v in vectors:
                    col1, col2 = st.columns([0.8, 0.2])
                    with col1:
                        st.write(f"• {v.replace('-', ' > ')}")
                    with col2:
                        if st.button("Delete", key=f"del-{v}"):
                            vector_service = VectorService(*v.split("-"))
                            ok, msg = vector_service.delete_vector()
                            if ok:
                                st.success(msg)
                                if st.session_state.selected_vector == v:
                                    st.session_state.selected_vector = None
                                    st.session_state.chat_history = []
                                st.experimental_rerun()
                            else:
                                st.error(msg)
        else:
            st.info("No chapter vectors found.")


if __name__ == "__main__":
    main()
