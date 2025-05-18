import os
import streamlit as st
from typing import List, Dict
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from chat import create_chat_engine_runnable, get_available_vectors
from vector import process_pdf_file_runnable, delete_vector

# Setup
VECTORS_DIR = "vectors"
st.set_page_config(page_title="Educational Chat Assistant", layout="wide")

# Session state
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
    st.title("Educational Chat Assistant")
    tab1, tab2, tab3 = st.tabs(["Chat", "Create Vectors", "Manage Vectors"])

    # Chat tab
    with tab1:
        with st.sidebar:
            st.header("Select Chapter")
            vectors = get_available_vectors()
            if not vectors:
                st.warning("No chapters. Create vectors first.")
            else:
                stds = sorted({v.split("-")[0] for v in vectors})
                sel_std = st.selectbox("Standard", stds)
                subs = sorted({v.split("-")[1] for v in vectors if v.startswith(f"{sel_std}-")})
                sel_sub = st.selectbox("Subject", subs)
                chaps = sorted({v.split("-")[2] for v in vectors if v.startswith(f"{sel_std}-{sel_sub}-")})
                sel_chap = st.selectbox("Chapter", chaps)
                vec = f"{sel_std}-{sel_sub}-{sel_chap}"
                if st.button("Chat with this chapter"):
                    st.session_state.selected_vector = vec
                    st.session_state.chat_history = []
                    st.rerun()

        if st.session_state.selected_vector:
            st.header(f"Chatting: {st.session_state.selected_vector.replace('-', ' > ')}")
            for m in st.session_state.chat_history:
                with st.chat_message(m["role"]):
                    st.write(m["content"])
            user_q = st.chat_input("Ask a question...")
            if user_q:
                st.session_state.chat_history.append({"role":"user","content":user_q})
                with st.chat_message("user"): st.write(user_q)
                # Invoke chat engine
                chat = create_chat_engine_runnable(st.session_state.selected_vector)
                formatted = format_chat_history(st.session_state.chat_history)
                resp = chat.invoke({"question": user_q, "chat_history": formatted})
                answer = resp.get("answer") or resp.get("output") or str(resp)
                with st.chat_message("assistant"): st.write(answer)
                st.session_state.chat_history.append({"role":"assistant","content":answer})
        else:
            st.info("Select a chapter to start chatting.")

    # Create Vectors tab
    with tab2:
        st.header("Create New Chapter Vectors")
        with st.form("vec_form"):
            col1, col2 = st.columns(2)
            with col1:
                std = st.text_input("Standard", "11")
                sub = st.text_input("Subject", "Mathematics")
            with col2:
                chap = st.text_input("Chapter", "Sets")
                upload = st.file_uploader("Upload PDF", type="pdf")
            submit = st.form_submit_button("Process PDF")
        if submit and upload:
            with st.spinner("Processing..."):
                runner = process_pdf_file_runnable(std, sub, chap)
                result = runner.invoke(upload.getvalue())
                if result.get("success"):
                    st.success(result.get("message"))
                else:
                    st.error(result.get("message"))

    # Manage Vectors tab
    with tab3:
        st.header("Manage Chapters")
        vectors = get_available_vectors()
        if not vectors:
            st.info("No vectors found.")
        else:
            to_del = st.multiselect("Select to delete", vectors)
            if st.button("Delete Selected"):
                for v in to_del:
                    ok,msg = delete_vector(v)
                    if ok: st.success(msg)
                    else: st.error(msg)
                if st.session_state.selected_vector in to_del:
                    st.session_state.selected_vector=None
                    st.session_state.chat_history=[]
                st.rerun()
            with st.expander("Available Chapters"):
                for v in vectors:
                    s,sub,c = v.split("-")
                    st.write(f"• {s} > {sub} > {c}")

if __name__ == "__main__":
    os.makedirs(VECTORS_DIR, exist_ok=True)
    main()
