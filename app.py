import sys
import os
import streamlit as st
from pathlib import Path

# Import LangChain components
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


# --- Configuration ---
VECTOR_STORE_PATH = os.path.join(os.getcwd(), "vectors")
EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07"  # Google's embedding model
LLM_MODEL = "models/gemini-2.5-pro-preview-05-06"  # Google's LLM model

def check_vector_store_exists(path):
    """Check if vector store files exist at the given path"""
    index_path = os.path.join(path, "index.faiss")
    pkl_path = os.path.join(path, "index.pkl")
    return os.path.exists(index_path) and os.path.exists(pkl_path)

@st.cache_resource
def load_vector_store(path):
    """Load FAISS vector store from disk with error handling"""
    try:
        # Check if the vector store exists
        if not check_vector_store_exists(path):
            return None
        
        # Initialize embeddings and load vector store
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        vs = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        return vs
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def extract_metadata_options(_vector_store):
    """Extract unique metadata values from the vector store"""
    vector_store = _vector_store
    if vector_store is None:
        return [], [], [], []
        
    # Get documents from FAISS docstore
    docs = vector_store.docstore._dict.values()
    
    # Extract metadata from documents
    metas = [doc.metadata for doc in docs]
    
    # Extract unique values for each metadata field
    standards = sorted({m.get('standard', 'Unknown') for m in metas})
    subjects = sorted({m.get('subject', 'Unknown') for m in metas})
    chapters = sorted({m.get('chapter', 'Unknown') for m in metas})
    
    return metas, standards, subjects, chapters

def display_setup_instructions():
    """Display instructions for setting up the vector store"""
    st.title("📚 RAG QA System Setup")
    st.error("Vector store not found! Please create it first.")
    
    st.markdown("""
    ## How to Create Your Vector Store
    
    You need to create a FAISS vector store before using this application. You can:
    
    1. Run the included `create_vector_store.py` script:
    ```bash
    streamlit run create_vector_store.py
    ```
    
    2. Or use the command below to create it manually:
    ```python
    from langchain.document_loaders import DirectoryLoader
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_community.vectorstores import FAISS
    
    # Load your documents (customize the path)
    loader = DirectoryLoader("./documents", glob="**/*.pdf")
    documents = loader.load()
    
    # Add metadata to documents (make sure each document has standard, subject, chapter fields)
    # ... (custom metadata extraction code) ...
    
    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
    
    # Create and save vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("faiss_vector_store")
    ```
    """)
    
    # Allow user to update the vector store path
    st.markdown("## Vector Store Location")
    new_path = st.text_input("Vector Store Path:", value=VECTOR_STORE_PATH)
    if st.button("Check Path"):
        if check_vector_store_exists(new_path):
            st.success(f"Vector store found at {new_path}!")
            st.info("Update your code with this path and restart the application.")
        else:
            st.error(f"No vector store found at {new_path}")
    
def main():
    """Main application function"""
    # Set up page configuration
    st.set_page_config(page_title="Gyantra", page_icon="📚", layout="wide")
    
    # Sidebar title
    st.sidebar.title("💬 Gyantra\nYour AI Learning Assistant")
    
    # Load vector store
    vector_store = load_vector_store(VECTOR_STORE_PATH)
    
    # Check if vector store loaded successfully
    if vector_store is None:
        display_setup_instructions()
        return
        
    # Extract metadata options
    metas, standards, subjects, chapters = extract_metadata_options(vector_store)
    
    # --- Sidebar filters ---
    st.sidebar.markdown("## Document Filters")
    selected_standard = st.sidebar.selectbox("Standard", standards)
    
    # Filter subjects based on selected standard
    available_subjects = sorted({
        m['subject'] for m in metas 
        if m.get('standard') == selected_standard
    })
    selected_subject = st.sidebar.selectbox("Subject", available_subjects)
    
    # Filter chapters based on selected standard and subject
    available_chapters = sorted({
        m['chapter'] for m in metas 
        if m.get('standard') == selected_standard and m.get('subject') == selected_subject
    })
    selected_chapter = st.sidebar.selectbox("Chapter", available_chapters)
    
    # Create a retriever with metadata filters
    retriever = vector_store.as_retriever(
        search_kwargs={
            "filter": {
                "standard": selected_standard,
                "subject": selected_subject,
                "chapter": selected_chapter
            },
            "k": 5  # Number of documents to retrieve
        }
    )
    
    # Set up memory with output_key to fix the error
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # This specifies which output to store
    )
    
    # Set up LLM and conversational chain
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True
    )
    
    # Main content area
    st.title("📚 Gyantra")
    st.markdown(f"**Current Chapter:** {selected_standard} > {selected_subject} > {selected_chapter}")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if this is an assistant message with sources
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("View Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- **{source}**")
    
    # Chat input
    user_query = st.chat_input("Ask a question about this chapter...")
    
    # Process user input
    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Display assistant response with a spinner
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                # Get response from LLM
                try:
                    result = conversation({"question": user_query})
                    answer = result["answer"]
                    sources = [doc.metadata.get('source_file', 'Unknown') for doc in result['source_documents']]
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                    
                    # Display the response
                    message_placeholder.markdown(answer)
                    
                    # Display sources
                    with st.expander("View Sources"):
                        for source in sources:
                            st.markdown(f"- **{source}**")
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
    
    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Built with LangChain, Streamlit & Google Gemini")
    
    # Add a button to clear chat history
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        memory.clear()
        st.rerun()

if __name__ == "__main__":
    main()