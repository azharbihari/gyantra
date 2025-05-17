import os
import sys
import streamlit as st
from pathlib import Path
import time
import re

# Import LangChain components
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
DOCUMENTS_DIR = os.path.join(os.getcwd(), "documents")
VECTOR_STORE_PATH = os.path.join(os.getcwd(), "vectors")
EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07"  # Google's embedding model

# Configure page
st.set_page_config(
    page_title="Vector Store Creator",
    page_icon="🗃️",
    layout="wide"
)

def create_directory_if_not_exists(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        return f"Created directory: {path}"
    return f"Directory already exists: {path}"

def extract_metadata_from_filename(filename):
    """
    Extract metadata from filename using regex pattern.
    Expected format: Standard_Subject_Chapter.pdf
    Example: Grade10_Science_Chapter3_Motion.pdf
    """
    # Remove extension
    base_name = os.path.splitext(filename)[0]
    
    # Try to match pattern: Standard_Subject_Chapter
    pattern = r"^(.+?)_(.+?)_(.+)$"
    match = re.match(pattern, base_name)
    
    if match:
        standard, subject, chapter = match.groups()
        return {
            "standard": standard,
            "subject": subject,
            "chapter": chapter,
            "source_file": filename
        }
    
    # Fallback if pattern doesn't match
    return {
        "standard": "Unknown",
        "subject": "Unknown",
        "chapter": base_name,  # Use the whole filename as chapter
        "source_file": filename
    }

def load_documents_with_metadata():
    """Load documents from directory and add metadata"""
    st.info(f"Looking for documents in: {DOCUMENTS_DIR}")
    
    # Create directory if it doesn't exist
    create_directory_if_not_exists(DOCUMENTS_DIR)
    
    # Check if documents exist
    pdf_files = list(Path(DOCUMENTS_DIR).glob("**/*.pdf"))
    if not pdf_files:
        st.warning("No PDF files found in the documents directory.")
        return []
    
    st.info(f"Found {len(pdf_files)} PDF files. Processing...")
    progress_bar = st.progress(0)
    
    documents = []
    for i, pdf_path in enumerate(pdf_files):
        try:
            # Load PDF document
            loader = PyPDFLoader(str(pdf_path))
            pdf_docs = loader.load()
            
            # Extract metadata from filename
            metadata = extract_metadata_from_filename(pdf_path.name)
            
            # Update metadata for each page
            for doc in pdf_docs:
                doc.metadata.update(metadata)
            
            documents.extend(pdf_docs)
            
            # Update progress
            progress = (i + 1) / len(pdf_files)
            progress_bar.progress(progress)
            st.text(f"Processed: {pdf_path.name}")
            
        except Exception as e:
            st.error(f"Error processing {pdf_path.name}: {str(e)}")
    
    st.success(f"Successfully processed {len(documents)} pages from {len(pdf_files)} PDF files")
    return documents

def create_chunks(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks for better retrieval"""
    if not documents:
        return []
        
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Split documents
    st.info(f"Splitting documents into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    chunks = text_splitter.split_documents(documents)
    st.success(f"Created {len(chunks)} chunks from {len(documents)} document pages")
    
    return chunks

def create_vector_store(documents):
    """Create vector store from documents"""
    if not documents:
        st.error("No documents to process.")
        return False
    
    try:
        # Initialize embeddings
        st.info(f"Initializing embeddings model: {EMBEDDING_MODEL}")
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        
        # Create vector store
        st.info("Creating FAISS vector store (this may take a while)...")
        start_time = time.time()
        
        with st.spinner("Creating embeddings and building index..."):
            vector_store = FAISS.from_documents(documents, embeddings)
        
        elapsed_time = time.time() - start_time
        st.success(f"Vector store created in {elapsed_time:.2f} seconds")
        
        # Save vector store
        st.info(f"Saving vector store to {VECTOR_STORE_PATH}")
        vector_store.save_local(VECTOR_STORE_PATH)
        st.success("Vector store saved successfully!")
        
        return True
    
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return False

def display_document_preview(documents):
    """Display a preview of processed documents and their metadata"""
    if not documents:
        return
    
    st.subheader("Document Preview")
    num_preview = min(5, len(documents))  # Preview up to 5 documents
    
    for i, doc in enumerate(documents[:num_preview]):
        with st.expander(f"Document {i+1} - {doc.metadata.get('source_file', 'Unknown')}"):
            st.write("**Metadata:**")
            for key, value in doc.metadata.items():
                st.write(f"- {key}: {value}")
            
            st.write("**Content Preview:**")
            # Display first 300 characters
            content_preview = doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else "")
            st.text(content_preview)
    
    if len(documents) > num_preview:
        st.info(f"Showing {num_preview} of {len(documents)} documents. All will be processed.")

def main():
    st.title("🗃️ Vector Store Creator for RAG System")
    
    # Instructions
    with st.expander("Instructions", expanded=True):
        st.markdown("""
        ## How to use this tool
        
        1. **Prepare your documents**:
           - Place PDF files in the `documents` folder
           - Name them with the format: `Standard_Subject_Chapter.pdf`
           - Example: `Grade10_Science_Chapter3_Motion.pdf`
        
        2. **Create the vector store**:
           - Click the "Create Vector Store" button below
           - The process will:
             - Extract metadata from file names
             - Process PDFs
             - Create embeddings
             - Build a FAISS vector store
        
        3. **Use the vector store**:
           - Once created, you can use this vector store with the main RAG application
        """)
    
    # Configuration options
    with st.expander("Configuration"):
        global DOCUMENTS_DIR, VECTOR_STORE_PATH
        
        documents_dir = st.text_input("Documents Directory:", DOCUMENTS_DIR)
        vector_store_path = st.text_input("Vector Store Path:", VECTOR_STORE_PATH)
        
        if st.button("Update Paths"):
            DOCUMENTS_DIR = documents_dir
            VECTOR_STORE_PATH = vector_store_path
            st.success("Paths updated!")
        
        # Chunking options
        do_chunking = st.checkbox("Split documents into chunks", value=False)
        
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input("Chunk Size:", min_value=100, value=1000)
        with col2:
            chunk_overlap = st.number_input("Chunk Overlap:", min_value=0, value=200)
    
    # Check if vector store already exists
    vector_store_exists = os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss"))
    
    if vector_store_exists:
        st.warning(f"Vector store already exists at {VECTOR_STORE_PATH}")
        st.warning("Creating a new vector store will overwrite the existing one.")
        
        if st.button("Create New Vector Store (Overwrite Existing)"):
            # Load documents with metadata
            documents = load_documents_with_metadata()
            
            # Display document preview
            display_document_preview(documents)
            
            # Process documents
            if do_chunking:
                documents = create_chunks(documents, chunk_size, chunk_overlap)
            
            # Create vector store
            if create_vector_store(documents):
                st.balloons()
                st.success("Vector store created successfully! You can now use it with the RAG application.")
    else:
        if st.button("Create Vector Store"):
            # Load documents with metadata
            documents = load_documents_with_metadata()
            
            # Display document preview
            display_document_preview(documents)
            
            # Process documents
            if do_chunking:
                documents = create_chunks(documents, chunk_size, chunk_overlap)
            
            # Create vector store
            if create_vector_store(documents):
                st.balloons()
                st.success("Vector store created successfully! You can now use it with the RAG application.")
    
    # Additional information
    st.markdown("---")
    st.subheader("File Naming Convention")
    st.markdown("""
    For best results, name your PDF files in this format:
    ```
    Standard_Subject_Chapter.pdf
    ```
    
    Examples:
    - `Grade10_Science_Chapter3_Motion.pdf`
    - `CBSE_Mathematics_Integration.pdf`
    - `IB_History_WorldWar2.pdf`
    
    This helps the RAG system organize and filter your documents properly.
    """)

if __name__ == "__main__":
    main()