import os
import tempfile
import shutil
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import Runnable

# Constants
EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07"
VECTORS_DIR = "vectors"

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_vector_path(standard, subject, chapter):
    return os.path.join(VECTORS_DIR, f"{standard}-{subject}-{chapter}")

def delete_vector(vector_name: str):
    vector_path = os.path.join(VECTORS_DIR, vector_name)
    try:
        if os.path.exists(vector_path):
            shutil.rmtree(vector_path)
            return True, f"Successfully deleted vector: {vector_path}"
        return False, f"Vector not found: {vector_path}"
    except Exception as e:
        logger.error(f"Error deleting vector: {e}")
        return False, str(e)

class ProcessPDFRunnable(Runnable):
    """Runnable to process PDF and create FAISS vectors"""
    def __init__(self, standard: str, subject: str, chapter: str):
        self.vector_path = get_vector_path(standard, subject, chapter)
        os.makedirs(self.vector_path, exist_ok=True)
        self.standard = standard
        self.subject = subject
        self.chapter = chapter

    def invoke(self, pdf_bytes: bytes) -> dict:
        try:
            # Save PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            # Load and split
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(documents)

            # Embed and store
            embeddings = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            db = FAISS.from_documents(chunks, embeddings)
            db.save_local(self.vector_path)

            os.unlink(tmp_path)
            return {"success": True, "message": f"Processed {len(chunks)} chunks to {self.vector_path}"}
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return {"success": False, "message": str(e)}


def process_pdf_file_runnable(standard: str, subject: str, chapter: str) -> ProcessPDFRunnable:
    return ProcessPDFRunnable(standard, subject, chapter)