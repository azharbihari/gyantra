### 📁 services/vector_service.py

import os
import shutil
import tempfile
import logging
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Constants
EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07"
VECTORS_DIR = "vectors"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorService:
    def __init__(self, standard: str, subject: str, chapter: str):
        self.vector_name = f"{standard}-{subject}-{chapter}"
        self.vector_path = os.path.join(VECTORS_DIR, self.vector_name)
        os.makedirs(self.vector_path, exist_ok=True)

    def process_pdf(self, pdf_bytes: bytes) -> dict:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(documents)

            embeddings = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )

            db = FAISS.from_documents(chunks, embeddings)
            db.save_local(self.vector_path)

            os.unlink(tmp_path)
            return {"success": True, "message": f"Processed {len(chunks)} chunks to {self.vector_path}"}
        except Exception as e:
            logger.error(f"Failed to process PDF: {e}")
            return {"success": False, "message": str(e)}

    def delete_vector(self) -> tuple[bool, str]:
        try:
            if os.path.exists(self.vector_path):
                shutil.rmtree(self.vector_path)
                return True, f"Deleted vector: {self.vector_path}"
            return False, "Vector not found"
        except Exception as e:
            logger.error(f"Error deleting vector: {e}")
            return False, str(e)
