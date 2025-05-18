import os
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import Runnable
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage

# Constants
EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07"
LLM_MODEL = "models/gemini-2.5-flash-preview-04-17"
VECTORS_DIR = "vectors"

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_available_vectors():
    """Return list of available vector directories."""
    if os.path.exists(VECTORS_DIR):
        return sorted([d for d in os.listdir(VECTORS_DIR) if os.path.isdir(os.path.join(VECTORS_DIR, d))])
    return []

class ChatEngineRunnable(Runnable):
    """Runnable Chat Engine using ConversationalRetrievalChain"""
    def __init__(self, vector_name: str):
        vector_path = os.path.join(VECTORS_DIR, vector_name)
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        self.db = FAISS.load_local(
            vector_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0.2,
            top_p=0.95,
            convert_system_message_to_human=True   
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )

    def invoke(self, inputs: dict):
        return self.chain.invoke(inputs)


def create_chat_engine_runnable(vector_name: str) -> ChatEngineRunnable:
    """Factory to create ChatEngineRunnable"""
    try:
        return ChatEngineRunnable(vector_name)
    except Exception as e:
        logger.error(f"Failed to create ChatEngineRunnable: {e}")
        raise