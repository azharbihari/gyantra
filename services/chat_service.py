### 📁 services/chat_service.py

import os
import logging
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Constants
EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07"
LLM_MODEL = "models/gemini-2.5-flash-preview-04-17"
VECTORS_DIR = "vectors"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, vector_name: str):
        vector_path = os.path.join(VECTORS_DIR, vector_name)
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

        self.db = FAISS.load_local(
            vector_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        self.retriever = self.db.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
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

    def chat(self, question: str, chat_history: list[BaseMessage]):
        return self.chain.invoke({
            "question": question,
            "chat_history": chat_history
        })

def get_available_vectors() -> list[str]:
    path = VECTORS_DIR
    if os.path.exists(path):
        return sorted([
            d for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        ])
    return []