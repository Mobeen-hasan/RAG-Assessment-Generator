import os
import shutil
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class VectorStoreManager:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        # Dynamic path: We don't hardcode "./chroma_db" anymore.
        # We will let the create method decide the path.

    def create_vector_store(self, documents):
        """
        Creates a NEW vector store in a unique directory.
        """
        # Generate a unique folder name using timestamp to avoid Windows Lock errors
        unique_id = int(time.time())
        persist_directory = f"./chroma_db_{unique_id}"
        
        print(f" -> Creating new DB at {persist_directory}")
        
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=persist_directory
        )
        return vector_store

    # We no longer need get_vector_store because we pass the object directly in Streamlit