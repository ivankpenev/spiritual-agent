from typing import List, Dict, Any
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

class LivesOfTheSaintsRAG:
    """
    Retrieval-Augmented Generation system for lives of the saints information.
    """
    
    def __init__(self, vector_db_path: str = "data/lives_of_the_saints_vectordb"):
        self.vector_db_path = vector_db_path
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize vector DB if it exists
        if os.path.exists(vector_db_path):
            self.vector_db = Chroma(
                persist_directory=vector_db_path,
                embedding_function=self.embeddings
            )
        else:
            self.vector_db = None
    
    async def query(self, query: str, top_k: int = 5) -> str:
        """
        Query the vector database for relevant information about lives of the saints.
        
        Args:
            query: The query about lives of the saints
            top_k: Number of top results to return
            
        Returns:
            A string containing relevant information about lives of the saints
        """
        if self.vector_db is None:
            return "The lives of the saints database has not been initialized yet."
        
        # Retrieve relevant documents
        docs = self.vector_db.similarity_search(query, k=top_k)
        
        # Format the results
        results = []
        for i, doc in enumerate(docs):
            results.append(f"Source {i+1}:\n{doc.page_content}\n")
        
        return "\n".join(results) 