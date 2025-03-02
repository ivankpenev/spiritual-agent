from langchain.embeddings import OpenAIEmbeddings
import os
from typing import List, Dict, Any

class EmbeddingUtils:
    """
    Utility for generating embeddings for text.
    """
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key
        )
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a piece of text.
        
        Args:
            text: The text to generate an embedding for
            
        Returns:
            A list of floats representing the embedding
        """
        return await self.embeddings.aembed_query(text)
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple pieces of text.
        
        Args:
            texts: A list of texts to generate embeddings for
            
        Returns:
            A list of lists of floats representing the embeddings
        """
        return await self.embeddings.aembed_documents(texts) 