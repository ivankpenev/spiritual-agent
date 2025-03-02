import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

class LivesOfTheSaintsScraper:
    """
    Web scraper for collecting information about lives of the saints and spiritual fathers.
    """
    
    def __init__(self, output_dir: str = "data/lives_of_the_saints_raw"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Sources for lives of the saints information
        self.sources = [
            "https://www.orthodoxchristian.info/pages/Saints.html",
            "https://orthodoxwiki.org/Category:Saints",
            # Add more sources as needed
        ]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    
    async def scrape_all(self) -> List[Dict[str, Any]]:
        """
        Scrape information about lives of the saints from all sources.
        
        Returns:
            A list of dictionaries containing information about lives of the saints
        """
        all_saints_data = []
        
        for source in self.sources:
            try:
                saints_data = await self.scrape_source(source)
                all_saints_data.extend(saints_data)
                time.sleep(1)  # Be respectful to the servers
            except Exception as e:
                print(f"Error scraping {source}: {e}")
        
        return all_saints_data
    
    async def scrape_source(self, url: str) -> List[Dict[str, Any]]:
        """
        Scrape information about lives of the saints from a specific source.
        
        Args:
            url: The URL to scrape
            
        Returns:
            A list of dictionaries containing information about lives of the saints
        """
        # This is a placeholder implementation
        # In a real implementation, you would parse the HTML and extract information
        
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # This is just a placeholder - actual implementation would depend on the website structure
        saints_data = []
        
        # Save the raw HTML
        source_name = url.split('/')[-1].split('.')[0]
        with open(os.path.join(self.output_dir, f"{source_name}.html"), 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        return saints_data
    
    async def create_vector_db(self, saints_data: List[Dict[str, Any]], vector_db_path: str = "data/lives_of_the_saints_vectordb"):
        """
        Create a vector database from the scraped lives of the saints information.
        
        Args:
            saints_data: A list of dictionaries containing information about lives of the saints
            vector_db_path: Path to store the vector database
        """
        documents = []
        
        for saint_data in saints_data:
            # Convert saint data to documents
            # This is a placeholder - actual implementation would depend on your data structure
            doc = Document(
                page_content=saint_data.get("description", ""),
                metadata={
                    "name": saint_data.get("name", ""),
                    "feast_day": saint_data.get("feast_day", ""),
                    "source": saint_data.get("source", "")
                }
            )
            documents.append(doc)
        
        # Split documents into chunks
        splits = self.text_splitter.split_documents(documents)
        
        # Create vector store
        embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        vector_db = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=vector_db_path
        )
        
        vector_db.persist()
        
        return vector_db 