from fastapi import FastAPI, Depends, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import os
from dotenv import load_dotenv

from app.agents.core_agent import CoreAgent
from app.rag.lives_of_the_saints_scraper import LivesOfTheSaintsScraper
from app.rag.lives_of_the_saints_rag import LivesOfTheSaintsRAG
from app.utils.embedding_utils import EmbeddingUtils

# Load environment variables
load_dotenv()

app = FastAPI(title="Spiritual Father AI")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
core_agent = CoreAgent(openai_api_key=os.getenv("OPENAI_API_KEY"))
lives_of_the_saints_scraper = LivesOfTheSaintsScraper()
lives_of_the_saints_rag = LivesOfTheSaintsRAG()
embedding_utils = EmbeddingUtils()

class QueryRequest(BaseModel):
    query: str

class EmbeddingRequest(BaseModel):
    text: str

class BatchEmbeddingRequest(BaseModel):
    texts: List[str]

@app.post("/api/chat")
async def chat(request: QueryRequest):
    """
    Process a user query and return a response from the spiritual father AI.
    """
    try:
        response = await core_agent.process_query(request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scrape-lives-of-the-saints")
async def scrape_lives_of_the_saints():
    """
    Scrape information about lives of the saints from various sources.
    """
    try:
        saints_data = await lives_of_the_saints_scraper.scrape_all()
        await lives_of_the_saints_scraper.create_vector_db(saints_data)
        return {"message": f"Successfully scraped information about {len(saints_data)} saints"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query-lives-of-the-saints")
async def query_lives_of_the_saints(request: QueryRequest):
    """
    Query the lives of the saints RAG system for information.
    """
    try:
        results = await lives_of_the_saints_rag.query(request.query)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/embedding")
async def get_embedding(request: EmbeddingRequest):
    """
    Generate an embedding for a piece of text.
    """
    try:
        embedding = await embedding_utils.get_embedding(request.text)
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch-embedding")
async def get_batch_embeddings(request: BatchEmbeddingRequest):
    """
    Generate embeddings for multiple pieces of text.
    """
    try:
        embeddings = await embedding_utils.get_embeddings(request.texts)
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 