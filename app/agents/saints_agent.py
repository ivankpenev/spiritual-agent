from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from app.rag.saints_rag import SaintsRAG

class SaintsQueryInput(BaseModel):
    """Input for the Saints Expert query."""
    query: str = Field(description="The query about saints or spiritual fathers")

class SaintsAgent:
    """
    Subagent specialized in knowledge about lives of saints and spiritual fathers.
    """
    
    def __init__(self, openai_api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(api_key=openai_api_key, model=model_name)
        self.rag = SaintsRAG()
        
        # Create the agent prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert on the lives of saints and spiritual fathers.
            Your purpose is to provide accurate, detailed information about saints, their lives,
            teachings, and spiritual wisdom. Use the RAG system to retrieve relevant information
            when needed. Always cite your sources when possible.
            """),
            ("human", "{input}"),
        ])
    
    async def query(self, query: str) -> str:
        """
        Process a query about saints and return a response.
        
        Args:
            query: The query about saints or spiritual fathers
            
        Returns:
            A detailed response with information about the saints
        """
        # Retrieve relevant information from the RAG system
        rag_results = await self.rag.query(query)
        
        # Generate a response using the retrieved information
        response = await self.llm.ainvoke(
            self.prompt.format_messages(
                input=f"Query: {query}\n\nRelevant information: {rag_results}"
            )
        )
        
        return response.content
    
    def get_tool(self) -> BaseTool:
        """
        Create and return a tool for the core agent to use this subagent.
        
        Returns:
            A tool that can be used by the core agent
        """
        
        async def _run(query: str) -> str:
            return await self.query(query)
        
        return BaseTool(
            name="SaintsExpert",
            description="Use this tool when you need information about saints, spiritual fathers, their lives, teachings, or wisdom.",
            func=_run,
            args_schema=SaintsQueryInput
        ) 