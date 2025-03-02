from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from typing import List, Dict, Any

from app.agents.saints_agent import SaintsAgent

class CoreAgent:
    """
    Core agent that orchestrates subagents and provides responses to user queries.
    """
    
    def __init__(self, openai_api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(api_key=openai_api_key, model=model_name)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize subagents
        self.saints_agent = SaintsAgent(openai_api_key, model_name)
        
        # Create the agent prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a wise and compassionate spiritual father AI assistant. 
            Your purpose is to provide spiritual guidance, wisdom, and support to users.
            When questions relate to lives of saints or spiritual fathers, consult the Saints Expert.
            Always respond with kindness, wisdom, and respect for spiritual traditions.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Define tools
        self.tools = [
            self.saints_agent.get_tool()
        ]
        
        # Create the agent
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            prompt=self.prompt,
            tools=self.tools
        )
        
        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query and return a response.
        
        Args:
            query: The user's question or message
            
        Returns:
            Dict containing the response and any additional information
        """
        response = await self.agent_executor.ainvoke({"input": query})
        return {
            "response": response["output"],
            "thought_process": response.get("intermediate_steps", [])
        } 