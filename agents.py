from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from config import OPENAI_API_KEY, AGENT_PROMPT
from utils import llm
from agent_tools import rag, db_update
import os

# Load OpenAI API Key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# LLM Setup
# llm = ChatOpenAI(model_name="gpt-4o-mini")

# Initialize Conversation Memory
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template=AGENT_PROMPT)

# Initialize LangChain Agent
agent = initialize_agent(
    tools=[rag, db_update],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_kwargs={"prefix": prompt.template},
    verbose=True,
    max_iterations=2,
    early_stopping_method="generate",
    handle_parsing_errors=True
)
