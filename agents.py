from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from config import OPENAI_API_KEY, AGENT_PROMPT
from tools import rag, db_update
import os

# Load OpenAI API Key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# if not os.environ.get("OPENAI_API_KEY"):
#   os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
# llm = init_chat_model("gpt-4o-mini", model_provider="openai")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# LLM Setup
llm = ChatOpenAI(model_name="gpt-4o-mini")


# Memory for agent (so it doesn’t forget like a goldfish)
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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
