import  config
from langchain_openai import OpenAIEmbeddings
import os
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

llm = ChatOpenAI(model_name="gpt-4o-mini")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

session_memories = {}

def get_or_create_memory(session_id: str):
    if session_id not in session_memories:
        print("CREATING NEW MEMORY FOR:", repr(session_id))
        session_memories[session_id] = ConversationSummaryMemory(
           llm=llm,
           memory_key="chat_history",
           return_messages=True
        )
    else:
        print("USING EXISTING MEMORY FOR:", repr(session_id))
    return session_memories[session_id]