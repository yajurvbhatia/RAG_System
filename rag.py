import  config
from langchain_openai import OpenAIEmbeddings
import os
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

llm = ChatOpenAI(model_name="gpt-4o-mini")

# llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

session_memories = {}

def get_or_create_memory(session_id: str):
    print("session id type is:", type(session_id))
    print("Current Session Memories:", list(session_memories.keys()))
    print("Incoming session_id:", repr(session_id))
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