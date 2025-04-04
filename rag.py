import  config
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = config.open_ai_key

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")