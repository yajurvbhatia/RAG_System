import  config
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
# import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from langchain.chat_models import init_chat_model


import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = config.open_ai_key


llm = init_chat_model("gpt-4o-mini", model_provider="openai")



from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")



# llm = AzureChatOpenAI(
#     azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
#     api_key= config.AZURE_OPENAI_API_KEY,
#     azure_deployment=config.CHAT_COMPLETIONS_DEPLOYMENT_NAME,
#     openai_api_version=config.AZURE_OPENAI_API_VERSION
# )

# embeddings = AzureOpenAIEmbeddings(
#     azure_endpoint=config.ENDPOINT,
#     api_key= config.API_KEY,
#     # azure_deployment=config.AZURE_OPENAI_DEPLOYMENT_NAME,
#     openai_api_version=config.API_VERSION
# )
 
# embedding_dim = len(embeddings.embed_query("hello world"))
# index = faiss.IndexFlatL2(embedding_dim)

# vector_store = FAISS(
#     embedding_function=embeddings,
#     index=index,
#     docstore=InMemoryDocstore(),
#     index_to_docstore_id={},
# )

 
