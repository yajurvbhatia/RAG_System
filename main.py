from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from agents import agent
import uuid
from typing import Optional

import os
import tempfile

from utils import embeddings, get_or_create_memory, session_memories
from config import VECTOR_STORE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, AGENT_PROMPT

app = FastAPI()


@app.post("/process-pdf/")
async def process_pdf(file: UploadFile = File(...)):
    try:
        # Create directory for vector stores if it doesn't exist
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        # Load PDF with LangChain
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Remove temporary file
        os.unlink(tmp_path)
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        # Save vector store locally
        filename = file.filename.replace(".pdf", "")
        save_path = os.path.join(VECTOR_STORE_DIR, f"{filename}_faiss_index")
        vectorstore.save_local(save_path)
        
        return {
            "message": "PDF processed successfully",
            "vector_store_name": save_path,
            "num_documents": len(splits)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


# Request model
class QueryRequest(BaseModel):
    query: str
    vector_store_name: str
    session_id: Optional[str] = None


@app.post("/query-pdf/")
async def query_pdf(request: QueryRequest):

    if not request.session_id:
        session_id = str(uuid.uuid4())
    else:
        session_id = request.session_id

    # Run the agent
    user_query = str(request.query)
    agent.memory = get_or_create_memory(session_id)
    session_memories[session_id] = agent.memory
    print("Session memory stored")

    response = agent.run(f"'user_query': {user_query}, 'session_id': {session_id}")
    print(f"\n🤖 Agent: {response}")
    print(f"\n🤖 Memory: {agent.memory.buffer}")

    return {
        "agent_response": response,
        "session_id": session_id
    }