from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from agents import create_agent
import uuid
from typing import Optional

import os
import tempfile

from utils import llm, embedder, active_sessions, Session, connect_to_db, save_vectors, embedder, get_last_conversation_chat_history
from config import CHUNK_SIZE, CHUNK_OVERLAP

from pgvector.psycopg2 import register_vector

app = FastAPI()


@app.post("/process-pdf/")
async def process_pdf(file: UploadFile = File(...)):
    try:
        # print("Starting PDF processing...")

        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        # Load PDF with LangChain
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        # print("Loaded Document...")
        
        # Remove temporary file
        os.unlink(tmp_path)
        
        # Split documents
        # print("Splitting the document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(documents)
        
        # Getting Embeddings for chunks
        # print("Getting embeddings for chunks...")
        texts = [doc.page_content for doc in splits]
        vectors = embedder.embed_documents(texts)

        # Save the vectors to PGVector
        # print("Saving vectors to PGVector...")
        conn = connect_to_db()
        register_vector(conn)
        save_vectors(conn, texts, vectors)

        return {
            "message": "PDF Chunked and Indexed in Vector DB successfully",
            "num_documents": len(splits)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


# Request model
class QueryRequest(BaseModel):
    query: str
    user_id: int
    session_id: Optional[str] = None
    conversation_id: Optional[int] = None


@app.post("/query-pdf/")
async def query_pdf(request: QueryRequest):

    # Maintaining session
    if not request.session_id:  # If no session_id is provided, create a new one
        session_id = str(uuid.uuid4())
    else: # If session_id is provided, use it
        session_id = request.session_id
    session = Session(session_id, request.user_id, request.conversation_id)
    active_sessions[session_id] = session

    # Initializing the agent
    memory = session.memory

    # Maintaing conversation history (memory string) to pass to the agent
    if request.conversation_id:
        # print("User referred to a previous conversation.")
        # The user tries to refer to an a conversation get chat history
        memory_string = get_last_conversation_chat_history(request.conversation_id, request.user_id)
        # print("Memory string is:", memory_string)
    else:
        memory_string = memory.load_memory_variables({})["chat_history"]

    # Explicitly passing memory as context
    user_query = str(request.query)

    inputs = f"'query_context': {memory_string}, 'user_query': {user_query},'session_id': {session.session_id}, 'user_id': {session.user_id}"

    # inputs = {
    #     "query_with_context": query_with_context, 
    #     "session_id": session_id, 
    #     "user_id": session.user_id
    # }
    agent = create_agent(llm, session_id)
    # print("\n\n\n\nAGENT INPUT KEYS\n\n\n")
    # print(agent.input_keys)

    # Run the agent with the query and context
    response = agent.invoke({"input":inputs})
    # print(f"\n🤖 Agent: {response}")
    # print(f"\n🤖 Memory: {memory.buffer}")

    return {"agent_response": response["output"],
            "session_id": session.session_id
        }