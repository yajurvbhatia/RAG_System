from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

import os
import tempfile 
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI  # or any other LLM you prefer
import os

from rag import embeddings, llm

app = FastAPI()

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
VECTOR_STORE_DIR = "vector_stores"

# Create directory for vector stores if it doesn't exist
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

@app.post("/process-pdf/")
async def process_pdf(file: UploadFile = File(...)):
    try:
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
        # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        # Save vector store locally
        filename = file.filename.replace(".pdf", "")
        save_path = os.path.join(VECTOR_STORE_DIR, f"{filename}_faiss_index")
        vectorstore.save_local(save_path)
        
        return {
            "message": "PDF processed successfully",
            "vector_store_path": save_path,
            "num_documents": len(splits)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


# Request model
class QueryRequest(BaseModel):
    query: str
    vector_store_name: str


@app.post("/query-pdf/")
async def query_pdf(request: QueryRequest):
    try:
        # Construct the full path to the vector store
        vector_store_path = os.path.join(
            VECTOR_STORE_DIR, 
            f"{request.vector_store_name}_faiss_index"
        )
        
        cwd = os.getcwd()
        vector_store_path = cwd + vector_store_path
        # Load the FAISS index with allow_dangerous_deserialization
        vectorstore = FAISS.load_local(
            vector_store_path, 
            embeddings,
            allow_dangerous_deserialization=True  # Only safe because we created these files
        )
        # Check if vector store exists
        if not os.path.exists(vector_store_path):
            raise HTTPException(status_code=404, detail="Vector store not found")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
        
        # Execute the query
        result = qa_chain({"query": request.query})
        
        return {
            "query": request.query,
            "answer": result["result"],
            "source_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": doc.metadata.get("score", None)  # Include similarity score if available
                } 
                for doc in result["source_documents"]
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


