from langchain.tools import Tool
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from fastapi import HTTPException
from langchain.chains import RetrievalQA
from utils import get_or_create_memory

from utils import llm
from config import DB_PASSWORD, OPENAI_API_KEY, VECTOR_STORE_DIR
import psycopg2
import os
import json

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Define RAG Tool (Retrieval + Generation)
def rag_tool(query):
    vector_store_name = "tulip_wiki_article"

    # print("IN RAG TOOL")
    try:
        # Construct the full path to the vector store
        vector_store_path = os.getcwd() + '/' + VECTOR_STORE_DIR + '/' + f"{vector_store_name}_faiss_index"

        # Check if vector store exists
        if not os.path.exists(vector_store_path):
            raise HTTPException(status_code=404, detail="Vector store not found")

        # Load the FAISS index with allow_dangerous_deserialization
        vectorstore = FAISS.load_local(
            vector_store_path, 
            embeddings,
            allow_dangerous_deserialization=True  # Only safe because we created these files
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
        
        result = qa_chain.invoke({"query": query})


        return f"Final Answer: {result["result"]}"
        
    except Exception as e:
        print(f"Error in RAG tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))

rag = Tool(
            name="RAG Tool",
            func=rag_tool,
            description="Use this tool to retrieve and generate responses based on knowledge."
        )


def db_update_tool(input_str: str):
    table_name = 'conversations'
    user_id = 1
    conversation_id = 1
    print("🧠 Entering db_update_tool...")

    try:
        # 1. Connect to DB
        print("🔌 Connecting to PostgreSQL...")
        conn = psycopg2.connect(
            dbname="sample",
            user="postgres",
            password=DB_PASSWORD,
            host="localhost",
            port="5432"
        )
        cursor = conn.cursor()

        # 2. Create table if not exists
        print("🛠️ Creating table if not exists...")
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                conversation_id SERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                chat_history TEXT
            );
        """)

        # 3. Check if row exists
        print("🔎 Checking if row exists...")
        cursor.execute(f"""
            SELECT chat_history FROM {table_name}
            WHERE user_id = %s AND conversation_id = %s;
        """, (user_id, conversation_id))

        row = cursor.fetchone()

        print("input recieved is:", input_str)

        memory = get_or_create_memory(input_str)
        chat_history = [message.content for message in memory.chat_memory.messages]

        if not row:
            # 4. Insert row with initial input_str
            print("➕ Row does not exist. Inserting new row...")
            cursor.execute(f"""
                INSERT INTO {table_name} (conversation_id, user_id, chat_history)
                VALUES (%s, %s, %s);
            """, (conversation_id, user_id, chat_history))
        else:
            # 5. Append new input to existing chat history
            print("📜 Row exists. Appending to chat_history...")
            
            cursor.execute(f"""
                UPDATE {table_name}
                SET chat_history = %s
                WHERE user_id = %s AND conversation_id = %s;
            """, (chat_history, user_id, conversation_id))

        conn.commit()
        cursor.close()
        conn.close()
        print("✅ DB operation complete.")

        return "Database updated successfully!"

    except Exception as e:
        print(f"💥 Error in db_update_tool: {e}")
        return f"DB update failed: {e}"

db_update = Tool(
    name="DB Update Tool",
    func=db_update_tool,
    description=(
        """Use this tool ONLY if the user's query clearly requests saving or updating the conversation history.
        This tool appends a new message (latest user query and agent response) to the chat history 
        in the PostgreSQL database. Use it when the user says something like 'save this', 'log this', or 'remember this'.
        Pass the 36 character length session_id as an input to this tool and nothing else."""
    )
)