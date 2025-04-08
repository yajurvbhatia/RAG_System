from langchain.tools import Tool
from fastapi import HTTPException
import os

from utils import connect_to_db, embedder, active_sessions
from config import OPENAI_API_KEY


if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Define Retrieval Tool
def retrieve(query):
    top_k = 5
    try:
        # Getting query vector
        query_vector = embedder.embed_query(query)
        vector_str = "[" + ",".join([str(x) for x in query_vector]) + "]"

        # Fetching similar vectors
        conn = connect_to_db()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT content, embedding <=> %s::vector AS distance FROM documents ORDER BY distance LIMIT %s",
            (vector_str, top_k)
        )
        similar_vectors = cursor.fetchall()
        
        relevant_info = [item[0] for item in similar_vectors]
        return f"relevant information: {relevant_info}"
        
    except Exception as e:
        print(f"Error in Retrieval tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))

retrieval_tool = Tool(
            name="Retrieval Tool",
            func=retrieve,
            description="Use this tool to retrieve relevant knowledge."
        )

# Function to save conversation to PostgreSQL
def db_save_conversation(session_id: str):
    table_name = 'conversations'

    # Getting Session info
    session = active_sessions.get(session_id)
    user_id = session.user_id
    conversation_id = session.conversation_id
    memory = session.memory

    # All messages in the conversation is the chat history
    chat_history = [message.content for message in memory.chat_memory.messages]

    try:
        conn = connect_to_db()
        cursor = conn.cursor()

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                conversation_id SERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                chat_history TEXT
            );
        """)

        # User tries to refer to a conversation that already exists
        if conversation_id:
            cursor.execute(f"""
                SELECT chat_history FROM {table_name}
                WHERE user_id = %s AND conversation_id = %s;
            """, (user_id, conversation_id))
            row = cursor.fetchone()

            # Check if row exists
            if row:
                # Append new input to existing chat history
                cursor.execute(f"""
                    UPDATE {table_name}
                    SET chat_history = %s
                    WHERE user_id = %s AND conversation_id = %s;
                """, (chat_history, user_id, conversation_id))
        
        # User tries to save the conversation as a new conversation
        else:
            # Insert row with initial input_str
            cursor.execute(f"""
                INSERT INTO {table_name} (user_id, chat_history)
                VALUES (%s, %s);
            """, (user_id, chat_history))

        conn.commit()
        cursor.close()
        conn.close()

        return "Final Answer: Database updated successfully!"

    except Exception as e:
        print(f"Error in db_update_tool: {e}")
        return f"DB update failed: {e}"

db_save_conversation_tool = Tool(
    name="DB Update Tool",
    func=db_save_conversation,
    description=(
        """Use this tool ONLY if the user's query clearly requests saving or updating the conversation history.
        This tool appends a new message (latest user query and agent response) to the chat history 
        in the PostgreSQL database. Use it when the user says something like 'save this', 'log this', or 'remember this'.
        Pass the 36 character length session_id as an input to this tool and nothing else."""
    )
)

# Function to get all conversations for a given user_id
def db_get_user_conversations(user_id: int):
    # print("Input to get user conversations is:", user_id)
    user_id = str(user_id)
    table_name = 'conversations'

    try:
        conn = connect_to_db()
        cursor = conn.cursor()

        # Fetch all conversations for the given user_id
        cursor.execute(f"""
            SELECT conversation_id, chat_history FROM {table_name}
            WHERE user_id = %s;
        """, (user_id))
        conversations = cursor.fetchall()

        # Process and return the results
        cursor.close()
        conn.close()

        # Format the results as a list of dictionaries
        result = [
            {"conversation_id": row[0], "chat_history": row[1]}
            for row in conversations
        ]
        return "Final Answer: " + str(result)

    except Exception as e:
        print(f"Error in db_read_tool: {e}")
        raise HTTPException(status_code=500, detail=f"DB read failed: {e}")

db_get_user_conversations_tool = Tool(
    name="DB Read Tool",
    func=db_get_user_conversations,
    description=(
        """Connects to the PostgreSQL server and retrieves all conversations for a given user ID.
        Args:
        user_id (int): The ID of the user whose conversations are to be retrieved. Do not pass anything else into the function.
        Returns:
        list: A list of conversations for the given user ID.
        """
    )
)
