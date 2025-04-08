from langchain.tools import Tool
from fastapi import HTTPException
import os
# from langchain.tools import tool
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Annotated
from typing import List

from utils import connect_to_db, embedder, active_sessions
from config import OPENAI_API_KEY, SERP_API_KEY
import requests

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

class RetrievalInput(BaseModel):
    query: str = Field(description="The search query")

class RetrievalOutput(BaseModel):
    success: bool = Field(description="True if relevant documents are found, False if no relevant documents are found.")
    info: str = Field(description="Aggregated information from all the relevant documents found.")

def retrieval_tool_fn(query: str):
    top_k = 5
    try:
        query_vector = embedder.embed_query(query)
        vector_str = "[" + ",".join([str(x) for x in query_vector]) + "]"

        conn = connect_to_db()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT content, embedding <=> %s::vector AS distance FROM documents ORDER BY distance LIMIT %s",
            (vector_str, top_k)
        )
        similar_vectors = cursor.fetchall()

        relevant_info = ""
        for item in similar_vectors:
            if item[1] < 0.20:
                relevant_info += str(item[0])

        if not relevant_info:
            return RetrievalOutput(success=False, info="")

        return RetrievalOutput(success=True, info=relevant_info)

    except Exception as e:
        print(f"Error in Retrieval tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))

retrieval_tool = StructuredTool.from_function(
    name="retrieval_tool",
    description="Searches through documents for a query string.",
    func=retrieval_tool_fn,
    args_schema=RetrievalInput
)


class SaveConversationInput(BaseModel):
    session_id: str = Field(description="The id of the ongoing session.")

class SaveConversationOutput(BaseModel):
    success: bool = Field(description="True if conversation was saved properly, False if there was an error in saving the conversation.")
    error: str = Field(description="Detail of the error that occured while saving the conversation, Empty if no error occured.")

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

        return SaveConversationOutput(success=True, error="")

    except Exception as e:
        print(f"Error in db_update_tool: {e}")
        return SaveConversationOutput(success=False, error=e)

db_save_conversation_tool = StructuredTool.from_function(
    name="DB Update Tool",
    func=db_save_conversation,
    description="""Use this tool ONLY if the user's query clearly requests saving or updating the conversation history.
        This tool appends a new message (latest user query and agent response) to the chat history 
        in the PostgreSQL database. Use it when the user says something like 'save this', 'log this', or 'remember this'.
        Pass the 36 character length session_id as an input to this tool and nothing else.""",
    args_schema=SaveConversationInput
)

class GetConversationsInput(BaseModel):
    user_id: str = Field(description="The id of the user whose conversations need to be fetched.")

class Conversation(BaseModel):
    conversation_id: int = Field(description="The id of the conversation the agent has had with the user.")
    chat_history: str = Field(description="The history of the conversation as a string.")

class GetConversationsOutput(BaseModel):
    success: bool = Field(description="True if prior conversations for the user are found, False if none are found.")
    user_id: str = Field(description="The id of the user whose conversations need to be fetched.")
    conversations: List[Conversation]
    error: str = Field(description="Detail of the error that occured while getting the conversations, Empty if no error occured.")

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

        conversations = [Conversation(conversation_id=row[0], chat_history=row[1]) for row in conversations]
        
        return GetConversationsOutput(success=True, user_id=user_id, conversations=conversations, error="")

    except Exception as e:
        print(f"Error in db_read_tool: {e}")
        return GetConversationsOutput(success=False, user_id=user_id, conversations=[], error=e)


db_get_user_conversations_tool = StructuredTool.from_function(
    name="DB Read Tool",
    func=db_get_user_conversations,
    description="""Connects to the PostgreSQL server and retrieves all conversations for a given user ID.
        Args:
        user_id (int): The ID of the user whose conversations are to be retrieved. Do not pass anything else into the function.
        Returns:
        list: A list of conversations for the given user ID.
        """,
    args_schema=GetConversationsInput
)

class WebSearchInput(BaseModel):
    query: str = Field(description="The search query used to fetch information from the internet.")

class WebSearchOutput(BaseModel):
    success: bool = Field(description="True if the web search is successful, False if it is not.")
    result: str = Field(description="Aggregated information from the internet relevant to the search query.")
    error: str = Field(description="Details of an error that occured, Empty if no error occurs.")

# Function to search the web for information
def web_search(query: str):
    try:
        url = "https://serpapi.com/search"
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERP_API_KEY,
            "start": 0,
            "num": 5
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        response = response.json()

        organic_results = response["organic_results"]
        all_snippets = '.'.join([str(result["snippet"]) for result in organic_results])
        return WebSearchOutput(success=True, result=all_snippets, error="")
    
    except Exception as e:
        return WebSearchOutput(success=False, result="", error=e)

get_info_from_web = Tool(
    name="Web Search Tool",
    func=web_search,
    description="""Query the internet to get multiple short descriptions with information from different sources on the web.""",
    args_schema=WebSearchInput
)
