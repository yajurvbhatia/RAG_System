from langchain.agents import initialize_agent, AgentType
import os

from config import OPENAI_API_KEY
from utils import active_sessions
from agent_tools import retrieval_tool, db_save_conversation_tool, db_get_user_conversations_tool


if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

OPTIMIZED_AGENT_PROMPT = """
Assistant Role:
You are a helpful assistant and you have three main tasks to fulfill:
1. Answer any knowledge based user questions by summarizing information that you get from the retrieval_tool.
2. Save conversations in the database when prompted by the user, by using the db_save_conversation_tool.
3. Fetch all conversation history for a given user by using db_get_user_conversations_tool.
4. Answer any general inquiries by not using any tool whatsoever, and only using the prior conversation instead.

Tool Instructions:
- retrieval_tool: Clarify the query, retrieve factual info, and summarize it.
If no relevant information is found, reply "I'm unable to answer that due to insufficient information."
- db_save_conversation_tool: Invoke only if the user says something along the lines of "save this",
"log this", or "remember this" (case-insensitive). Use only the provided 36-character session_id - session_id as the input to the tool.
- db_get_user_conversations_tool: When requested, use the user_id to get all conversations with the user.
If none exist, respond "No conversations found for the provided session ID." Always include the conversation_id.

Rules:
- All responses must be easy to understand and in a single string.
- If any tool output includes "Final Answer:", stop further actions and respond to the user.
- Use only one tool per turn.
- Ensure clarity, precision, and helpfulness.
"""


def create_agent(llm, session_id):

    # Initialize LangChain Agent
    agent = initialize_agent(
        tools=[retrieval_tool, db_save_conversation_tool, db_get_user_conversations_tool],
        llm=llm,
        memory=active_sessions[session_id].memory,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={"prefix": OPTIMIZED_AGENT_PROMPT},
        verbose=True,
        max_iterations=2,
        early_stopping_method="generate",
        handle_parsing_errors=True
    )

    return agent