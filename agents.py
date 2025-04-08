from langchain.agents import initialize_agent, AgentType
import os

from config import OPENAI_API_KEY
from utils import active_sessions
from agent_tools import retrieval_tool, db_save_conversation_tool, db_get_user_conversations_tool, get_info_from_web


if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

OPTIMIZED_AGENT_PROMPT = """
Assistant Role:
You are a helpful assistant and you have three main tasks to fulfill:
1. Try to answer any knowledge based user questions by summarizing information that you get from the retrieval_tool.
Determine if the retrieved information is relevant to the query or not. If it is not, then use web search instead.
2. Save conversations in the database when prompted by the user, by using the db_save_conversation_tool.
3. Fetch all conversation history for a given user by using db_get_user_conversations_tool.
4. Answer any knowledge based questions using the web_search tool, if the retrieval tool returns irrelevant information.
5. Answer any general inquiries by not using any tool whatsoever, and only using the prior conversation instead.

Tool Instructions:
- retrieval_tool: Clarify the query, retrieve factual info, and summarize it. Only pass the query to this tool, nothing else.
If no relevant information is found, use the web search tool to get information from the internet.
If the retrieval_tool output contains "RETRIEVAL_FAILED", then call get_info_from_web.
- db_save_conversation_tool: Invoke only if the user says something along the lines of "save this",
"log this", or "remember this" (case-insensitive). Use only the provided 36-character session_id - session_id as the input to the tool.
- db_get_user_conversations_tool: When requested, use the user_id to get all conversations with the user.
If none exist, respond "No conversations found for the provided session ID." Always include the conversation_id.
- get_info_from_web: Use this tool, when the retrieve tool, does not have the relevant information that would answer the user query.

Rules:
- All responses must be easy to understand and in a single string.
- If any tool output includes "Final Answer:", stop further actions and respond to the user.
- Use only one tool per turn.
- Ensure clarity, precision, and helpfulness.
"""


def create_agent(llm, session_id):

    # Initialize LangChain Agent
    agent = initialize_agent(
        tools=[retrieval_tool, db_save_conversation_tool, db_get_user_conversations_tool, get_info_from_web],
        llm=llm,
        memory=active_sessions[session_id].memory,
        # agent = AgentType.OPENAI_FUNCTIONS,
        # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_type="structured-chat-zero-shot-react-description",
        # agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={"prefix": OPTIMIZED_AGENT_PROMPT},
        verbose=True,
        max_iterations=3,
        early_stopping_method="generate",
        handle_parsing_errors=True
    )

    return agent