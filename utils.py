import  config
from langchain_openai import OpenAIEmbeddings
import os
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory
import psycopg2

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

llm = ChatOpenAI(model_name="gpt-4o-mini")

embedder = OpenAIEmbeddings()

active_sessions = {} # structure {ss_id: session}


class Session:
    
    def __init__(self, session_id, user_id, conversation_id):
        self.session_id = session_id
        self.user_id = user_id
        self.memory = self.get_or_create_memory()
        self.conversation_id = conversation_id

    def get_or_create_memory(self):
        if self.session_id not in active_sessions:
            # print("CREATING NEW MEMORY FOR:", repr(self.session_id))
            memory = ConversationSummaryMemory(
                llm=llm,
                memory_key="chat_history",
                return_messages=True
            )
        else:
            # print("USING EXISTING MEMORY FOR:", repr(self.session_id))
            memory = active_sessions[self.session_id].memory
        
        return memory


def connect_to_db():
    try:
        conn = psycopg2.connect(dbname=config.DB_NAME, user=config.DB_USER, password=config.DB_PASSWORD, host=config.DB_HOST, port=config.DB_PORT)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None


def save_vectors(conn, chunks, vectors):
# Create the table if it does not exist
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding VECTOR(1536)
        );
    """)
    conn.commit()

    for chunk, vector in zip(chunks, vectors):
        cursor.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (chunk, vector)
        )
    conn.commit()
    # print("Vectors saved successfully to PGVector Database!")


def get_last_conversation_chat_history(conversation_id, user_id):
    table_name = 'conversations'

    conn = connect_to_db()
    cursor = conn.cursor()

    # Return no chat history by default
    result = ""

    if user_id and conversation_id:
        # Check if entry exists in DB
        cursor.execute(f"""
                       SELECT chat_history FROM {table_name}
                       WHERE user_id = %s AND conversation_id = %s;
                       """, (user_id, conversation_id))
        row = cursor.fetchone()

        if row:
            result = row[0]

    return result