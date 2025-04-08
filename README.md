# Project Requirements:
1. Active postgres server running
2. OpenAI API Key
3. Serp API Key

# Project setup

- Make a virtual environment
python3 -m venv rag_venv

- Activate it
source rag_venv/bin/activate

- Install dependencies
pip install -r requirements.txt

# Run the server (from the project directory)
uvicorn main:app --reload

# Use PostMan APIs to call it (included in the repo)
1. Index a PDF in vector storage with the endpoint /process-pdf
2. Query the Agent /query-pdf

# Supported query flows (currently):
1. Ask questions about stored document
2. Ask Agent to save the current conversation
3. Ask Agent to fetch previous conversations
4. Continue a conversation with the Agent