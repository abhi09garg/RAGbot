# config.py
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Put in .env
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4o-mini"
VECTOR_STORE_PATH = "vector_store"
DOCS_PATH = "docs"
