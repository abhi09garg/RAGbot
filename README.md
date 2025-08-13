# RAGbot

🤖 **RAGbot** is a Retrieval-Augmented Generation (RAG) chatbot built using LangChain, FAISS, and OpenAI embeddings. It can answer questions based on a custom knowledge base.

---

## Features

- Ingests documents or datasets (e.g., SQuAD) and converts them into vector embeddings.
- Stores embeddings in a FAISS vector store.
- Uses OpenAI LLMs to generate answers based on retrieved relevant documents.
- Fully modular: easy to swap datasets, LLMs, or vector stores.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/RAGbot.git
cd RAGbot


Setting up your python Environment:

python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
pip install --upgrade pip
pip install -r requirements.txt
