# ingest.py
import os
from datasets import load_dataset
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Path for vector store
VECTOR_STORE_PATH = "vector_store"

def main():
    # 1. Load dataset (SQuAD as example)
    dataset = load_dataset("squad")

    train_data = dataset["train"].select(range(50))
    # Convert to a list of dicts (context + question)
    docs = []
    for item in train_data:
        text = f"Question: {item['question']}\nContext: {item['context']}\nAnswer: {item['answers']['text']}"
        docs.append({"text": text})

    # 2. Convert to LangChain docs
    from langchain.schema import Document
    documents = [Document(page_content=d["text"]) for d in docs]

    # 3. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)

    # 4. Create embeddings
    embeddings = OpenAIEmbeddings()

    # 5. Create FAISS vector store
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # Ensure folder exists
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    # 6. Save locally
    vectorstore.save_local(VECTOR_STORE_PATH)
    print(f"✅ Vector store created at: {VECTOR_STORE_PATH}")

if __name__ == "__main__":
    main()
