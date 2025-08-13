from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def get_retriever():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever()
