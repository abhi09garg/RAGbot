# chatbot.py
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from retriever import get_retriever
from config import CHAT_MODEL, OPENAI_API_KEY

def create_chatbot():
    retriever = get_retriever()
    llm = ChatOpenAI(model=CHAT_MODEL, openai_api_key=OPENAI_API_KEY)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
