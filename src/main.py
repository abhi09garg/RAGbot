# chatbot.py
from retriever import get_retriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

def create_chatbot():
    retriever = get_retriever()

    template = """
    You are a helpful assistant. Use the provided context to answer the question.
    If the answer is not in the context, say "I don't know".
    
    Context:
    {context}
    
    Question: {question}
    Answer:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain
