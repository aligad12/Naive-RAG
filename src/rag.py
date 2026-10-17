import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.ingest import get_embeddings, DB_DIR

load_dotenv()  # reads GROQ_API_KEY from .env

db = Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())
retriever = db.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_template(
"""You answer questions about a person's CV using ONLY the context below.
If the context does not contain the answer, reply exactly:
"I couldn't find that information in the CV."
Do not use outside knowledge or guess.

Context:
{context}

Question: {question}
Answer:"""
)

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)  # check Groq's docs for current model names
chain = prompt | llm | StrOutputParser()

def ask(question: str):
    docs = retriever.invoke(question)                      # 1. retrieve
    context = "\n\n".join(d.page_content for d in docs)    # 2. stuff into prompt
    answer = chain.invoke({"context": context, "question": question})  # 3. generate
    return answer, docs