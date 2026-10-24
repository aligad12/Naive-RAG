import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.ingest import get_embeddings, DB_DIR
from src.query_transfrom import generate_queries
from src.rerank import rerank

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

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)  # check Groq's docs for current model names
chain = prompt | llm | StrOutputParser()

TOP_K = 4
CANDIDATES_PER_QUERY = 6

def naive_retrieve(question: str):
    """The original retrieval (plain top-k similarity). Kept as a baseline for src/compare.py."""
    return db.similarity_search_with_score(question, k=TOP_K)

def retrieve(question: str):
    """Multi-Query (pre-retrieval) -> vector search -> cross-encoder rerank (post-retrieval)."""
    # PRE-RETRIEVAL: original question + LLM-written variants
    queries = generate_queries(question, llm)

    # search with every query, merge into one pool without duplicates
    pool = {}
    for q in queries:
        for doc, dist in db.similarity_search_with_score(q, k=CANDIDATES_PER_QUERY):
            key = (doc.metadata.get("page"), doc.page_content)
            if key not in pool or dist < pool[key][1]:
                pool[key] = (doc, dist)

    candidates = []
    for doc, dist in pool.values():
        doc.metadata["distance"] = dist  # keep the vector distance so the UI can show it
        candidates.append(doc)

    # POST-RETRIEVAL: rerank against the ORIGINAL question and keep the best TOP_K
    results = rerank(question, candidates, top_n=TOP_K)
    return results, {"queries": queries, "candidates": len(candidates)}

def ask(question: str):
    results,info = retrieve(question)   # 1. retrieve
    context = "\n\n".join(doc.page_content for doc, _ in results)  # 2. stuff into prompt
    answer = chain.invoke({"context": context, "question": question})  # 3. generate
    return answer, results,info   # results = list of (Document, distance)