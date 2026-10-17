from langchain_chroma import Chroma
from src.ingest import get_embeddings, DB_DIR

db = Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())

while True:
    q = input("Question: ")
    for doc, score in db.similarity_search_with_score(q, k=4):
        print(f"\n[distance {score:.3f}] page {doc.metadata['page']}")
        print(doc.page_content)