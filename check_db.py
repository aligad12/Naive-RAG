# check_db.py
from langchain_chroma import Chroma
from src.ingest import get_embeddings, DB_DIR

db = Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())
print("chunks stored:", db._collection.count())

for doc, score in db.similarity_search_with_score("when will ali graduate", k=4):
    print(f"\n[distance {score:.3f}] page {doc.metadata.get('page')}")
    print(doc.page_content)