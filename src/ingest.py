from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.loader import load_cv
from src.splitter import split_docs

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_DIR = "chroma_db"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name = EMBED_MODEL)

def build_index():
    chunks = split_docs(load_cv())
    Chroma.from_documents(chunks, get_embeddings(), persist_directory = DB_DIR)
    print(f"indexd {len(chunks)} chunks")


if __name__ == "__main__":
    build_index()