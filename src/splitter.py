from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_docs(docs, chunk_size=600, chunk_overlap= 100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    return splitter.split_documents(docs)
