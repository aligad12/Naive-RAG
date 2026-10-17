from langchain_community.document_loaders import PyPDFLoader

def load_cv(path="data\cv.pdf"):
    loader = PyPDFLoader(path)
    docs = loader.load()
    return docs

if __name__ == "__main__":
    docs = load_cv()
    print(len(docs), "pages")
    print(docs[0].page_content[:500])
    print(docs[0].metadata)