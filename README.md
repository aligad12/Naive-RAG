# Ask My CV: Naive RAG Application

A Retrieval-Augmented Generation (RAG) app that answers questions about my CV. Answers are generated only from retrieved CV content, and when the CV doesn't contain the answer, the app says so instead of inventing one.

![App screenshot](docs/screenshot.png)

## Features

- Chat interface built with Streamlit, with conversation history
- Answers grounded in retrieved CV chunks, with sources and similarity distance shown for every answer
- Explicit fallback for questions the CV cannot answer
- Local embeddings (no embedding API key needed) and a persistent vector store

## Architecture

![Architecture](docs/architecture.png)

1. **Indexing** (once): load the CV PDF, split it into overlapping chunks, embed each chunk, store the vectors in Chroma.
2. **Question answering** (each question): embed the question, retrieve the top-4 most similar chunks, insert them with the question into a prompt that restricts the model to the provided context, and generate the answer with an LLM.

This is deliberately **naive RAG**. There is no query rewriting, query expansion, HyDE, reranking, contextual compression, hybrid search, or MMR.

## Tech stack

| Component | Choice |
|---|---|
| Language | Python 3.10 |
| Orchestration | LangChain |
| PDF loading | PyPDFLoader |
| Chunking | RecursiveCharacterTextSplitter (600 characters, 100 overlap) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, runs locally) |
| Vector store | Chroma (persisted to `chroma_db/`) |
| LLM | Groq API (`openai/gpt-oss-20b`) |
| UI | Streamlit |

## Project structure

```
cv-naive-rag/
├── app.py                 # Streamlit chat interface
├── src/
│   ├── loader.py          # load the CV PDF
│   ├── splitter.py        # chunking
│   ├── ingest.py          # embed + store in Chroma
│   ├── rag.py             # retrieval + prompt + LLM
│   └── retrieve_test.py   # retrieval-only sanity check
├── data/cv.pdf            # knowledge base (see note below)
├── docs/                  # diagrams, screenshot, evaluation
├── .streamlit/config.toml # UI theme
├── .env.example
└── requirements.txt
```

## Installation

```bash
git clone <your-repo-url>
cd cv-naive-rag

python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate

pip install -r requirements.txt
```

Create a free API key at https://console.groq.com/keys, then:

```bash
copy .env.example .env         # macOS/Linux: cp .env.example .env
```

and set `GROQ_API_KEY` in `.env`. Never commit `.env`.

> **Note:** <the CV in data/cv.pdf is <included / a redacted version / not included. Place your own PDF at data/cv.pdf>>.

## Usage

```bash
# 1. Build the vector index (run once, and again whenever the CV changes)
python -m src.ingest

# 2. Start the app
streamlit run app.py
```

Open http://localhost:8501.

## Evaluation

I tested <N> questions across skills, education, experience, projects, certifications, and questions not covered by the CV. Full results are in [docs/evaluation.md](docs/evaluation.md).

| Category | Result |
|---|---|
| Questions answerable from the CV | <x/y answered correctly> |
| Questions not in the CV | <x/y correctly answered "not found"> |

**Observed limitations of naive RAG:** <e.g. questions phrased very differently from the CV's wording can retrieve the wrong chunk; broad questions such as "summarize everything" are limited to the top-4 chunks>.

## Troubleshooting

- **Model not found (404):** hosted model names get retired. Check https://console.groq.com/docs/models and update the model name in `src/rag.py`.
- **Empty or irrelevant answers:** run `python -m src.ingest` again and confirm it prints `Indexed N chunks` with N above 0.
- **`No module named torchvision`:** `pip install torchvision`.

## License

<MIT or your choice>