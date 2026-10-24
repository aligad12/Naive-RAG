"""Post-retrieval technique: cross-encoder reranking.

The vector search returns candidates ranked by embedding distance. A cross-encoder
re-scores each candidate by reading the question and the chunk TOGETHER, which is
more accurate than comparing two separately computed embeddings. We keep only the
best top_n for the prompt.
"""
from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # small, runs locally on CPU, no API key

_model = None


def get_reranker():
    """Load the model once (first call downloads it), then reuse it."""
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def rerank(question: str, docs: list, top_n: int = 4):
    """Return the top_n docs as (Document, score) pairs, best first. Higher score = more relevant."""
    if not docs:
        return []
    scores = get_reranker().predict([(question, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    return [(doc, float(score)) for doc, score in ranked[:top_n]]