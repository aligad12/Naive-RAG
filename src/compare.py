"""Side-by-side check: naive retrieval vs Multi-Query + rerank.

Run:  python -m src.compare
"""
from src.rag import naive_retrieve, retrieve

QUESTIONS = [
    "When will he graduate?",
    "What did he build with Kafka?",
    "Which projects use machine learning?",
    "Summarize the project experience.",
    "Did he win any awards?",
    "How good is his English?",
]


def label(doc):
    page = doc.metadata.get("page", 0) + 1
    text = doc.page_content.strip().replace("\n", " ")[:75]
    return f"p{page} | {text}".encode("ascii", "ignore").decode()  # safe for any Windows console


for q in QUESTIONS:
    naive = naive_retrieve(q)
    improved, info = retrieve(q)
    print("=" * 100)
    print("Q:", q)
    print("  extra queries:", info["queries"][1:])
    print(f"  candidates after merging all queries: {info['candidates']}")
    print("  NAIVE top-4 (vector distance, lower = closer):")
    for doc, dist in naive:
        print(f"    [{dist:.3f}] {label(doc)}")
    print("  MULTI-QUERY + RERANK top-4 (rerank score, higher = better):")
    for doc, score in improved:
        print(f"    [{score:.3f}] {label(doc)}")