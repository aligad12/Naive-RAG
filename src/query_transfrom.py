"""Pre-retrieval technique: Multi-Query expansion.

The user's question is rewritten by the LLM into a few alternative search queries,
each phrased the way a CV would phrase it. Every query is searched separately, so a
chunk that the original wording missed can still be found by one of the rewrites.
"""
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

N_VARIANTS = 3

prompt = ChatPromptTemplate.from_template(
"""You help a search system find information inside a person's CV.
The CV has these sections: summary, education, experience, projects, technical skills,
certifications and languages. CV text is short and keyword-heavy (for example
"Expected Graduation: 2027 | GPA: 3.56/4.0" or "PySpark, Apache Kafka, Databricks"),
but users ask in natural language.

Rewrite the question below as {n} different search queries that could match the CV's wording.
Each query must look at the question from a different angle: use synonyms, the keywords
or section names a CV would use, or split a broad question into its parts.
Do not answer the question and do not invent facts.
Output only the queries, one per line, with no numbering and no extra text.

Question: {question}"""
)

# removes list markers the LLM sometimes adds in spite of the instructions: "1.", "2)", "-", "*", "•"
_MARKER = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*")


def generate_queries(question: str, llm, n: int = N_VARIANTS) -> list[str]:
    """Return [original question, variant 1, variant 2, ...]."""
    chain = prompt | llm | StrOutputParser()
    try:
        raw = chain.invoke({"question": question, "n": n})
    except Exception as e:  # rate limit, network, retired model...
        print(f"[multi-query] LLM call failed, using the original question only: {e}")
        return [question]

    variants = []
    for line in raw.splitlines():
        line = _MARKER.sub("", line).strip().strip('"')
        if line and line.lower() != question.lower() and line not in variants:
            variants.append(line)

    # the original question always stays first, so a bad rewrite can never make retrieval worse
    return [question] + variants[:n]