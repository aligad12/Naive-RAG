import html
import streamlit as st
from src.rag import ask, TOP_K

st.set_page_config(page_title="Ask my CV", page_icon="📄", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* ---------- base ---------- */
html, body, [class*="css"], button, input, textarea {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
footer, #MainMenu { visibility: hidden; }
.stApp { background: radial-gradient(1200px 600px at 20% -10%, #1B2250 0%, #0D1020 55%); }
.block-container { max-width: 1000px; padding-top: 1.5rem; }

[data-testid="stMain"] p, [data-testid="stMain"] li, [data-testid="stMain"] strong,
[data-testid="stMain"] h1, [data-testid="stMain"] h2,
[data-testid="stMain"] h3, [data-testid="stMain"] h4 { color: #E8ECFA; }

/* ---------- hero ---------- */
.hero {
    background: linear-gradient(120deg, #4338CA 0%, #0E7490 100%);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 22px; padding: 2.2rem 2.4rem; margin-bottom: 1.6rem;
    box-shadow: 0 12px 40px rgba(67, 56, 202, 0.35);
}
.hero h1 { color: #FFFFFF !important; font-size: 3rem; font-weight: 800;
           margin: 0 0 0.4rem 0; letter-spacing: -0.02em; }
.hero p  { color: #D6E4FF !important; font-size: 1.2rem; margin: 0; }
.chips { margin-top: 1rem; }
.chip { display: inline-block; background: rgba(255,255,255,0.16); color: #FFFFFF;
        padding: 0.3rem 0.9rem; border-radius: 999px; font-size: 0.95rem;
        margin-right: 0.5rem; font-weight: 600; }

/* ---------- chat messages ---------- */
[data-testid="stChatMessage"] {
    background: #161B33; border: 1px solid #272F55; border-radius: 18px;
    padding: 1.1rem 1.3rem; margin-bottom: 1rem;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {
    font-size: 1.18rem; line-height: 1.65; color: #E8ECFA !important;
}
[data-testid="stChatMessage"] strong { color: #FFFFFF !important; }

/* ---------- source cards ---------- */
.src { background: #10142A; border: 1px solid #272F55; border-left: 6px solid var(--c);
       border-radius: 12px; padding: 0.9rem 1.1rem; margin-bottom: 0.8rem; }
.src, .src * { color: #DDE3FA !important; }
.src-head { display: flex; gap: 0.6rem; align-items: center; margin-bottom: 0.5rem;
            font-size: 0.95rem; font-weight: 700; }
.badge { background: var(--c); color: #0D1020 !important; padding: 0.15rem 0.7rem;
         border-radius: 999px; font-size: 0.85rem; font-weight: 800; }
.meta { color: #8F9AC9 !important; font-weight: 500; }
.src-body { font-size: 1.02rem; line-height: 1.6; }

/* expander */
[data-testid="stExpander"] { background: #161B33; border: 1px solid #272F55; border-radius: 14px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {
    color: #9FB0FF !important; font-weight: 700; font-size: 1.05rem; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #121739 0%, #0A0D20 100%);
                                   border-right: 1px solid #272F55; width: 340px !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] { color: #E8ECFA !important; }
section[data-testid="stSidebar"] .stButton button {
    background: #1B2250; border: 1px solid #343E7A; border-radius: 12px;
    padding: 0.7rem 0.9rem; font-size: 1rem; font-weight: 600; text-align: left;
    transition: all 0.15s; }
section[data-testid="stSidebar"] .stButton button p { color: #E8ECFA !important; font-size: 1rem; }
section[data-testid="stSidebar"] .stButton button:hover {
    background: #FBBF24; border-color: #FBBF24; }
section[data-testid="stSidebar"] .stButton button:hover p { color: #0D1020 !important; }
section[data-testid="stSidebar"] hr { border-color: #272F55; }

/* ---------- input ---------- */
[data-testid="stChatInput"] { border-radius: 16px; border: 2px solid #6C7BFF; background: #161B33; }
[data-testid="stChatInput"] textarea { color: #E8ECFA !important; font-size: 1.1rem; }
[data-testid="stBottom"] > div { background: transparent; }
</style>
""",
    unsafe_allow_html=True,
)

COLORS = ["#2F3FD8", "#0FA3B1", "#F59E0B", "#E4572E"]

EXAMPLES = [
    "What are the technical skills?",
    "Where did he study and what was the degree?",
    "Summarize the project experience.",
    "Which certifications are listed?",
    "What is the salary expectation?",  # not in CV: tests the fallback
]

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(results):
    with st.expander(f"Sources: {len(results)} chunks retrieved"):
        if not results:
            st.warning("No chunks were retrieved.")
        for i, (doc, dist) in enumerate(results, 1):
            page = doc.metadata.get("page", 0) + 1
            color = COLORS[(i - 1) % len(COLORS)]
            body = html.escape(doc.page_content).replace("\n", "<br>")
            st.markdown(
                f"<div class='src' style='--c:{color}'>"
                f"<div class='src-head'><span class='badge'>Chunk {i}</span>"
                f"<span class='meta'>page {page} · distance {dist:.3f} (lower = closer)</span></div>"
                f"<div class='src-body'>{body}</div></div>",
                unsafe_allow_html=True,
            )


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### 💡 Try a question")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown("### ⚙️ Pipeline")
    st.caption(
        f"**Embeddings:** all-MiniLM-L6-v2\n\n**Vector store:** Chroma\n\n"
        f"**Retrieval:** top-{TOP_K} similarity\n\n**LLM:** Groq"
    )

# ---------- hero ----------
st.markdown(
    "<div class='hero'><h1>📄 Ask my CV</h1>"
    "<p>Answers come only from the CV. If something isn't in it, the app says so.</p>"
    "<div class='chips'><span class='chip'>Retrieval-Augmented Generation</span>"
    "<span class='chip'>Chroma</span><span class='chip'>LangChain</span></div></div>",
    unsafe_allow_html=True,
)

# ---------- history ----------
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "📄"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])
        if msg.get("sources") is not None:
            render_sources(msg["sources"])

# ---------- input ----------
question = st.chat_input("Ask about skills, education, projects, experience...")
question = question or st.session_state.pop("pending", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(question)

    with st.chat_message("assistant", avatar="📄"):
        try:
            with st.spinner("Searching the CV..."):
                answer, results = ask(question)
            st.write(answer)
            render_sources(results)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": results}
            )
        except Exception as e:
            st.error(
                "Could not generate an answer. Check that GROQ_API_KEY is set in .env, "
                "that the model name is still available, and that the index exists "
                f"(python -m src.ingest).\n\nDetails: {e}"
            )