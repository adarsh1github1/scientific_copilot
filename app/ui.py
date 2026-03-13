import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import VECTOR_DB_DIR

# ── Load vector DB once (cached across reruns) ─────────────────────────────
@st.cache_resource
def load_vector_db():
    embedder = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"device": "mps"},
    )
    db = FAISS.load_local(
        str(VECTOR_DB_DIR),
        embeddings=embedder,
        allow_dangerous_deserialization=True,
    )
    return db


# ── Page ───────────────────────────────────────────────────────────────────
st.title("🔬 Scientific Copilot")
st.caption("Search ArXiv research papers using natural language")

query = st.text_input("Enter your query:", placeholder="e.g. How does BERT use attention?")
top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
search = st.button("Search")

# ── Search ─────────────────────────────────────────────────────────────────
if search:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching..."):
            try:
                db = load_vector_db()
                results = db.similarity_search_with_score(query.strip(), k=top_k)
            except Exception as e:
                st.error(f"Failed to load vector DB: {e}")
                st.stop()

        st.markdown(f"**{len(results)} results for:** *{query}*")
        st.divider()

        for i, (doc, score) in enumerate(results, start=1):
            source = doc.metadata.get("source", "Unknown paper")
            # Strip .pdf.txt suffix for cleaner display
            paper_id = source.replace(".pdf.txt", "").replace(".txt", "")

            with st.expander(f"Result {i} — `{paper_id}`  (score: {score:.4f})", expanded=i == 1):
                st.markdown(doc.page_content)
                st.markdown(f"**📄 Reference:** [{paper_id}](https://arxiv.org/abs/{paper_id})")
