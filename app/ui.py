import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import VECTOR_DB_DIR
from query import get_answer

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
st.caption("Ask questions about ArXiv research papers — answers grounded in real papers")

query = st.text_input("Enter your query:", placeholder="e.g. How does BERT use attention?")
top_k = st.slider("Number of sources to retrieve", min_value=1, max_value=10, value=5)
search = st.button("Search")

# ── Search ─────────────────────────────────────────────────────────────────
if search:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Retrieving papers and generating answer..."):
            try:
                db = load_vector_db()
                answer, sources = get_answer(query.strip(), db, top_k=top_k)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        # ── Answer ─────────────────────────────────────────────────────────
        st.subheader("Answer")
        st.write(answer)

        # ── Sources ────────────────────────────────────────────────────────
        st.divider()
        st.subheader(f"📄 Sources ({len(sources)} papers retrieved)")

        for i, doc in enumerate(sources, start=1):
            source = doc.metadata.get("source", "Unknown")
            paper_id = source.replace(".pdf.txt", "").replace(".txt", "")

            with st.expander(f"Reference {i} — `{paper_id}`"):
                st.markdown(doc.page_content)
                st.markdown(f"🔗 [View on ArXiv](https://arxiv.org/abs/{paper_id})")
