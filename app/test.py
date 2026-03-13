from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import VECTOR_DB_DIR

embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"},   # CPU is fine for retrieval
)

vector_db = FAISS.load_local(
    str(VECTOR_DB_DIR),
    embeddings=embedder,
    allow_dangerous_deserialization=True,
)

print(f"Total documents in DB: {vector_db.index.ntotal}")

# Quick sanity-check query
query = "transformer attention mechanism"
results = vector_db.similarity_search(query, k=3)
print(f"\nTop 3 results for: '{query}'")
for i, doc in enumerate(results):
    print(f"\n[{i+1}] Source: {doc.metadata.get('source', 'unknown')}")
    print(f"     {doc.page_content[:200]}...")
