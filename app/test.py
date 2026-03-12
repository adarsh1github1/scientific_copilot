from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import VECTOR_DB_DIR


vector_db = Chroma(persist_directory=str(VECTOR_DB_DIR))
doc_count = vector_db._collection.count()

print(f"Vector DB path: {VECTOR_DB_DIR}")
print(f"Total documents in DB: {doc_count}")

# Uncomment the block below once the local embedding model is available.
# embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
# vector_db = Chroma(persist_directory=str(VECTOR_DB_DIR), embedding_function=embedder)
# retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# query = "Explain the transormers model architecture"

# results = retriever.get_relevant_documents(query)

# print(results)

# for i, doc in enumerate(results):
#     print(f"\nResult {i+1}: ")
#     print(doc.page_content[:500])
#     print(f"--- Source: {doc.metadata['source']}")
