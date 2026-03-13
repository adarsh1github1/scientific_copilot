from pathlib import Path

from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import DATA_CHUNKS_DIR, VECTOR_DB_DIR

# ── Config ─────────────────────────────────────────────────────────────────
FILE_BATCH_SIZE = 200          # process N chunk-files at a time (~low RAM)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# ── Embedder ───────────────────────────────────────────────────────────────
embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "mps"},       # Apple Silicon GPU
    encode_kwargs={"batch_size": 128},
)

# ── Helpers ────────────────────────────────────────────────────────────────
def parse_chunks(txt_file: Path) -> list[Document]:
    """Parse one chunk .txt file into LangChain Documents."""
    try:
        content = txt_file.read_text(encoding="utf-8", errors="ignore")
        return [
            Document(page_content=chunk.strip(), metadata={"source": txt_file.name})
            for chunk in content.split("--- Chunk")
            if chunk.strip()
        ]
    except Exception as e:
        print(f"  [WARN] Skipping {txt_file.name}: {e}")
        return []

# ── Main ingestion loop ────────────────────────────────────────────────────
txt_files = sorted(DATA_CHUNKS_DIR.glob("*.txt"))
total_files = len(txt_files)
print(f"Found {total_files} chunk files in {DATA_CHUNKS_DIR}")
print(f"Processing in batches of {FILE_BATCH_SIZE} files\n")

vector_db = None
total_docs = 0

for batch_start in range(0, total_files, FILE_BATCH_SIZE):
    batch_files = txt_files[batch_start: batch_start + FILE_BATCH_SIZE]
    batch_num = batch_start // FILE_BATCH_SIZE + 1
    total_batches = (total_files + FILE_BATCH_SIZE - 1) // FILE_BATCH_SIZE

    # Parse this batch into Documents
    batch_docs = []
    for f in batch_files:
        batch_docs.extend(parse_chunks(f))

    if not batch_docs:
        continue

    total_docs += len(batch_docs)
    print(f"  Batch {batch_num}/{total_batches} | files: {len(batch_files)} | "
          f"chunks: {len(batch_docs)} | total so far: {total_docs}")

    # Embed and add to FAISS
    if vector_db is None:
        vector_db = FAISS.from_documents(batch_docs, embedding=embedder)
    else:
        vector_db.add_documents(batch_docs)

    # Free RAM immediately after each batch
    del batch_docs

# ── Save ───────────────────────────────────────────────────────────────────
if vector_db is not None:
    vector_db.save_local(str(VECTOR_DB_DIR))
    print(f"\n✅ FAISS index saved to {VECTOR_DB_DIR}")
    print(f"   Total documents indexed: {total_docs}")
else:
    print("❌ No documents were indexed — check DATA_CHUNKS_DIR path.")
