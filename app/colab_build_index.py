# ============================================================
# Scientific Copilot — Build FAISS Index on Google Colab
# ============================================================
# HOW TO USE:
#   1. Open colab.research.google.com → New notebook
#   2. Runtime → Change runtime type → T4 GPU
#   3. Copy-paste each cell (separated by # --- CELL --- below)
# ============================================================


# --- CELL 1: Mount Google Drive ---
from google.colab import drive
drive.mount('/content/drive')


# --- CELL 2: Install dependencies ---
# Run this once — takes ~1-2 min
import subprocess
subprocess.run([
    "pip", "install", "-q",
    "langchain",
    "langchain-community",
    "langchain-huggingface",
    "faiss-cpu",          # faiss-gpu has broken builds on Colab; CPU is fine here
    "sentence-transformers",
    "pypdf",
])
print("✅ Dependencies installed")


# --- CELL 3: Configure paths ---
import os

# ⚠️ UPDATE THIS to where you uploaded chunks.zip in Google Drive
GDRIVE_ZIP_PATH  = "/content/drive/MyDrive/scientific_copilot/chunks.zip"
GDRIVE_INDEX_DIR = "/content/drive/MyDrive/scientific_copilot/vector_db"

CHUNKS_DIR  = "/content/chunks"
VECTOR_DIR  = "/content/vector_db"
FILE_BATCH  = 500   # how many chunk FILES to parse at once (threading)
EMBED_BATCH = 5000  # how many DOCUMENTS to embed + add to FAISS at once (GPU)

os.makedirs(CHUNKS_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(GDRIVE_INDEX_DIR, exist_ok=True)

print(f"Chunks dir : {CHUNKS_DIR}")
print(f"Vector dir : {VECTOR_DIR}")
print(f"GDrive save: {GDRIVE_INDEX_DIR}")


# --- CELL 4: Unzip chunk files ---
import zipfile

print("Unzipping chunks...")
with zipfile.ZipFile(GDRIVE_ZIP_PATH, "r") as z:
    z.extractall(CHUNKS_DIR)

chunk_files = sorted([
    os.path.join(root, f)
    for root, _, files in os.walk(CHUNKS_DIR)
    for f in files if f.endswith(".txt")
])
print(f"✅ Found {len(chunk_files)} chunk files")


# --- CELL 5: Load embedder (downloads ~130MB model once) ---
from langchain_huggingface import HuggingFaceEmbeddings

embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cuda"},    # T4 GPU on Colab
    encode_kwargs={"batch_size": 512},  # T4 has 16GB VRAM — push it hard
)
print("✅ Embedder loaded on CUDA")


# --- CELL 6: Parse ALL files in parallel (CPU I/O — fast) ---
# NOTE: Only file PARSING is parallelized. GPU embedding must stay
# sequential — CUDA is a single device and is already fully utilized
# internally via batch_size=512.
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from langchain.docstore.document import Document

def parse_file(filepath):
    try:
        content = Path(filepath).read_text(encoding="utf-8", errors="ignore")
        return [
            Document(
                page_content=chunk.strip(),
                metadata={"source": os.path.basename(filepath)}
            )
            for chunk in content.split("--- Chunk")
            if chunk.strip()
        ]
    except Exception as e:
        print(f"  [SKIP] {filepath}: {e}")
        return []

print(f"Parsing {len(chunk_files)} files in parallel...")
all_docs = []
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(parse_file, f): f for f in chunk_files}
    for i, future in enumerate(as_completed(futures)):
        all_docs.extend(future.result())
        if (i + 1) % 1000 == 0:
            print(f"  Parsed {i+1}/{len(chunk_files)} files | docs so far: {len(all_docs)}")

print(f"\n✅ Parsed {len(chunk_files)} files → {len(all_docs):,} total chunks")


# --- CELL 7: Embed chunks into FAISS (GPU — sequential but fully optimised) ---
# The embedder uses batch_size=512 internally so the T4 is saturated.
# Parallelising this would cause CUDA race conditions — don't do it.
from langchain_community.vectorstores import FAISS

vector_db = None
total_batches = (len(all_docs) + EMBED_BATCH - 1) // EMBED_BATCH

for i in range(0, len(all_docs), EMBED_BATCH):
    batch = all_docs[i : i + EMBED_BATCH]
    batch_num = i // EMBED_BATCH + 1
    print(f"  Embedding batch {batch_num}/{total_batches} | {len(batch):,} chunks")

    if vector_db is None:
        vector_db = FAISS.from_documents(batch, embedding=embedder)
    else:
        vector_db.add_documents(batch)

print(f"\n✅ Indexed {len(all_docs):,} chunks into FAISS")


# --- CELL 8: Save FAISS index to Google Drive ---
import shutil

vector_db.save_local(VECTOR_DIR)

for fname in ["index.faiss", "index.pkl"]:
    shutil.copy2(os.path.join(VECTOR_DIR, fname),
                 os.path.join(GDRIVE_INDEX_DIR, fname))
    print(f"  Saved → {os.path.join(GDRIVE_INDEX_DIR, fname)}")

print("\n✅ FAISS index saved to Google Drive!")
print(f"   Download from: {GDRIVE_INDEX_DIR}")
print("   → index.faiss  &  index.pkl")
print("\nPlace them in your local: vector_db/")
