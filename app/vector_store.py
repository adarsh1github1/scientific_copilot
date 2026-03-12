# the following code let to errors hence commenting it out.
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import DATA_CHUNKS_DIR, VECTOR_DB_DIR

#load the chunks
#create doucment from the chunks ,then append to doc list
#create vectordb from the doc list

embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "mps"},      # Apple Silicon GPU
    encode_kwargs={"batch_size": 128}    # Larger batches
)

docs = []
def process_file(txt_file):
    try:
        # print("The file being processed is : ", txt_file)
        with txt_file.open("r") as f:
            content = f.read()

        chunks = content.split("--- Chunk")

        return [
            Document(page_content=chunk.strip(), metadata={"source": txt_file.name})
            for chunk in chunks
            if chunk.strip()
        ]

    except Exception as e:
        print(f"Error processing {txt_file} : {e}")
        return []


with ThreadPoolExecutor() as executor:
    txt_files = sorted(DATA_CHUNKS_DIR.glob("*.txt"))
    print(f"Loading {len(txt_files)} chunk files from {DATA_CHUNKS_DIR}")
    futures = [executor.submit(process_file, txt_file) for txt_file in txt_files]

    for future in as_completed(futures):
        docs.extend(future.result())

print("Total chunks found is : ", len(docs))

vector_db = FAISS.from_documents(docs[:5000], embedding=embedder)

for i in range(5000, len(docs), 5000):
    vector_db.add_documents(docs[i:i+5000])

vector_db.save_local(str(VECTOR_DB_DIR))



print("Vector Store  in Chroma DB created!")
