from concurrent.futures import ThreadPoolExecutor

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import ARXIV_INPUT_DIR, DATA_CHUNKS_DIR

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def create_chunks(filepath):
    pdfloader = PyPDFLoader(filepath)
    documents = pdfloader.load()
    chunks = splitter.split_documents(documents)

    chunk_texts = [chunk.page_content for chunk in chunks]
    output_file = DATA_CHUNKS_DIR / f"{filepath.name}.txt"
    if output_file.exists():
        return

    with output_file.open("w") as f:
        for i, chunk in enumerate(chunk_texts):
            f.write(f"--- Chunk {i+1} ---{chunk}\n\n\n")







if __name__ == "__main__":
    DATA_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_list = [path for path in ARXIV_INPUT_DIR.iterdir() if path.suffix == ".pdf"]

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(create_chunks, pdf_list)
