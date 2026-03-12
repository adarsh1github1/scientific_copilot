from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
ARXIV_INPUT_DIR = PROJECT_ROOT / "arxiv_data" / "2003"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"

