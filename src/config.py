import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "tfidf")  # 推荐 tfidf，部署轻量稳定

DATA_DIR = ROOT_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
PARSED_DIR = DATA_DIR / "parsed"
LABELS_DIR = DATA_DIR / "labels"
STORAGE_DIR = ROOT_DIR / "storage"
REPORTS_DIR = ROOT_DIR / "reports"
