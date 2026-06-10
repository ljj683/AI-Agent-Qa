from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from pathlib import Path
import shutil

from src.rag_engine import RAGEngine
from src.config import RAW_DOCS_DIR

app = FastAPI(title="AI-Agent-Qa API", version="2.0.0")
rag = RAGEngine()


class AskRequest(BaseModel):
    question: str
    history: list = []


@app.get("/")
def index():
    return {
        "message": "AI-Agent-Qa API 已启动",
        "features": [
            "上传文档问答", "PDF/Word文档解析", "文本分块", "TF-IDF向量化",
            "余弦相似度检索", "RAG问答", "文本分类", "神经网络", "多文档问答"
        ]
    }


@app.post("/classify")
def classify(req: AskRequest):
    label = rag.classify_question(req.question)
    return {"question": req.question, "label": label}


@app.post("/ask")
def ask(req: AskRequest):
    return rag.ask(req.question, req.history)


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """可选接口：上传文档到 raw_docs/uploaded。Streamlit 上传问答不依赖该接口。"""
    upload_dir = RAW_DOCS_DIR / "uploaded"
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / Path(file.filename).name
    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"message": "上传成功", "path": str(save_path)}
