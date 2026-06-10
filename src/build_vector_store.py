import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import STORAGE_DIR, EMBEDDING_MODEL, VECTOR_BACKEND


def load_chunks(path: Path = STORAGE_DIR / "chunks.jsonl"):
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_with_sentence_transformers(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(embeddings), {"backend": "sentence_transformer", "model_name": EMBEDDING_MODEL}


def build_with_tfidf(texts):
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=1,
        max_df=0.95
    )
    matrix = vectorizer.fit_transform(texts)
    joblib.dump(vectorizer, STORAGE_DIR / "tfidf_vectorizer.joblib")
    # 保存稀疏矩阵
    joblib.dump(matrix, STORAGE_DIR / "tfidf_matrix.joblib")
    return None, {"backend": "tfidf", "model_name": "TfidfVectorizer-char-2-4"}


def build_vector_store():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()
    texts = [item["text"] for item in chunks]
    if not texts:
        raise RuntimeError("没有可向量化的文本块，请先运行 document_parser.py 和 text_preprocess.py")

    meta = None
    if VECTOR_BACKEND in ["auto", "sentence_transformer"]:
        try:
            embeddings, meta = build_with_sentence_transformers(texts)
            np.save(STORAGE_DIR / "embeddings.npy", embeddings)
        except Exception as e:
            if VECTOR_BACKEND == "sentence_transformer":
                raise
            print(f"SentenceTransformer 构建失败，自动改用 TF-IDF。原因：{e}")
            _, meta = build_with_tfidf(texts)
    elif VECTOR_BACKEND == "tfidf":
        _, meta = build_with_tfidf(texts)
    else:
        raise ValueError(f"未知 VECTOR_BACKEND：{VECTOR_BACKEND}")

    (STORAGE_DIR / "chunk_meta.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    (STORAGE_DIR / "vector_store_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"向量库构建完成，文本块数量：{len(chunks)}，后端：{meta['backend']}")


if __name__ == "__main__":
    build_vector_store()
