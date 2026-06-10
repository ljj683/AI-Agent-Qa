"""一键重建内置知识库和文本分类模型。"""
from src.text_preprocess import build_chunks
from src.build_vector_store import build_vector_store
from src.train_text_classifier import train_and_evaluate

if __name__ == "__main__":
    print("[1/3] 文本清洗与分块...")
    build_chunks()
    print("[2/3] 构建 TF-IDF 向量库...")
    build_vector_store()
    print("[3/3] 训练文本分类模型...")
    train_and_evaluate()
    print("全部完成。")
