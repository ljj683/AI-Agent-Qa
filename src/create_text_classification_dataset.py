"""重新生成自建文本分类数据集的备用脚本。当前项目已经附带 data/labels/text_classification.csv。"""
import pandas as pd
from pathlib import Path
from src.config import LABELS_DIR

samples = [
    ("什么是文本分类", "文本分类"),
    ("朴素贝叶斯如何用于文本分类", "文本分类"),
    ("什么是神经网络", "神经网络"),
    ("反向传播算法有什么作用", "神经网络"),
    ("什么是RAG检索增强生成", "RAG问答"),
    ("文档问答系统为什么需要检索增强", "RAG问答"),
    ("如何解析PDF文档", "文档解析"),
    ("扫描版PDF为什么需要OCR", "文档解析"),
    ("什么是向量化存储", "向量检索"),
    ("余弦相似度如何计算文本相似度", "向量检索"),
    ("准确率是什么意思", "模型评估"),
    ("F1值是什么意思", "模型评估"),
]

# 简单扩增为演示数据；正式数据使用项目附带的 180 条 CSV。
rows = []
for text, label in samples:
    rows.append({"text": text, "label": label})

LABELS_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(LABELS_DIR / "text_classification_minimal.csv", index=False, encoding="utf-8-sig")
print("已生成 text_classification_minimal.csv")
