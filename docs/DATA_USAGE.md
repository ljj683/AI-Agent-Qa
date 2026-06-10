# 数据集使用说明

本项目使用 3 个 PDF 文档和 1 个自建文本分类 CSV。

## 1. 《机器学习基础及应用》课程 PDF

- 文件位置：`data/raw_docs/course_material/机器学习基础及应用.pdf`
- 用途：课程知识库主文档，覆盖机器学习开发环境、朴素贝叶斯、文本分类、神经网络等课程知识点。
- 处理说明：该文件可能属于扫描/图片型 PDF，普通文本提取字符较少，建议使用 MinerU 或 OCR 转 Markdown 后放入 `data/parsed/`。

## 2. 中国信通院《人工智能生成内容（AIGC）白皮书》

- 文件位置：`data/raw_docs/whitepaper/AIGC白皮书.pdf`
- 用途：中文 AI 白皮书扩展文档，用于 AIGC、大模型、应用场景等问题的问答测试。
- 来源：中国信息通信研究院、京东探索研究院公开白皮书。

## 3. Stanford AI Index Report 2025

- 文件位置：`data/raw_docs/whitepaper/AI_Index_Report_2025.pdf`
- 用途：英文长文档问答、多文档检索和跨语言检索测试。
- 来源：Stanford HAI AI Index Report 2025。

## 4. 自建文本分类数据集 text_classification.csv

- 文件位置：`data/labels/text_classification.csv`
- 样本数量：180 条。
- 类别数量：6 类，每类 30 条。
- 类别：文本分类、神经网络、RAG问答、文档解析、向量检索、模型评估。
- 用途：训练朴素贝叶斯、逻辑回归、MLP 神经网络三种文本分类模型，并进行 Accuracy、Precision、Recall、F1 多指标评估。
- 说明：该 CSV 为自建数据集，不是从 Kaggle/UCI/白皮书直接下载。它根据课程内容、方向六任务要求和项目功能模块人工整理生成。
