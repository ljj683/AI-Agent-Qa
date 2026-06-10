# AI-Agent-Qa：基于 DeepSeek + RAG + 机器学习模型的多格式文档问答 AI 智能体

本项目是一个高级课程项目 / 作品集级 AI Agent 项目，目标是在学生项目范围内实现一个功能完整、可部署、可演示的文档问答系统。系统不局限于固定三份文档，而是支持用户上传任意 PDF、Word、TXT、Markdown 文档后进行即时问答。

## 1. 项目核心功能

- 支持 PDF / DOCX / TXT / MD 多格式文档上传
- 支持单文件或多文件同时上传
- 自动完成文档解析、文本清洗和文本分块
- 使用 TF-IDF 完成文本向量化
- 使用余弦相似度完成 Top-K 检索
- 调用 DeepSeek 生成自然语言回答
- 展示答案来源片段、来源文件和相似度
- 保留内置知识库问答，用于展示多文档跨文档问答
- 集成朴素贝叶斯、逻辑回归和 MLP 神经网络文本分类模型
- 展示 Accuracy、Precision、Recall、F1 等模型评估指标
- 可部署到 Streamlit Cloud，公开访问
- 可配合 Dify 智能体展示知识库、RAG 工作流和文件上传问答

## 2. 技术栈

- Python 3.10+
- Streamlit：Web 应用界面
- PyMuPDF：PDF 文本解析
- python-docx：Word 文档解析
- scikit-learn：TF-IDF、余弦相似度、文本分类模型、GridSearchCV
- DeepSeek API：大模型回答生成
- Dify：智能体工作流与知识库问答展示
- GitHub + Streamlit Cloud：代码托管与公网部署

## 3. 项目结构

```text
AI-Agent-Qa/
├── data/
│   ├── parsed/                  # 内置演示知识库文本
│   ├── labels/                  # 文本分类数据集
│   ├── raw_docs/                # 原始文档存放目录，不建议上传大 PDF 到 GitHub
│   └── samples/                 # 示例文件目录
├── reports/                     # 模型评估结果
├── storage/                     # 已构建好的向量库和分类模型
├── src/
│   ├── streamlit_app.py         # Streamlit 主应用
│   ├── upload_doc_engine.py     # 上传文档解析与临时 RAG 检索
│   ├── rag_engine.py            # 内置知识库 RAG 引擎
│   ├── config.py                # 配置文件
│   ├── text_preprocess.py       # 文本清洗与分块
│   ├── build_vector_store.py    # 内置知识库向量化
│   ├── train_text_classifier.py # 文本分类模型训练
│   └── api_server.py            # FastAPI 本地接口，可选
├── docs/
│   ├── FULL_PROCESS_GUIDE.md    # 完整操作流程
│   ├── ERROR_GUIDE.md           # 常见报错与解决方法
│   ├── DIFY_GUIDE.md            # Dify 智能体搭建指南
│   └── DEPLOYMENT_GUIDE.md      # 部署指南
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 4. 本地运行

### 4.1 创建虚拟环境

```bash
cd AI-Agent-Qa
python -m venv .venv
```

Windows 激活：

```bash
.venv\Scripts\activate
```

macOS / Linux 激活：

```bash
source .venv/bin/activate
```

### 4.2 安装依赖

```bash
pip install -r requirements.txt
```

### 4.3 配置 DeepSeek

复制 `.env.example` 为 `.env`，并填写：

```env
DEEPSEEK_API_KEY=你的真实DeepSeek Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
VECTOR_BACKEND=tfidf
TOP_K=5
```

### 4.4 运行 Streamlit

```bash
streamlit run src/streamlit_app.py
```

打开页面后，优先测试第一个 Tab：`上传文档问答`。

## 5. 重新构建内置知识库

如果你修改了 `data/parsed/` 中的文本文件，运行：

```bash
python -m src.text_preprocess
python -m src.build_vector_store
```

如果你修改了文本分类数据集，运行：

```bash
python -m src.train_text_classifier
```

也可以一次性运行：

```bash
python run_pipeline.py
```

## 6. Streamlit Cloud 部署 Secrets

在 Streamlit Cloud 的 `Settings -> Secrets` 中填写：

```toml
DEEPSEEK_API_KEY = "你的真实DeepSeek Key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
VECTOR_BACKEND = "tfidf"
TOP_K = "5"
```

## 7. 推荐演示顺序

1. 打开 Streamlit Web 应用
2. 进入“上传文档问答”
3. 上传一份新的 PDF 或 Word，不使用内置三份文档
4. 提问：这份文档主要讲了什么？
5. 展示答案、来源片段和相似度
6. 再问：请总结三个重点
7. 打开“内置知识库问答”展示多文档问答
8. 打开“模型评估结果”展示三种模型对比
9. 打开 Dify 智能体链接展示平台化智能体
10. 打开 GitHub 仓库展示完整代码

## 8. 注意事项

- `.env` 不要上传 GitHub。
- `storage/` 目录不要删除，Streamlit Cloud 需要读取里面的向量库和模型文件。
- 扫描版 PDF 可能无法直接解析文字，建议先用 MinerU/OCR 转 Markdown 或 TXT。
- DeepSeek 模型建议默认使用 `deepseek-v4-flash`，课堂演示更稳定。
- 如果 `deepseek-v4-flash` 不可用，可改为 `deepseek-chat`。
