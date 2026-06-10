# AI-Agent-Qa 完整操作流程

## 一、项目定位

本项目定位为高级课程项目 / 作品集级 AI Agent 项目。系统支持用户上传任意 PDF、Word、TXT、Markdown 文档，自动完成文档解析、文本分块、TF-IDF 向量化、余弦相似度检索，并结合 DeepSeek 生成回答。

内置的三份文档只是演示知识库，不代表系统只能回答这三份文档。

## 二、从零开始操作

### 1. 解压项目

将压缩包 `AI-Agent-Qa.zip` 解压到本地，例如：

```text
D:\AI-Agent-Qa
```

用 PyCharm 打开这个文件夹。

### 2. 创建虚拟环境

在 PyCharm Terminal 中运行：

```bash
python -m venv .venv
```

激活虚拟环境：

```bash
.venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置 DeepSeek

复制 `.env.example`，重命名为 `.env`。

填写：

```env
DEEPSEEK_API_KEY=你的真实DeepSeek Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
VECTOR_BACKEND=tfidf
TOP_K=5
```

### 5. 运行 Streamlit

```bash
streamlit run src/streamlit_app.py
```

浏览器会打开本地页面。

## 三、必须测试的功能

### 1. 上传文档问答

进入第一个 Tab：

```text
📤 上传文档问答
```

上传一个新的 PDF 或 Word，不要用内置三份文档。

测试问题：

```text
这份文档主要讲了什么？
```

再测试：

```text
请总结这份文档的三个重点。
```

再测试一个文档中不存在的问题：

```text
这份文档有没有讲新能源汽车电池管理？
```

理想结果：系统能回答文档相关问题；如果文档没有依据，会说明“上传文档中没有找到明确依据”。

### 2. 内置知识库问答

进入第二个 Tab：

```text
📚 内置知识库问答
```

测试：

```text
什么是神经网络？
```

```text
什么是朴素贝叶斯算法？
```

```text
AIGC 的主要应用场景有哪些？
```

### 3. 模型评估结果

进入第三个 Tab：

```text
📊 模型评估结果
```

检查是否显示朴素贝叶斯、逻辑回归、MLP 神经网络的 Accuracy、Precision、Recall、F1。

## 四、重新训练与构建

如果修改 `data/parsed/` 的内置知识库文本：

```bash
python -m src.text_preprocess
python -m src.build_vector_store
```

如果修改 `data/labels/text_classification.csv`：

```bash
python -m src.train_text_classifier
```

如果想一键重跑：

```bash
python run_pipeline.py
```

## 五、GitHub 上传

```bash
git init
git add .
git commit -m "initial AI-Agent-Qa project"
git branch -M main
git remote add origin 你的GitHub仓库地址
git push -u origin main
```

后续更新：

```bash
git add .
git commit -m "update AI-Agent-Qa"
git push
```

## 六、Streamlit Cloud 部署

1. 登录 Streamlit Community Cloud
2. New app
3. 选择 GitHub 仓库
4. Main file path 填：

```text
src/streamlit_app.py
```

5. Settings -> Secrets 填：

```toml
DEEPSEEK_API_KEY = "你的真实DeepSeek Key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
VECTOR_BACKEND = "tfidf"
TOP_K = "5"
```

6. Deploy

## 七、Dify 智能体建议

保留两个 Dify 场景：

### 场景 1：知识库 RAG 问答

流程：

```text
用户输入 -> 知识检索 -> LLM -> 直接回复
```

### 场景 2：上传文档问答

流程：

```text
用户输入 + 文件上传 -> 文档提取器 -> LLM -> 直接回复
```

## 八、最终展示顺序

1. 打开 Streamlit Web 应用
2. 展示上传文档问答
3. 上传新的 PDF / Word
4. 展示答案、检索片段、相似度
5. 展示内置知识库问答
6. 展示模型评估结果
7. 展示 Dify 智能体
8. 展示 GitHub 仓库
