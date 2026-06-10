# AI-Agent-Qa 部署指南

## 一、GitHub 上传

```bash
git init
git add .
git commit -m "initial AI-Agent-Qa project"
git branch -M main
git remote add origin 你的GitHub仓库地址
git push -u origin main
```

后续修改：

```bash
git add .
git commit -m "update project"
git push
```

## 二、Streamlit Cloud 部署

1. 进入 Streamlit Community Cloud
2. 点击 New app
3. 选择 GitHub 仓库
4. Branch 选择 main
5. Main file path 填：

```text
src/streamlit_app.py
```

6. Advanced settings / Secrets 填：

```toml
DEEPSEEK_API_KEY = "你的真实DeepSeek Key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
VECTOR_BACKEND = "tfidf"
TOP_K = "5"
```

7. 点击 Deploy

## 三、部署后测试

必须测试：

1. 上传新的 PDF
2. 上传新的 Word
3. 提问“这份文档主要讲了什么？”
4. 查看答案、来源片段、相似度
5. 测试内置知识库问答
6. 测试模型评估结果页面

## 四、公开展示链接

最终保存：

```text
GitHub 项目链接：
Streamlit Web 应用链接：
Dify 智能体链接：
```
