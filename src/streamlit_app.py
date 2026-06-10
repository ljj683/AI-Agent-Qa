import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# 让 Streamlit Cloud / 本地命令行都能正确 import src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_secrets_to_env():
    """优先读取 Streamlit Secrets，其次读取本地 .env。"""
    keys = [
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "VECTOR_BACKEND",
        "TOP_K",
    ]
    try:
        for key in keys:
            if key in st.secrets:
                os.environ[key] = str(st.secrets[key])
    except Exception:
        # 本地没有 secrets.toml 时会进入这里，不影响 .env 读取。
        pass


load_secrets_to_env()

from src.rag_engine import RAGEngine  # noqa: E402
from src.upload_doc_engine import UploadedDocumentIndex, generate_answer_with_deepseek  # noqa: E402


st.set_page_config(
    page_title="AI-Agent-Qa | 多格式文档问答智能体",
    page_icon="🤖",
    layout="wide",
)

CUSTOM_CSS = """
<style>
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 45%, #fdf2f8 100%);
}
section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.78);
    backdrop-filter: blur(18px);
    border-right: 1px solid rgba(148, 163, 184, 0.25);
}
.hero {
    padding: 32px 36px;
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.96), rgba(124, 58, 237, 0.92));
    color: white;
    box-shadow: 0 22px 50px rgba(79, 70, 229, 0.22);
    margin-bottom: 28px;
}
.hero-badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 700;
    background: rgba(255, 255, 255, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.25);
    margin-bottom: 16px;
}
.hero-title {
    font-size: 44px;
    line-height: 1.15;
    font-weight: 900;
    letter-spacing: -1.1px;
    margin-bottom: 14px;
}
.hero-desc {
    font-size: 18px;
    line-height: 1.8;
    color: rgba(255, 255, 255, 0.92);
    max-width: 1080px;
}
.feature-card {
    padding: 22px;
    border-radius: 22px;
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid rgba(148, 163, 184, 0.22);
    box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
    min-height: 145px;
}
.feature-icon {font-size: 30px; margin-bottom: 10px;}
.feature-title {font-size: 18px; font-weight: 850; color: #111827; margin-bottom: 8px;}
.feature-text {font-size: 14px; color: #475569; line-height: 1.65;}
.answer-box {
    padding: 22px 24px;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.95);
    border-left: 6px solid #4f46e5;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.10);
    line-height: 1.8;
}
.info-panel {
    padding: 22px;
    border-radius: 22px;
    background: rgba(248, 250, 252, 0.9);
    border: 1px solid rgba(203, 213, 225, 0.75);
    margin-bottom: 18px;
}
.small-note {color: #64748b; font-size: 14px; line-height: 1.7;}
div.stButton > button:first-child {
    border-radius: 999px;
    padding: 0.65rem 1.4rem;
    font-weight: 750;
    border: none;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
}
div.stButton > button:first-child:hover {
    box-shadow: 0 12px 26px rgba(79, 70, 229, 0.28);
    transform: translateY(-1px);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_builtin_rag(api_key_flag: str, model_name: str):
    return RAGEngine()


api_key_flag = "configured" if os.getenv("DEEPSEEK_API_KEY") else "missing"
model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

try:
    builtin_rag = load_builtin_rag(api_key_flag, model_name)
except Exception as e:
    st.error("系统初始化失败，请检查 storage 文件、模型文件和 Streamlit Secrets / .env 配置。")
    st.exception(e)
    st.stop()

if "upload_index" not in st.session_state:
    st.session_state.upload_index = None
if "upload_signature" not in st.session_state:
    st.session_state.upload_signature = None
if "upload_history" not in st.session_state:
    st.session_state.upload_history = []
if "builtin_history" not in st.session_state:
    st.session_state.builtin_history = []

with st.sidebar:
    st.markdown("## 🤖 AI-Agent-Qa")
    st.write("多格式文档问答AI智能体")
    st.divider()
    st.markdown("### 核心能力")
    st.write("- 任意 PDF / Word / TXT / MD 上传问答")
    st.write("- 文档解析、文本分块、TF-IDF 向量化")
    st.write("- 余弦相似度 Top-K 检索")
    st.write("- DeepSeek 大模型回答生成")
    st.write("- 内置多文档知识库问答")
    st.write("- 朴素贝叶斯、逻辑回归、MLP 文本分类")
    st.divider()
    st.markdown("### 运行配置")
    st.write("模型：", os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    st.write("Top-K：", os.getenv("TOP_K", "5"))
    st.write("向量后端：", os.getenv("VECTOR_BACKEND", "tfidf"))
    st.write("DeepSeek Key：", "已配置" if os.getenv("DEEPSEEK_API_KEY") else "未配置")

st.markdown(
    """
<div class="hero">
  <div class="hero-badge">AI Document Agent · RAG · DeepSeek · Machine Learning</div>
  <div class="hero-title">🤖 AI-Agent-Qa 多格式文档问答智能体</div>
  <div class="hero-desc">
    支持上传任意 PDF / Word / TXT / Markdown 文档，自动完成文档解析、文本分块、TF-IDF 向量化、
    余弦相似度检索，并结合 DeepSeek 大模型生成可追溯的智能回答。
  </div>
</div>
""",
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
<div class="feature-card">
  <div class="feature-icon">📤</div>
  <div class="feature-title">任意文档上传</div>
  <div class="feature-text">支持 PDF、Word、TXT、Markdown，可单文件或多文件同时上传。</div>
</div>
""", unsafe_allow_html=True)
with col2:
    st.markdown("""
<div class="feature-card">
  <div class="feature-icon">🔎</div>
  <div class="feature-title">RAG 检索增强</div>
  <div class="feature-text">文本分块、TF-IDF 向量化、余弦相似度 Top-K 检索。</div>
</div>
""", unsafe_allow_html=True)
with col3:
    st.markdown("""
<div class="feature-card">
  <div class="feature-icon">🧠</div>
  <div class="feature-title">机器学习模型</div>
  <div class="feature-text">集成朴素贝叶斯、逻辑回归、MLP 神经网络文本分类模型。</div>
</div>
""", unsafe_allow_html=True)
with col4:
    st.markdown("""
<div class="feature-card">
  <div class="feature-icon">⚡</div>
  <div class="feature-title">DeepSeek 回答生成</div>
  <div class="feature-text">基于检索上下文生成结构化答案，并展示来源片段与相似度。</div>
</div>
""", unsafe_allow_html=True)

st.write("")

tab_upload, tab_builtin, tab_model, tab_about = st.tabs([
    "📤 上传文档问答",
    "📚 内置知识库问答",
    "📊 模型评估结果",
    "🧠 项目说明",
])

with tab_upload:
    st.markdown("### 📤 上传文档，即刻问答")
    st.markdown(
        '<div class="small-note">上传任意 PDF / Word / TXT / Markdown 文档，系统会临时构建检索索引并基于上传内容回答问题。支持多文件同时上传。</div>',
        unsafe_allow_html=True,
    )
    st.info("扫描版 PDF 可能无法直接提取文字，建议先使用 MinerU/OCR 转为 Markdown 或 TXT 后再上传。")

    uploaded_files = st.file_uploader(
        "上传文档文件：",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="可上传一个或多个 PDF / Word / TXT / Markdown 文件。",
    )

    if uploaded_files:
        signature = tuple((f.name, f.size) for f in uploaded_files)
        if signature != st.session_state.upload_signature:
            with st.spinner("正在解析上传文档并构建临时 TF-IDF 检索索引..."):
                try:
                    index = UploadedDocumentIndex.from_uploaded_files(uploaded_files)
                    st.session_state.upload_index = index
                    st.session_state.upload_signature = signature
                    st.session_state.upload_history = []
                    st.success("文档解析与临时检索索引构建完成。")
                except Exception as exc:
                    st.error("上传文档处理失败。")
                    st.exception(exc)
                    st.stop()

    if st.session_state.upload_index is not None:
        summaries = st.session_state.upload_index.summaries
        summary_df = pd.DataFrame([s.__dict__ for s in summaries])
        summary_df = summary_df.rename(columns={
            "file_name": "文件名",
            "file_type": "文件类型",
            "file_size_mb": "文件大小(MB)",
            "text_length": "解析文本长度",
            "chunk_count": "文本分块数量",
        })
        st.markdown("#### 文档解析状态")
        st.dataframe(summary_df, use_container_width=True)

        total_chunks = sum(s.chunk_count for s in summaries)
        total_chars = sum(s.text_length for s in summaries)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("上传文件数", len(summaries))
        c2.metric("解析字符数", total_chars)
        c3.metric("文本块数量", total_chunks)
        c4.metric("检索方式", "TF-IDF + Cosine")

        question = st.text_input(
            "请输入针对上传文档的问题：",
            placeholder="例如：这份文档主要讲了什么？请总结三个重点。",
            key="upload_question",
        )

        ask_upload = st.button("基于上传文档问答", type="primary", key="upload_ask")
        clear_upload = st.button("清空上传问答历史", key="upload_clear")
        if clear_upload:
            st.session_state.upload_history = []
            st.success("上传文档问答历史已清空。")

        if ask_upload and question.strip():
            with st.spinner("正在检索上传文档并生成回答..."):
                top_k = int(os.getenv("TOP_K", "5"))
                contexts = st.session_state.upload_index.retrieve(question, top_k=top_k)
                question_type = builtin_rag.classify_question(question)
                answer = generate_answer_with_deepseek(question, contexts, st.session_state.upload_history)
                st.session_state.upload_history.append({"question": question, "answer": answer})

                st.subheader("问题分类结果")
                st.info(question_type)

                st.subheader("智能回答")
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                scores = [ctx.get("score", 0) for ctx in contexts]
                scores = [ctx.get("score", 0) for ctx in contexts]
                top_score = max(scores) if scores else 0
                avg_score = sum(scores) / len(scores) if scores else 0

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("最高相关性", f"{top_score:.4f}")
                with col2:
                    st.metric("平均相关性", f"{avg_score:.4f}")

                st.caption("说明：TF-IDF 余弦相似度主要用于比较同一次检索中不同片段的相对相关性，分数通常不会接近 1。")

                result_text = "问题：" + question + "\n\n回答：\n" + answer + "\n\n检索片段：\n" + "\n\n".join(
                    f"片段{i + 1}｜来源：{ctx.get('source_file')}｜相似度：{ctx.get('score', 0):.4f}\n{ctx.get('text', '')}"
                    for i, ctx in enumerate(contexts)
                )
                st.download_button(
                    "下载本次问答结果",
                    data=result_text,
                    file_name="upload_qa_result.txt",
                    mime="text/plain",
                )

                st.subheader("上传文档检索片段")
                for i, ctx in enumerate(contexts, start=1):
                    with st.expander(f"片段 {i} | 来源：{ctx.get('source_file')} | 相似度：{ctx.get('score', 0):.4f}"):
                        st.write(ctx.get("text", ""))

        if st.session_state.upload_history:
            st.markdown("#### 最近上传文档问答历史")
            for i, item in enumerate(st.session_state.upload_history[-5:], start=1):
                with st.expander(f"历史 {i}：{item['question']}"):
                    st.write(item["answer"])
    else:
        st.warning("请先上传 PDF、Word、TXT 或 Markdown 文档。")

with tab_builtin:
    st.markdown("### 📚 内置多文档知识库问答")
    st.markdown(
        '<div class="small-note">内置知识库用于演示多文档跨文档问答能力。系统并不局限于这些文件，核心入口是“上传文档问答”。</div>',
        unsafe_allow_html=True,
    )

    builtin_question = st.text_input(
        "请输入你的问题：",
        placeholder="例如：什么是神经网络？AIGC 的主要应用场景有哪些？",
        key="builtin_question",
    )
    b1, b2 = st.columns([1, 1])
    ask_builtin = b1.button("检索内置知识库", type="primary", key="builtin_ask")
    clear_builtin = b2.button("清空内置问答历史", key="builtin_clear")
    if clear_builtin:
        st.session_state.builtin_history = []
        st.success("内置问答历史已清空。")

    if ask_builtin and builtin_question.strip():
        with st.spinner("正在检索内置知识库并生成回答..."):
            try:
                result = builtin_rag.ask(builtin_question, st.session_state.builtin_history)
                st.session_state.builtin_history.append({
                    "question": builtin_question,
                    "answer": result.get("answer", ""),
                })

                st.subheader("问题分类结果")
                st.info(result.get("question_type", "未分类"))

                st.subheader("智能回答")
                st.markdown(f'<div class="answer-box">{result.get("answer", "未返回答案")}</div>', unsafe_allow_html=True)

                contexts = result.get("contexts", [])
                scores = [ctx.get("score", 0) for ctx in contexts]
                avg_score = sum(scores) / len(scores) if scores else 0
                st.metric("本次检索平均相关性", f"{avg_score:.4f}")

                st.subheader("检索到的相关文档片段")
                for i, ctx in enumerate(contexts, start=1):
                    with st.expander(f"片段 {i} | 来源：{ctx.get('source_file', '未知来源')} | 相似度：{ctx.get('score', 0):.4f}"):
                        st.write(ctx.get("text", ""))
            except Exception as exc:
                st.error("内置知识库问答出错，请检查 DeepSeek Key、模型名或 storage 文件。")
                st.exception(exc)

    if st.session_state.builtin_history:
        st.markdown("#### 最近内置问答历史")
        for i, item in enumerate(st.session_state.builtin_history[-5:], start=1):
            with st.expander(f"历史 {i}：{item['question']}"):
                st.write(item["answer"])

with tab_model:
    st.markdown("### 📊 机器学习模型评估结果")
    st.markdown(
        '<div class="small-note">项目使用自建文本分类数据集训练朴素贝叶斯、逻辑回归和 MLP 神经网络模型，并进行多指标评估。</div>',
        unsafe_allow_html=True,
    )
    metrics_path = PROJECT_ROOT / "reports" / "model_metrics.csv"
    best_model_path = PROJECT_ROOT / "reports" / "best_model.txt"

    if metrics_path.exists():
        try:
            df = pd.read_csv(metrics_path)

            display_cols = [
                "model",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
            ]

            show_df = df[display_cols].copy()

            for col in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
                show_df[col] = show_df[col].round(4)

            st.dataframe(show_df, use_container_width=True)
            st.info(
                "说明：本项目使用的小型自建文本分类数据集样本量有限，MLP 神经网络在小样本场景下容易出现训练不稳定，因此最终选择 F1-macro 表现更好的逻辑回归模型作为最优分类模型。"
            )
            if best_model_path.exists():
                st.success(best_model_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception as exc:
            st.warning("模型评估文件存在，但读取失败。")
            st.exception(exc)
    else:
        st.warning("未找到 reports/model_metrics.csv，请先运行 python -m src.train_text_classifier")

    st.markdown("""
<div class="info-panel">
<b>模型设计说明：</b><br>
本模块基于自建文本分类数据集，对用户问题进行主题分类实验。实验对比了朴素贝叶斯、逻辑回归和 MLP 神经网络三种模型，
并使用 Accuracy、Precision-macro、Recall-macro 和 F1-macro 进行综合评价。
从实验结果看，逻辑回归模型在整体指标上表现较稳定，因此被选为当前系统的问题分类模型。
</div>
""", unsafe_allow_html=True)

with tab_about:
    st.markdown("### 🧠 项目说明与技术路线")
    st.markdown("""
<div class="info-panel">
<h4>项目定位</h4>
本系统面向多格式文档问答场景，支持用户上传 PDF、Word、TXT、Markdown 文档，自动完成文本解析、分块、检索和问答生成。
</div>

<div class="info-panel">
<h4>核心技术路线</h4>
用户上传文档 → PDF / Word / TXT / Markdown 解析 → 文本清洗 → 文本分块 → TF-IDF 向量化 → 余弦相似度 Top-K 检索 → DeepSeek 生成回答 → 展示答案、来源片段和相似度。
</div>

<div class="info-panel">
<h4>项目亮点</h4>
<ul>
<li>支持任意 PDF / Word / TXT / Markdown 上传问答，不局限于固定三份文档；</li>
<li>使用 TF-IDF 与余弦相似度实现轻量、可解释的 RAG 检索；</li>
<li>回答结果展示来源片段和相似度，增强可追溯性；</li>
<li>集成 DeepSeek V4 系列模型生成结构化答案；</li>
<li>训练朴素贝叶斯、逻辑回归、MLP 神经网络文本分类模型；</li>
<li>结合 Streamlit Web 应用和 Dify 智能体完成双入口展示。</li>
</ul>
</div>
""", unsafe_allow_html=True)
