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
:root {
    --bg: #f6f8fb;
    --surface: #ffffff;
    --surface-soft: #f8fafc;
    --line: #d9e2ec;
    --line-strong: #cbd5e1;
    --ink: #172033;
    --muted: #5f6f89;
    --accent: #0f9f9a;
    --accent-deep: #1268b3;
    --accent-soft: #e8f7f6;
    --shadow: 0 12px 28px rgba(20, 32, 54, 0.08);
}

.stApp {
    color: var(--ink);
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.72), rgba(246, 248, 251, 0.94)),
        linear-gradient(135deg, #eef8f7 0%, #f6f8fb 44%, #eef4fb 100%);
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1260px;
    padding-top: 1.35rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: var(--ink);
    letter-spacing: 0;
}

p {
    color: var(--muted);
}

section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.94);
    border-right: 1px solid var(--line);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li {
    color: var(--ink);
}

.sidebar-brand {
    padding: 8px 0 18px;
}

.brand-row {
    display: flex;
    align-items: center;
    gap: 10px;
}

.brand-mark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 34px;
    height: 34px;
    border-radius: 8px;
    color: #ffffff;
    font-weight: 900;
    background: linear-gradient(135deg, var(--accent), var(--accent-deep));
    box-shadow: 0 8px 18px rgba(18, 104, 179, 0.18);
}

.brand-name {
    font-size: 18px;
    font-weight: 850;
    color: var(--ink);
}

.brand-subtitle {
    margin-top: 9px;
    font-size: 13px;
    line-height: 1.6;
    color: var(--muted);
}

.side-section-title {
    margin: 18px 0 10px;
    font-size: 12px;
    font-weight: 850;
    color: #6b7c93;
    text-transform: uppercase;
}

.capability-list {
    display: grid;
    gap: 8px;
}

.capability-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 10px;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--surface-soft);
    font-size: 13px;
    color: var(--ink);
}

.capability-dot {
    width: 6px;
    height: 6px;
    border-radius: 999px;
    background: var(--accent);
}

.status-card {
    padding: 12px;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--surface);
    box-shadow: 0 8px 20px rgba(20, 32, 54, 0.05);
}

.config-row {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid #edf2f7;
    font-size: 13px;
}

.config-row:last-child {
    border-bottom: 0;
}

.config-row span {
    color: var(--muted);
}

.config-row strong {
    color: var(--ink);
    text-align: right;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 8px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 12px;
}

.status-pill.ok {
    color: #067a6f;
    background: #e4f7f4;
}

.status-pill.warn {
    color: #9a5b00;
    background: #fff4d6;
}

.hero {
    display: grid;
    grid-template-columns: minmax(0, 1.5fr) minmax(280px, 0.75fr);
    gap: 28px;
    align-items: stretch;
    padding: 30px;
    border: 1px solid rgba(18, 104, 179, 0.18);
    border-radius: 8px;
    color: #ffffff;
    background:
        linear-gradient(135deg, rgba(15, 159, 154, 0.96) 0%, rgba(18, 104, 179, 0.98) 100%);
    box-shadow: var(--shadow);
    margin-bottom: 18px;
    overflow: hidden;
}

.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.28);
    background: rgba(255, 255, 255, 0.12);
    font-size: 12px;
    font-weight: 800;
    color: rgba(255, 255, 255, 0.92);
}

.hero-title {
    margin: 16px 0 10px;
    font-size: clamp(30px, 4vw, 46px);
    line-height: 1.12;
    font-weight: 900;
    letter-spacing: 0;
    color: #ffffff;
}

.hero-desc {
    max-width: 780px;
    font-size: 16px;
    line-height: 1.78;
    color: rgba(255, 255, 255, 0.9);
}

.hero-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 18px;
}

.hero-chip {
    padding: 6px 10px;
    border-radius: 999px;
    color: #ffffff;
    background: rgba(255, 255, 255, 0.13);
    border: 1px solid rgba(255, 255, 255, 0.2);
    font-size: 12px;
    font-weight: 800;
}

.hero-panel {
    display: grid;
    gap: 10px;
    align-content: center;
}

.hero-stat {
    padding: 12px 14px;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.18);
}

.hero-stat strong {
    display: block;
    color: #ffffff;
    font-size: 19px;
    line-height: 1.2;
}

.hero-stat span {
    display: block;
    margin-top: 4px;
    color: rgba(255, 255, 255, 0.82);
    font-size: 12px;
}

.feature-card {
    min-height: 138px;
    padding: 18px;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid var(--line);
    box-shadow: 0 8px 22px rgba(20, 32, 54, 0.06);
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}

.feature-card:hover {
    transform: translateY(-2px);
    border-color: rgba(15, 159, 154, 0.45);
    box-shadow: 0 14px 28px rgba(20, 32, 54, 0.1);
}

.feature-index {
    width: 34px;
    height: 28px;
    border-radius: 8px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 14px;
    color: var(--accent-deep);
    background: var(--accent-soft);
    font-size: 12px;
    font-weight: 900;
}

.feature-title {
    margin-bottom: 8px;
    color: var(--ink);
    font-size: 16px;
    font-weight: 850;
}

.feature-text {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.68;
}

div[data-testid="stTabs"] {
    margin-top: 10px;
}

button[data-baseweb="tab"] {
    padding: 11px 18px;
    border-radius: 8px 8px 0 0;
    font-size: 14px;
    font-weight: 800;
    color: #50627b;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent-deep);
    background: rgba(255, 255, 255, 0.68);
}

.section-title {
    margin: 16px 0 14px;
}

.section-eyebrow {
    margin-bottom: 6px;
    color: var(--accent-deep);
    font-size: 12px;
    font-weight: 850;
    text-transform: uppercase;
}

.section-title h2,
.section-title h3 {
    margin: 0;
    color: var(--ink);
    font-weight: 900;
    letter-spacing: 0;
}

.section-title p {
    margin: 8px 0 0;
    max-width: 820px;
    color: var(--muted);
    font-size: 14px;
    line-height: 1.72;
}

.notice {
    margin: 12px 0 18px;
    padding: 12px 14px;
    border: 1px solid #bde8e3;
    border-radius: 8px;
    color: #0b6d68;
    background: #edfafa;
    font-size: 14px;
    line-height: 1.7;
}

.info-panel {
    padding: 18px 20px;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid var(--line);
    box-shadow: 0 8px 22px rgba(20, 32, 54, 0.06);
    margin-bottom: 14px;
    color: var(--ink);
    line-height: 1.82;
}

.info-panel h4 {
    margin: 0 0 8px;
    color: var(--ink);
}

.answer-box {
    padding: 18px 20px;
    border-radius: 8px;
    background: #ffffff;
    border: 1px solid var(--line);
    border-left: 4px solid var(--accent);
    box-shadow: 0 10px 24px rgba(20, 32, 54, 0.08);
    color: var(--ink);
    line-height: 1.86;
}

.small-note {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.75;
}

div[data-testid="stMetric"] {
    padding: 14px 16px;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.88);
    box-shadow: 0 8px 18px rgba(20, 32, 54, 0.05);
}

div.stButton > button,
div[data-testid="stFormSubmitButton"] > button,
div[data-testid="stDownloadButton"] > button {
    border-radius: 8px;
    padding: 0.64rem 1rem;
    font-weight: 850;
    border: 1px solid var(--line-strong);
    background: #ffffff;
    color: var(--ink);
    box-shadow: 0 6px 14px rgba(20, 32, 54, 0.06);
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:hover {
    border-color: rgba(18, 104, 179, 0.42);
    transform: translateY(-1px);
    box-shadow: 0 10px 18px rgba(20, 32, 54, 0.1);
}

div.stButton > button[kind="primary"],
div.stButton > button[data-testid="baseButton-primary"],
div[data-testid="stFormSubmitButton"] > button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button[data-testid="baseButton-primary"] {
    border: 1px solid rgba(15, 159, 154, 0.2);
    background: linear-gradient(135deg, var(--accent), var(--accent-deep));
    color: #ffffff;
    box-shadow: 0 8px 18px rgba(18, 104, 179, 0.16);
}

div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="baseButton-primary"]:hover,
div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] > button[data-testid="baseButton-primary"]:hover {
    border-color: rgba(18, 104, 179, 0.4);
    box-shadow: 0 12px 22px rgba(18, 104, 179, 0.22);
}

div[data-testid="stTextInput"] input {
    border-radius: 8px;
    border: 1px solid var(--line-strong);
    background: #ffffff;
}

section[data-testid="stFileUploaderDropzone"] {
    border-radius: 8px;
    border: 1.5px dashed rgba(15, 159, 154, 0.55);
    background: rgba(255, 255, 255, 0.82);
}

section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent-deep);
    background: #ffffff;
}

div[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--line);
}

div[data-testid="stAlert"] {
    border-radius: 8px;
    border: 1px solid var(--line);
}

details {
    border-radius: 8px !important;
    border: 1px solid var(--line) !important;
    background: rgba(255, 255, 255, 0.78) !important;
}

@media (max-width: 900px) {
    .hero {
        grid-template-columns: 1fr;
        padding: 22px;
    }

    .hero-title {
        font-size: 31px;
    }
}

footer {
    visibility: hidden;
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

model_label = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
top_k_label = os.getenv("TOP_K", "5")
vector_backend_label = os.getenv("VECTOR_BACKEND", "tfidf")
api_key_ready = bool(os.getenv("DEEPSEEK_API_KEY"))
api_key_label = "已配置" if api_key_ready else "未配置"
api_key_class = "ok" if api_key_ready else "warn"

with st.sidebar:
    st.markdown(
        """
<div class="sidebar-brand">
    <div class="brand-row">
        <div class="brand-mark">AI</div>
        <div>
            <div class="brand-name">AI-Agent-Qa</div>
            <div class="brand-subtitle">面向多格式文档的 RAG 问答工作台</div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        """
<div class="side-section-title">Core capabilities</div>
<div class="capability-list">
    <div class="capability-item"><span class="capability-dot"></span>多格式文档上传问答</div>
    <div class="capability-item"><span class="capability-dot"></span>文档解析、清洗与分块</div>
    <div class="capability-item"><span class="capability-dot"></span>TF-IDF 向量化检索</div>
    <div class="capability-item"><span class="capability-dot"></span>Top-K 来源片段追溯</div>
    <div class="capability-item"><span class="capability-dot"></span>DeepSeek 结构化生成</div>
    <div class="capability-item"><span class="capability-dot"></span>文本分类模型评估</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        f"""
<div class="side-section-title">Runtime</div>
<div class="status-card">
    <div class="config-row"><span>模型</span><strong>{model_label}</strong></div>
    <div class="config-row"><span>Top-K</span><strong>{top_k_label}</strong></div>
    <div class="config-row"><span>向量后端</span><strong>{vector_backend_label}</strong></div>
    <div class="config-row"><span>DeepSeek Key</span><strong><span class="status-pill {api_key_class}">{api_key_label}</span></strong></div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
<div class="hero">
    <div>
        <div class="hero-kicker">AI Document Agent · RAG · DeepSeek</div>
        <div class="hero-title">多格式文档问答智能体</div>
        <div class="hero-desc">
            上传 PDF、Word、TXT、Markdown 文档后，系统自动完成解析、分块、检索、生成与来源追溯，
            帮你把散落的资料变成可查询、可验证的知识工作台。
        </div>
        <div class="hero-chip-row">
            <span class="hero-chip">PDF</span>
            <span class="hero-chip">DOCX</span>
            <span class="hero-chip">TXT</span>
            <span class="hero-chip">Markdown</span>
            <span class="hero-chip">TF-IDF + Cosine</span>
        </div>
    </div>
    <div class="hero-panel">
        <div class="hero-stat"><strong>{model_label}</strong><span>当前生成模型</span></div>
        <div class="hero-stat"><strong>Top-K {top_k_label}</strong><span>默认召回片段数量</span></div>
        <div class="hero-stat"><strong>{vector_backend_label.upper()}</strong><span>轻量可解释检索后端</span></div>
    </div>
</div>
""",
    unsafe_allow_html=True
)

feature_items = [
    ("01", "文档上传解析", "支持 PDF、Word、TXT、Markdown，完成文本抽取、清洗与基础结构化。"),
    ("02", "RAG 检索增强", "文本分块、TF-IDF 向量化、余弦相似度 Top-K 检索与片段追溯。"),
    ("03", "问题分类模型", "集成朴素贝叶斯、逻辑回归、MLP 神经网络，用于问题主题识别。"),
    ("04", "DeepSeek 生成", "基于检索上下文生成结构化回答，并展示来源片段与相似度。"),
]

feature_cols = st.columns(4)
for col, (index, title, text) in zip(feature_cols, feature_items):
    with col:
        st.markdown(
            f"""
<div class="feature-card">
    <div class="feature-index">{index}</div>
    <div class="feature-title">{title}</div>
    <div class="feature-text">{text}</div>
</div>
""",
            unsafe_allow_html=True,
        )

st.write("")

tab_upload, tab_builtin, tab_model, tab_about = st.tabs([
    "上传文档问答",
    "内置知识库",
    "模型评估",
    "项目说明",
])

with tab_upload:
    st.markdown(
        """
<div class="section-title">
    <div class="section-eyebrow">Upload workspace</div>
    <h2>上传文档，即刻问答</h2>
    <p>上传文档后，系统会临时构建检索索引，并基于上传内容回答问题。适合论文、报告、产品文档、项目资料等多文档问答场景。</p>
</div>
<div class="notice"><strong>解析建议：</strong>文本型 PDF 和 Word 可直接解析；扫描版 PDF 建议先使用 MinerU/OCR 转为 Markdown 或 TXT 后再上传。</div>
""",
        unsafe_allow_html=True,
    )

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
        st.markdown("### 文档解析状态")
        st.dataframe(summary_df, use_container_width=True)

        total_chunks = sum(s.chunk_count for s in summaries)
        total_chars = sum(s.text_length for s in summaries)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("上传文件数", len(summaries))
        c2.metric("解析字符数", total_chars)
        c3.metric("文本块数量", total_chunks)
        c4.metric("检索方式", "TF-IDF + Cosine")

        with st.form("upload_question_form", clear_on_submit=False):
            question = st.text_input(
                "请输入针对上传文档的问题：",
                placeholder="例如：这份文档主要讲了什么？请总结三个重点。",
                key="upload_question",
            )
            action_col, clear_col, _ = st.columns([1.35, 1.15, 4])
            ask_upload = action_col.form_submit_button(
                "基于上传文档问答",
                type="primary",
                use_container_width=True,
            )
            clear_upload = clear_col.form_submit_button(
                "清空问答历史",
                use_container_width=True,
            )
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
    st.markdown(
        """
<div class="section-title">
    <div class="section-eyebrow">Built-in knowledge base</div>
    <h2>内置多文档知识库问答</h2>
    <p>内置知识库用于演示多文档跨文档问答能力。系统并不局限于这些文件，核心入口仍是上传文档问答。</p>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.form("builtin_question_form", clear_on_submit=False):
        builtin_question = st.text_input(
            "请输入你的问题：",
            placeholder="例如：什么是神经网络？AIGC 的主要应用场景有哪些？",
            key="builtin_question",
        )
        b1, b2 = st.columns([1, 1])
        ask_builtin = b1.form_submit_button(
            "检索内置知识库",
            type="primary",
            use_container_width=True,
        )
        clear_builtin = b2.form_submit_button(
            "清空问答历史",
            use_container_width=True,
        )
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
    st.markdown(
        """
<div class="section-title">
    <div class="section-eyebrow">Model evaluation</div>
    <h2>机器学习模型评估结果</h2>
    <p>项目使用自建文本分类数据集训练朴素贝叶斯、逻辑回归和 MLP 神经网络模型，并进行多指标评估。</p>
</div>
""",
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
    st.markdown(
        """
<div class="section-title">
    <div class="section-eyebrow">Project overview</div>
    <h2>项目说明与技术路线</h2>
    <p>从文档上传到答案生成，系统以轻量、可解释、可追溯为核心目标，适合作为智能体文档问答演示与课程项目展示。</p>
</div>
""",
        unsafe_allow_html=True,
    )
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
