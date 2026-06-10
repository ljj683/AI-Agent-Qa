from mcp.server.fastmcp import FastMCP
from src.rag_engine import RAGEngine

mcp = FastMCP("doc_qa_mcp_server")
engine = RAGEngine()


@mcp.tool()
def ask_document(question: str) -> str:
    """基于已解析文档进行 RAG 问答。"""
    return engine.ask(question)["answer"]


@mcp.tool()
def classify_question(question: str) -> str:
    """对用户问题进行文本分类。"""
    return engine.classify_question(question)


@mcp.tool()
def retrieve_context(question: str) -> str:
    """根据问题检索最相关的文档片段。"""
    contexts = engine.retrieve(question)
    text = ""
    for i, ctx in enumerate(contexts, start=1):
        text += f"\n片段{i}｜来源：{ctx['source_file']}｜相似度：{ctx['score']:.4f}\n"
        text += ctx["text"][:500] + "\n"
    return text


if __name__ == "__main__":
    mcp.run()
