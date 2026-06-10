import json
from pathlib import Path
import numpy as np
import joblib
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, EMBEDDING_MODEL, TOP_K, STORAGE_DIR


class RAGEngine:
    def __init__(self):
        self.storage_dir = STORAGE_DIR
        self.chunks = json.loads((self.storage_dir / "chunk_meta.json").read_text(encoding="utf-8"))
        self.vector_meta = json.loads((self.storage_dir / "vector_store_meta.json").read_text(encoding="utf-8"))
        self.backend = self.vector_meta.get("backend")

        self.classifier = joblib.load(self.storage_dir / "ml_models" / "best_text_classifier.pkl")

        if self.backend == "sentence_transformer":
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            self.embeddings = np.load(self.storage_dir / "embeddings.npy")
        elif self.backend == "tfidf":
            self.vectorizer = joblib.load(self.storage_dir / "tfidf_vectorizer.joblib")
            self.tfidf_matrix = joblib.load(self.storage_dir / "tfidf_matrix.joblib")
        else:
            raise ValueError(f"未知向量库后端：{self.backend}")

        self.client = None
        if DEEPSEEK_API_KEY:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
            except Exception as e:
                print(f"OpenAI/DeepSeek 客户端初始化失败，将使用本地降级回答。原因：{e}")
                self.client = None

    def classify_question(self, question: str) -> str:
        try:
            return str(self.classifier.predict([question])[0])
        except Exception:
            return "未分类"

    def retrieve(self, question: str, top_k: int = TOP_K):
        if self.backend == "sentence_transformer":
            query_vec = self.embedding_model.encode([question], normalize_embeddings=True)[0]
            scores = np.dot(self.embeddings, query_vec)
        else:
            q = self.vectorizer.transform([question])
            scores = (self.tfidf_matrix @ q.T).toarray().ravel()

        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            item = dict(self.chunks[int(idx)])
            item["score"] = float(scores[int(idx)])
            results.append(item)
        return results

    def generate_answer(self, question: str, contexts: list, history: list | None = None):
        context_text = "\n\n".join([
            f"资料{i+1}｜来源：{c['source_file']}｜相似度：{c['score']:.4f}\n{c['text']}"
            for i, c in enumerate(contexts)
        ])

        history_text = ""
        if history:
            for h in history[-3:]:
                history_text += f"用户：{h.get('question', '')}\n助手：{h.get('answer', '')}\n"

        top_score = contexts[0]["score"] if contexts else 0

        prompt = f"""
        你是一个基于内置多文档知识库的问答助手。请根据系统检索到的知识库片段回答用户问题。

        注意：
        1. 当前内容来自“内置知识库”，不是用户临时上传文档；
        2. 只要检索片段中包含与问题相关的术语、定义、特征、应用、步骤或示例，就应结合片段内容进行概括回答；
        3. 只有当检索片段与用户问题完全无关时，才回答“内置知识库中没有找到明确依据”；
        4. 不要编造检索片段之外的具体数据、年份、结论或来源；
        5. 回答要结构清晰，适合学生理解；
        6. 回答结尾简要说明参考依据来自内置知识库检索结果。

        历史对话：
        {history_text}

        本次最高相似度：
        {top_score:.4f}

        检索到的知识库内容：
        {context_text}

        用户问题：
        {question}

        请基于上述知识库内容给出回答：
        """
        if not self.client:
            # 无 API Key 时的本地降级，保证课堂离线调试时系统仍能返回检索结果。
            return (
                "当前未配置 DeepSeek API Key，以下为基于检索片段的本地降级回答：\n\n"
                + "\n\n".join([f"【{i+1}】来源：{c['source_file']}\n{c['text'][:300]}" for i, c in enumerate(contexts)])
            )

        response = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system",
                 "content": "你是基于内置知识库 RAG 的文档问答助手，需要结合检索片段进行准确、简洁、可追溯的回答。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content

    def ask(self, question: str, history: list | None = None):
        question_type = self.classify_question(question)
        contexts = self.retrieve(question)
        answer = self.generate_answer(question, contexts, history)
        return {
            "question": question,
            "question_type": question_type,
            "answer": answer,
            "contexts": contexts,
        }


if __name__ == "__main__":
    engine = RAGEngine()
    result = engine.ask("什么是神经网络？")
    print(result["answer"])
