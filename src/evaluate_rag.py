import pandas as pd
from src.rag_engine import RAGEngine
from src.config import REPORTS_DIR


def main():
    engine = RAGEngine()
    questions = [
        {"question": "什么是神经网络？", "expected_keyword": "神经网络"},
        {"question": "AIGC 的主要应用场景有哪些？", "expected_keyword": "AIGC"},
        {"question": "What are the top takeaways of the AI Index Report 2025?", "expected_keyword": "AI Index"},
    ]
    rows = []
    for item in questions:
        contexts = engine.retrieve(item["question"], top_k=5)
        joined = " ".join(c["source_file"] + " " + c["text"] for c in contexts)
        hit = item["expected_keyword"].lower() in joined.lower()
        rows.append({
            "question": item["question"],
            "expected_keyword": item["expected_keyword"],
            "hit_top5": int(hit),
            "top_sources": "; ".join(c["source_file"] for c in contexts[:3])
        })
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "rag_eval_result.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
