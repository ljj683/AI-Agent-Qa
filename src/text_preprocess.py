import json
import os
import re
from pathlib import Path
from src.config import PARSED_DIR, STORAGE_DIR


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.strip()


def split_text(text: str, chunk_size: int = 500, overlap: int = 100):
    if chunk_size <= overlap:
        raise ValueError("chunk_size 必须大于 overlap")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) >= 50:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_chunks(input_dir: Path = PARSED_DIR, output_path: Path = STORAGE_DIR / "chunks.jsonl"):
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    chunk_records = []

    for file_path in input_dir.glob("*.txt"):
        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
        if not raw_text or len(raw_text.strip()) == 0:
            continue
        cleaned = clean_text(raw_text)
        if len(cleaned) < 100:
            continue
        chunks = split_text(cleaned)
        for i, chunk in enumerate(chunks):
            chunk_records.append({
                "chunk_id": f"{file_path.stem}_{i}",
                "source_file": file_path.name,
                "text": chunk,
                "length": len(chunk)
            })

    # 去重
    unique = {}
    for item in chunk_records:
        unique[item["text"]] = item
    final_records = list(unique.values())

    with output_path.open("w", encoding="utf-8") as f:
        for item in final_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"共生成 {len(final_records)} 个有效文本块：{output_path}")
    return final_records


if __name__ == "__main__":
    build_chunks()
