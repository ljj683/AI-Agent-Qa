import os
from pathlib import Path
import fitz
import docx
from src.config import RAW_DOCS_DIR, PARSED_DIR, REPORTS_DIR


def parse_pdf(file_path: Path) -> str:
    text_list = []
    with fitz.open(file_path) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            if text and text.strip():
                text_list.append(f"\n[第{page_num}页]\n{text}")
    return "\n".join(text_list)


def parse_docx(file_path: Path) -> str:
    document = docx.Document(file_path)
    paragraphs = []
    for para in document.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def parse_document(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(file_path)
    if suffix == ".docx":
        return parse_docx(file_path)
    raise ValueError(f"暂不支持该文件类型：{suffix}")


def batch_parse_docs(input_dir: Path = RAW_DOCS_DIR, output_dir: Path = PARSED_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    warnings = []

    for file_path in input_dir.rglob("*"):
        if file_path.suffix.lower() not in [".pdf", ".docx"]:
            continue
        try:
            text = parse_document(file_path)
            out_name = file_path.stem + ".txt"
            out_path = output_dir / out_name
            out_path.write_text(text, encoding="utf-8", errors="ignore")

            record = {
                "file_name": file_path.name,
                "relative_path": str(file_path.relative_to(input_dir)),
                "parsed_path": str(out_path),
                "char_count": len(text)
            }
            records.append(record)

            if len(text) < 1000:
                warnings.append(
                    f"{file_path.name} 直接解析字符数较少（{len(text)}），可能是扫描/图片型PDF，建议用 MinerU/OCR 转 Markdown 后再运行 convert_md_to_txt.py。"
                )
            print(f"解析完成：{file_path.name}，字符数：{len(text)}")
        except Exception as e:
            warnings.append(f"解析失败：{file_path.name}，错误：{e}")
            print(f"解析失败：{file_path.name}，错误：{e}")

    import json
    (REPORTS_DIR / "parse_records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "parse_warnings.txt").write_text("\n".join(warnings), encoding="utf-8")


if __name__ == "__main__":
    batch_parse_docs()
