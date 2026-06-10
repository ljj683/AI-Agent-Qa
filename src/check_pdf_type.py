from pathlib import Path
import fitz
from src.config import RAW_DOCS_DIR, REPORTS_DIR


def check_pdf(pdf_path: Path) -> dict:
    doc = fitz.open(pdf_path)
    total_chars = 0
    page_chars = []
    for page in doc:
        text = page.get_text("text").strip()
        page_chars.append(len(text))
        total_chars += len(text)
    doc.close()
    return {
        "file": str(pdf_path),
        "pages": len(page_chars),
        "total_chars": total_chars,
        "avg_chars_per_page": round(total_chars / max(len(page_chars), 1), 2),
        "type": "扫描/图片型PDF，建议使用MinerU或OCR" if total_chars < 1000 else "文本型PDF，可直接解析"
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = list(RAW_DOCS_DIR.rglob("*.pdf"))
    lines = []
    for pdf in pdf_files:
        result = check_pdf(pdf)
        msg = (
            f"文件：{result['file']}\n"
            f"页数：{result['pages']}\n"
            f"直接提取字符数：{result['total_chars']}\n"
            f"平均每页字符数：{result['avg_chars_per_page']}\n"
            f"判断：{result['type']}\n"
            + "=" * 60
        )
        print(msg)
        lines.append(msg)
    (REPORTS_DIR / "pdf_type_check.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
