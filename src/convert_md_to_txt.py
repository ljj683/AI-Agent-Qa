"""将 MinerU 输出的 Markdown 转换为 RAG 知识库可用的 txt。

使用方法：
1. 把 MinerU 输出的 .md 文件放到 data/parsed/，建议命名为：机器学习基础及应用_mineru.md
2. 运行：python -m src.convert_md_to_txt
3. 程序会生成同名 txt，例如：data/parsed/机器学习基础及应用.txt
"""

from pathlib import Path
import re
from src.config import PARSED_DIR


def markdown_to_plain_text(markdown: str) -> str:
    text = markdown.replace("\r\n", "\n").replace("\r", "\n")

    # 删除 MinerU 图片引用，避免把 images/xxx.jpg 写进知识库
    text = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "\n", text)

    # Markdown 普通链接保留可读文本
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # HTML 表格尽量转成普通文本
    text = re.sub(r"</t[dh]>", " ", text, flags=re.I)
    text = re.sub(r"</tr>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)

    # 去除标题、列表、强调、代码、表格分隔符等 Markdown 标记
    text = re.sub(r"^[ \t]*#{1,6}[ \t]*", "", text, flags=re.M)
    text = re.sub(r"^[ \t]*[-*+][ \t]+", "", text, flags=re.M)
    text = re.sub(r"[`*_>|]", " ", text)

    # 清理空行和多余空格
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line:
            continue
        if line.lower().startswith("images/"):
            continue
        lines.append(line)

    return "\n".join(lines).strip()


def convert_one(md_path: Path, txt_path: Path) -> None:
    markdown = md_path.read_text(encoding="utf-8", errors="ignore")
    text = markdown_to_plain_text(markdown)
    txt_path.write_text(text, encoding="utf-8")
    print(f"转换完成：{md_path.name} -> {txt_path.name}，字符数：{len(text)}")


def main() -> None:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    md_files = list(PARSED_DIR.glob("*.md"))
    if not md_files:
        print("未发现 Markdown 文件。请先用 MinerU 将扫描版课程 PDF 转成 .md，并放入 data/parsed/。")
        return

    for md_path in md_files:
        stem = md_path.stem.replace("_mineru", "").replace("MinerU_markdown", "机器学习基础及应用")
        # 如果是课程教材 MinerU 文件，统一输出为课程知识库文件名
        if "机器学习" in md_path.stem or "MinerU" in md_path.stem:
            txt_path = PARSED_DIR / "机器学习基础及应用.txt"
        else:
            txt_path = PARSED_DIR / f"{stem}.txt"
        convert_one(md_path, txt_path)


if __name__ == "__main__":
    main()
