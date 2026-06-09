from pathlib import Path

def read_file(path: str) -> str:
    p = Path(path)
    if p.suffix == ".txt":
        return p.read_text(encoding="utf-8")
    elif p.suffix == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() for page in PdfReader(str(p)).pages)
    elif p.suffix == ".docx":
        from docx import Document
        return "\n".join(para.text for para in Document(str(p)).paragraphs)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")