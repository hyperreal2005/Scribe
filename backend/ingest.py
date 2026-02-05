import logging
from pathlib import Path
from typing import Iterable, List, Optional

from pypdf import PdfReader
from docx import Document

logger = logging.getLogger(__name__)

TEXT_EXTS = {".txt", ".md", ".markdown"}
PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx", ".doc"}


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp1252", errors="ignore")


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: List[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            parts.append(f"[Page {i}]\n{text}")
    return "\n\n".join(parts)


def _read_docx(path: Path) -> str:
    doc = Document(str(path))
    lines = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(lines)


def extract_text_from_path(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    try:
        if ext in TEXT_EXTS:
            return _read_text_file(path)
        if ext in PDF_EXTS:
            return _read_pdf(path)
        if ext in DOCX_EXTS:
            return _read_docx(path)
        return None
    except Exception as exc:
        logger.error("Failed to read %s: %s", path, exc)
        return None


def scan_folder(folder: Path, recursive: bool = True) -> Iterable[Path]:
    if not folder.exists():
        return []
    patterns = ["*.txt", "*.md", "*.markdown", "*.pdf", "*.docx", "*.doc"]
    paths: List[Path] = []
    for pattern in patterns:
        if recursive:
            paths.extend(folder.rglob(pattern))
        else:
            paths.extend(folder.glob(pattern))
    return paths
