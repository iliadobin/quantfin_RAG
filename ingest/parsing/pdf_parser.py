"""
PDF parsing utilities.

Goal: extract per-page text deterministically, so we can later chunk while
preserving page spans for citations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Literal


ExtractionMethod = Literal["pymupdf", "pdfplumber"]


@dataclass(frozen=True)
class ParsedPdf:
    pages: List[str]          # raw extracted text per page (1-indexed conceptually)
    method: ExtractionMethod  # which extractor was used


def extract_pages_pymupdf(pdf_path: Path, max_pages: Optional[int] = None) -> List[str]:
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    try:
        n = doc.page_count
        if max_pages is not None:
            n = min(n, max_pages)
        pages: List[str] = []
        for i in range(n):
            page = doc.load_page(i)
            pages.append(page.get_text("text") or "")
        return pages
    finally:
        doc.close()


def extract_pages_pdfplumber(pdf_path: Path, max_pages: Optional[int] = None) -> List[str]:
    import pdfplumber

    pages: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        n = len(pdf.pages)
        if max_pages is not None:
            n = min(n, max_pages)
        for i in range(n):
            p = pdf.pages[i]
            pages.append(p.extract_text() or "")
    return pages


def extract_pdf_pages(
    pdf_path: str | Path,
    *,
    preferred: Literal["auto", "pymupdf", "pdfplumber"] = "auto",
    max_pages: Optional[int] = None,
) -> ParsedPdf:
    """
    Extract raw page texts.

    - preferred='auto': try PyMuPDF first, then pdfplumber.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    errors: List[str] = []
    if preferred in ("auto", "pymupdf"):
        try:
            return ParsedPdf(pages=extract_pages_pymupdf(path, max_pages=max_pages), method="pymupdf")
        except Exception as e:
            errors.append(f"pymupdf: {e}")

    if preferred in ("auto", "pdfplumber"):
        try:
            return ParsedPdf(pages=extract_pages_pdfplumber(path, max_pages=max_pages), method="pdfplumber")
        except Exception as e:
            errors.append(f"pdfplumber: {e}")

    raise RuntimeError(f"Failed to parse PDF {path}. Errors: {errors}")


