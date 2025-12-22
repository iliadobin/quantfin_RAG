"""
Ingest pipeline: PDF -> normalized pages -> chunks (2 strategies).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any

from knowledge.models import ParsedPage, Chunk

from ingest.parsing.pdf_parser import extract_pdf_pages
from ingest.normalization.text_normalizer import normalize_page_text, strip_repeated_headers_footers
from ingest.chunking.fixed import chunk_pages_fixed, FixedChunkerConfig
from ingest.chunking.section_aware import chunk_pages_section_aware, SectionAwareChunkerConfig


@dataclass(frozen=True)
class ParseConfig:
    preferred_parser: Literal["auto", "pymupdf", "pdfplumber"] = "auto"
    max_pages: Optional[int] = None
    strip_headers_footers: bool = True


@dataclass(frozen=True)
class ChunkingConfig:
    fixed: FixedChunkerConfig = FixedChunkerConfig()
    section_aware: SectionAwareChunkerConfig = SectionAwareChunkerConfig()


def parse_pdf_to_pages(
    *,
    doc_id: str,
    pdf_path: str | Path,
    cfg: ParseConfig | None = None,
) -> List[ParsedPage]:
    if cfg is None:
        cfg = ParseConfig()

    parsed = extract_pdf_pages(pdf_path, preferred=cfg.preferred_parser, max_pages=cfg.max_pages)

    raw_pages = parsed.pages
    norm_pages = [normalize_page_text(t) for t in raw_pages]
    if cfg.strip_headers_footers:
        norm_pages = strip_repeated_headers_footers(norm_pages)

    out: List[ParsedPage] = []
    for idx, (raw, norm) in enumerate(zip(raw_pages, norm_pages), start=1):
        out.append(
            ParsedPage(
                doc_id=doc_id,
                page_number=idx,
                text=norm,
                char_count=len(norm),
                raw_char_count=len(raw or ""),
                extraction_method=parsed.method,
            )
        )
    return out


def pages_to_chunks(
    pages: List[ParsedPage],
    *,
    doc_id: str,
    cfg: ChunkingConfig | None = None,
) -> Dict[str, List[Chunk]]:
    if cfg is None:
        cfg = ChunkingConfig()

    fixed_chunks = list(chunk_pages_fixed(pages, doc_id=doc_id, cfg=cfg.fixed))
    section_chunks = list(chunk_pages_section_aware(pages, doc_id=doc_id, cfg=cfg.section_aware))
    return {"fixed": fixed_chunks, "section_aware": section_chunks}


def build_parse_report(
    *,
    doc_id: str,
    pdf_path: str | Path,
    pages: List[ParsedPage],
    chunks_by_strategy: Dict[str, List[Chunk]],
) -> Dict[str, Any]:
    return {
        "doc_id": doc_id,
        "pdf_path": str(pdf_path),
        "page_count": len(pages),
        "page_chars_total": sum(p.char_count for p in pages),
        "chunks": {k: len(v) for k, v in chunks_by_strategy.items()},
    }


