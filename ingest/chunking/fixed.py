"""
Fixed-size chunking (page-local) with character overlap.

This strategy is simple and produces precise page-span citations (including
char offsets within a single page).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, List

from knowledge.models import Chunk, PageSpan, ParsedPage


def _stable_chunk_id(doc_id: str, strategy: str, page_number: int, start_char: int, end_char: int) -> str:
    key = f"{doc_id}|{strategy}|p{page_number}|{start_char}:{end_char}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class FixedChunkerConfig:
    max_chars: int = 1800
    overlap_chars: int = 200
    min_chars: int = 250


def _snap_end_to_whitespace(text: str, end: int, *, window: int = 80) -> int:
    """Try to end chunk on a whitespace boundary for nicer snippets."""
    if end >= len(text):
        return len(text)
    lo = max(0, end - window)
    hi = min(len(text), end + window)
    # Prefer searching forward a little, else back.
    forward = text[end:hi]
    fpos = forward.find(" ")
    if fpos != -1:
        return end + fpos
    backward = text[lo:end]
    bpos = backward.rfind(" ")
    if bpos != -1:
        return lo + bpos
    return end


def chunk_pages_fixed(
    pages: List[ParsedPage],
    *,
    doc_id: str,
    cfg: FixedChunkerConfig | None = None,
) -> Iterable[Chunk]:
    if cfg is None:
        cfg = FixedChunkerConfig()

    step = max(1, cfg.max_chars - cfg.overlap_chars)
    for p in pages:
        text = (p.text or "").strip()
        if not text:
            continue

        i = 0
        while i < len(text):
            j = min(len(text), i + cfg.max_chars)
            j = _snap_end_to_whitespace(text, j)
            chunk_text = text[i:j].strip()
            if len(chunk_text) >= cfg.min_chars:
                start_char = i
                end_char = j
                chunk_id = _stable_chunk_id(doc_id, "fixed", p.page_number, start_char, end_char)
                yield Chunk(
                    id=f"{doc_id}_fixed_{chunk_id}",
                    doc_id=doc_id,
                    strategy="fixed",
                    text=chunk_text,
                    page_span=PageSpan(
                        start_page=p.page_number,
                        end_page=p.page_number,
                        start_char=start_char,
                        end_char=end_char,
                    ),
                    section_path=[],
                    token_count=len(chunk_text.split()),
                    char_count=len(chunk_text),
                    metadata={"page_char_count": p.char_count},
                )
            if j == len(text):
                break
            i += step


