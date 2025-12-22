"""
Section-aware chunking (lightweight heuristics).

This strategy attempts to detect headings and keeps a `section_path` to improve
retrieval quality and debugging. It may span multiple pages, so char offsets are
optional (page-level span is always provided).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from knowledge.models import Chunk, PageSpan, ParsedPage


_RE_NUMBERED_HEADING = re.compile(r"^\s*(\d+(?:\.\d+){0,3})\s+(.+?)\s*$")
_RE_ROMAN_HEADING = re.compile(r"^\s*([IVXLC]{1,6})\.\s+(.+?)\s*$")


def _is_all_caps(s: str) -> bool:
    letters = [ch for ch in s if ch.isalpha()]
    if len(letters) < 6:
        return False
    return all(ch.isupper() for ch in letters)


def _heading_depth_and_title(line: str) -> Tuple[int, str] | None:
    ln = line.strip()
    if not (3 <= len(ln) <= 100):
        return None

    m = _RE_NUMBERED_HEADING.match(ln)
    if m:
        numbering = m.group(1)
        title = m.group(2).strip()
        depth = numbering.count(".") + 1
        return depth, f"{numbering} {title}"

    m = _RE_ROMAN_HEADING.match(ln)
    if m:
        return 1, f"{m.group(1)}. {m.group(2).strip()}"

    # Heuristic: ALL CAPS line with no trailing period often looks like a heading
    if _is_all_caps(ln) and not ln.endswith("."):
        return 1, ln

    return None


def _stable_section_chunk_id(doc_id: str, section_key: str, idx: int) -> str:
    key = f"{doc_id}|section_aware|{section_key}|{idx}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class SectionAwareChunkerConfig:
    max_chars: int = 2400
    overlap_chars: int = 200
    min_chars: int = 300


def _split_with_overlap(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    step = max(1, max_chars - overlap_chars)
    parts: List[str] = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        parts.append(text[i:j].strip())
        if j == len(text):
            break
        i += step
    return [p for p in parts if p]


def chunk_pages_section_aware(
    pages: List[ParsedPage],
    *,
    doc_id: str,
    cfg: SectionAwareChunkerConfig | None = None,
) -> Iterable[Chunk]:
    if cfg is None:
        cfg = SectionAwareChunkerConfig()

    # Maintain a simple heading stack by depth.
    section_stack: List[str] = []
    current_buffer: List[str] = []
    current_start_page: int | None = None
    current_end_page: int | None = None

    def flush_buffer() -> Iterable[Chunk]:
        nonlocal current_buffer, current_start_page, current_end_page
        text = "\n".join(current_buffer).strip()
        if not text or current_start_page is None or current_end_page is None:
            current_buffer = []
            current_start_page = None
            current_end_page = None
            return []

        section_path = list(section_stack)
        section_key = " > ".join(section_path) if section_path else "ROOT"
        parts = _split_with_overlap(text, cfg.max_chars, cfg.overlap_chars)
        out: List[Chunk] = []
        for idx, part in enumerate(parts):
            if len(part) < cfg.min_chars:
                continue
            cid = _stable_section_chunk_id(doc_id, section_key, idx)
            out.append(
                Chunk(
                    id=f"{doc_id}_section_{cid}",
                    doc_id=doc_id,
                    strategy="section_aware",
                    text=part,
                    page_span=PageSpan(start_page=current_start_page, end_page=current_end_page),
                    section_path=section_path,
                    token_count=len(part.split()),
                    char_count=len(part),
                    metadata={"section_key": section_key},
                )
            )

        current_buffer = []
        current_start_page = None
        current_end_page = None
        return out

    emitted: List[Chunk] = []
    for p in pages:
        lines = (p.text or "").splitlines()
        if current_start_page is None:
            current_start_page = p.page_number
        current_end_page = p.page_number

        # Process line-by-line and detect headings.
        for ln in lines:
            h = _heading_depth_and_title(ln)
            if h:
                # Flush content accumulated under previous heading context.
                emitted.extend(flush_buffer())
                depth, title = h
                # Update stack.
                if depth <= 0:
                    depth = 1
                section_stack[:] = section_stack[: depth - 1]
                section_stack.append(title)
                # Start buffer after heading line (skip heading itself).
                if current_start_page is None:
                    current_start_page = p.page_number
                current_end_page = p.page_number
                continue
            current_buffer.append(ln)

    emitted.extend(flush_buffer())
    return emitted


