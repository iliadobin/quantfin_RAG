"""
Text normalization for parsed PDF pages.

This is intentionally conservative: normalize unicode, fix hyphenation across
line breaks, collapse excessive whitespace, and (optionally) strip repeated
headers/footers.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional


_RE_HYPHEN_LINEBREAK = re.compile(r"(?<=\w)-\n(?=\w)")
_RE_SPACES = re.compile(r"[ \t]+")
_RE_MANY_NEWLINES = re.compile(r"\n{3,}")


def normalize_page_text(text: str) -> str:
    if not text:
        return ""

    # Unicode normalize, strip soft hyphens
    t = unicodedata.normalize("NFKC", text).replace("\u00ad", "")

    # Fix hyphenation at line breaks: "volatil-\nity" -> "volatility"
    t = _RE_HYPHEN_LINEBREAK.sub("", t)

    # Normalize whitespace
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _RE_SPACES.sub(" ", t)

    # Trim line edges, keep line breaks
    t = "\n".join(line.strip() for line in t.split("\n"))

    # Collapse excessive blank lines
    t = _RE_MANY_NEWLINES.sub("\n\n", t).strip()

    return t


@dataclass(frozen=True)
class HeaderFooterStripConfig:
    lines_to_check: int = 2
    min_fraction: float = 0.60  # repeated on >= 60% pages
    max_line_len: int = 140     # ignore huge lines


def _top_bottom_lines(text: str, n: int) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    tops = lines[:n]
    bottoms = lines[-n:] if len(lines) > n else []
    return tops + bottoms


def strip_repeated_headers_footers(
    pages: List[str],
    *,
    cfg: Optional[HeaderFooterStripConfig] = None,
) -> List[str]:
    """
    Heuristic header/footer stripper.

    We look at first/last N non-empty lines on each page. Any line repeated on
    >= min_fraction of pages (casefolded) is stripped from those boundary lines.
    """
    if cfg is None:
        cfg = HeaderFooterStripConfig()
    if not pages:
        return pages

    candidates: List[str] = []
    for p in pages:
        for ln in _top_bottom_lines(p, cfg.lines_to_check):
            if 0 < len(ln) <= cfg.max_line_len:
                candidates.append(ln.casefold())

    if not candidates:
        return pages

    counts = Counter(candidates)
    threshold = max(2, int(len(pages) * cfg.min_fraction))
    repeated = {ln for ln, c in counts.items() if c >= threshold}
    if not repeated:
        return pages

    stripped_pages: List[str] = []
    for p in pages:
        raw_lines = p.splitlines()
        # Identify which physical line indices are in the boundary windows
        nonempty_indices = [i for i, ln in enumerate(raw_lines) if ln.strip()]
        boundary: set[int] = set()
        for i in nonempty_indices[: cfg.lines_to_check]:
            boundary.add(i)
        for i in nonempty_indices[-cfg.lines_to_check :]:
            boundary.add(i)

        new_lines: List[str] = []
        for i, ln in enumerate(raw_lines):
            if i in boundary and ln.strip() and ln.strip().casefold() in repeated:
                continue
            new_lines.append(ln)
        stripped_pages.append("\n".join(new_lines).strip())

    return stripped_pages


