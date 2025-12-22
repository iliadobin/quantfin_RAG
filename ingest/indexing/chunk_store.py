"""
Loading chunks from parsed artifacts and writing a consolidated chunk store.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

from knowledge.models import Chunk


def iter_chunk_files(parsed_root: str | Path, doc_ids: Iterable[str], strategy: str) -> Iterable[Path]:
    root = Path(parsed_root)
    filename = "chunks_fixed.jsonl" if strategy == "fixed" else "chunks_section_aware.jsonl"
    for doc_id in doc_ids:
        p = root / doc_id / filename
        if p.exists():
            yield p


def load_chunks_from_parsed(
    parsed_root: str | Path,
    *,
    doc_ids: List[str],
    strategy: str,
    limit_chunks: Optional[int] = None,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for f in iter_chunk_files(parsed_root, doc_ids, strategy):
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                chunks.append(Chunk.model_validate(obj))
                if limit_chunks is not None and len(chunks) >= limit_chunks:
                    return chunks
    return chunks


def write_chunk_store(out_path: str | Path, chunks: List[Chunk]) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch.model_dump(mode="json"), ensure_ascii=False) + "\n")


