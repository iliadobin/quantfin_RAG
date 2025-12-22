"""
FAISS vector index build/save utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
import numpy as np


@dataclass(frozen=True)
class FaissArtifacts:
    index_path: Path
    chunk_ids_path: Path


def build_faiss_ip(embeddings: np.ndarray) -> faiss.Index:
    """
    Build IndexFlatIP for cosine similarity (requires normalized embeddings).
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D [N, D]")
    n, d = embeddings.shape
    index = faiss.IndexFlatIP(d)
    if n:
        index.add(np.asarray(embeddings, dtype=np.float32))
    return index


def save_faiss(
    out_dir: str | Path,
    *,
    index: faiss.Index,
    chunk_ids: List[str],
) -> FaissArtifacts:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    index_path = out / "faiss.index"
    faiss.write_index(index, str(index_path))

    chunk_ids_path = out / "chunk_ids.npy"
    np.save(chunk_ids_path, np.asarray(chunk_ids, dtype=object), allow_pickle=True)

    return FaissArtifacts(index_path=index_path, chunk_ids_path=chunk_ids_path)


