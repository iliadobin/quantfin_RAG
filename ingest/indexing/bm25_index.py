"""
BM25 index build/save utilities.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi


_RE_TOK = re.compile(r"\b\w+\b", flags=re.UNICODE)


def simple_tokenize(text: str) -> List[str]:
    return _RE_TOK.findall((text or "").lower())


@dataclass(frozen=True)
class Bm25Artifacts:
    bm25_path: Path
    chunk_ids_path: Path


def build_bm25(texts: List[str]) -> BM25Okapi:
    tokenized = [simple_tokenize(t) for t in texts]
    return BM25Okapi(tokenized)


def save_bm25(
    out_dir: str | Path,
    *,
    bm25: BM25Okapi,
    chunk_ids: List[str],
) -> Bm25Artifacts:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bm25_path = out / "bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    chunk_ids_path = out / "chunk_ids.npy"
    np.save(chunk_ids_path, np.asarray(chunk_ids, dtype=object), allow_pickle=True)

    return Bm25Artifacts(bm25_path=bm25_path, chunk_ids_path=chunk_ids_path)


