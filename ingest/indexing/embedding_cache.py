"""
DuckDB-backed embedding cache.

We cache passage embeddings for chunks so repeated index builds (and future
retrieval debugging) are fast and reproducible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import duckdb
import numpy as np


@dataclass(frozen=True)
class EmbeddingCacheStats:
    hits: int
    misses: int
    inserted: int


class DuckDbEmbeddingCache:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Use a persistent DB file.
        self.con = duckdb.connect(str(self.db_path))
        self._init_schema()

    def close(self) -> None:
        try:
            self.con.close()
        except Exception:
            pass

    def _init_schema(self) -> None:
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
              model VARCHAR NOT NULL,
              chunk_id VARCHAR NOT NULL,
              dim INTEGER NOT NULL,
              vec BLOB NOT NULL,
              created_at TIMESTAMP DEFAULT now(),
              PRIMARY KEY (model, chunk_id)
            );
            """
        )

    @staticmethod
    def _pack(vec: np.ndarray) -> Tuple[int, bytes]:
        v = np.asarray(vec, dtype=np.float32)
        return int(v.shape[0]), v.tobytes(order="C")

    @staticmethod
    def _unpack(dim: int, blob: bytes) -> np.ndarray:
        v = np.frombuffer(blob, dtype=np.float32)
        if v.shape[0] != dim:
            raise ValueError(f"Embedding dim mismatch: expected {dim}, got {v.shape[0]}")
        return v

    def get_many(self, model: str, chunk_ids: List[str], *, batch_size: int = 500) -> Dict[str, np.ndarray]:
        """
        Returns a dict chunk_id -> embedding (float32).
        """
        if not chunk_ids:
            return {}
        out: Dict[str, np.ndarray] = {}
        for i in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[i : i + batch_size]
            placeholders = ",".join(["?"] * len(batch))
            rows = self.con.execute(
                f"SELECT chunk_id, dim, vec FROM embeddings WHERE model = ? AND chunk_id IN ({placeholders})",
                [model, *batch],
            ).fetchall()
            for cid, dim, blob in rows:
                out[str(cid)] = self._unpack(int(dim), blob)
        return out

    def put_many(self, model: str, items: Iterable[Tuple[str, np.ndarray]]) -> int:
        """
        Insert or replace embeddings. Returns number of rows inserted.
        """
        rows = []
        n = 0
        for chunk_id, vec in items:
            dim, blob = self._pack(vec)
            rows.append((model, chunk_id, dim, blob))
            n += 1

        if not rows:
            return 0
        self.con.executemany(
            "INSERT OR REPLACE INTO embeddings (model, chunk_id, dim, vec) VALUES (?, ?, ?, ?)",
            rows,
        )
        return n


def compute_e5_passage_embeddings(
    texts: List[str],
    *,
    model_name: str = "intfloat/e5-small-v2",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Compute normalized embeddings for passages using SentenceTransformers.
    """
    from sentence_transformers import SentenceTransformer

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    model = SentenceTransformer(model_name)

    # E5 recommends task prefixing.
    inputs = [f"passage: {t}" for t in texts]
    emb = model.encode(inputs, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(emb, dtype=np.float32)


def get_or_compute_passage_embeddings(
    cache: DuckDbEmbeddingCache,
    *,
    model_name: str,
    chunk_ids: List[str],
    texts: List[str],
    batch_size: int = 64,
    cache_lookup_batch: int = 800,
) -> Tuple[np.ndarray, EmbeddingCacheStats]:
    """
    For each (chunk_id, text) return embedding in the same order.
    """
    if len(chunk_ids) != len(texts):
        raise ValueError("chunk_ids and texts must be same length")

    cached = cache.get_many(model_name, chunk_ids, batch_size=cache_lookup_batch)
    hits = len(cached)
    misses_ids: List[str] = []
    misses_texts: List[str] = []
    for cid, txt in zip(chunk_ids, texts):
        if cid not in cached:
            misses_ids.append(cid)
            misses_texts.append(txt)

    inserted = 0
    if misses_ids:
        new_emb = compute_e5_passage_embeddings(misses_texts, model_name=model_name, batch_size=batch_size)
        cache.put_many(model_name, zip(misses_ids, list(new_emb)))
        inserted = len(misses_ids)
        for cid, vec in zip(misses_ids, new_emb):
            cached[cid] = np.asarray(vec, dtype=np.float32)

    # Assemble in original order
    matrix = np.stack([cached[cid] for cid in chunk_ids], axis=0) if chunk_ids else np.zeros((0, 0), dtype=np.float32)
    stats = EmbeddingCacheStats(hits=hits, misses=len(misses_ids), inserted=inserted)
    return matrix, stats


