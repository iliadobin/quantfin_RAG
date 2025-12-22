#!/usr/bin/env python3
"""
Build retrieval indices for the parsed corpus:
  - BM25 (rank-bm25)
  - Vector index (FAISS) using local embeddings (SentenceTransformers)
  - Embedding cache (DuckDB) to avoid recomputation

Output layout:
  data/indices/<profile>/<strategy>/
    - chunks.jsonl
    - bm25.pkl
    - faiss.index
    - chunk_ids.npy
    - report.json

Embedding cache:
  data/cache/embeddings.duckdb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ingest.manifest import load_corpus_manifest  # noqa: E402
from ingest.indexing.chunk_store import load_chunks_from_parsed, write_chunk_store  # noqa: E402
from ingest.indexing.bm25_index import build_bm25, save_bm25  # noqa: E402
from ingest.indexing.vector_index import build_faiss_ip, save_faiss  # noqa: E402
from ingest.indexing.embedding_cache import (  # noqa: E402
    DuckDbEmbeddingCache,
    get_or_compute_passage_embeddings,
)


def _strategy_choices() -> List[str]:
    return ["fixed", "section_aware", "both"]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs/corpus_public.yaml")
    ap.add_argument("--profile", default="public")
    ap.add_argument("--parsed-root", default="data/parsed")
    ap.add_argument("--out-root", default="data/indices")

    ap.add_argument("--strategy", default="fixed", choices=_strategy_choices())
    ap.add_argument("--embedding-model", default="intfloat/e5-small-v2")
    ap.add_argument("--embed-batch-size", type=int, default=64)
    ap.add_argument("--cache-db", default="data/cache/embeddings.duckdb")
    ap.add_argument("--limit-docs", type=int, default=None)
    ap.add_argument("--limit-chunks", type=int, default=None)
    args = ap.parse_args(argv)

    manifest = load_corpus_manifest(project_root / args.manifest)
    doc_ids = [d.id for d in manifest.documents]
    if args.limit_docs is not None:
        doc_ids = doc_ids[: args.limit_docs]

    strategies = ["fixed", "section_aware"] if args.strategy == "both" else [args.strategy]

    cache = DuckDbEmbeddingCache(project_root / args.cache_db)
    try:
        reports = []
        for strat in strategies:
            print("\n" + "=" * 80)
            print(f"Building indices: profile={args.profile} strategy={strat}")
            print("=" * 80)

            chunks = load_chunks_from_parsed(
                project_root / args.parsed_root,
                doc_ids=doc_ids,
                strategy=strat,
                limit_chunks=args.limit_chunks,
            )
            if not chunks:
                print(f"No chunks found for strategy={strat}. Did you run scripts/parse_corpus.py?")
                continue

            out_dir = (project_root / args.out_root / args.profile / strat).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

            # Consolidated store (used later by retrievers)
            write_chunk_store(out_dir / "chunks.jsonl", chunks)

            chunk_ids = [c.id for c in chunks]
            texts = [c.text for c in chunks]

            # BM25
            print(f"- BM25: tokenizing {len(texts)} chunks...")
            bm25 = build_bm25(texts)
            save_bm25(out_dir, bm25=bm25, chunk_ids=chunk_ids)

            # Vector (FAISS)
            print(f"- Embeddings: loading/caching model={args.embedding_model} ...")
            embeddings, stats = get_or_compute_passage_embeddings(
                cache,
                model_name=args.embedding_model,
                chunk_ids=chunk_ids,
                texts=texts,
                batch_size=args.embed_batch_size,
            )
            print(f"  cache: hits={stats.hits} misses={stats.misses} inserted={stats.inserted}")

            print("- FAISS: building IndexFlatIP...")
            index = build_faiss_ip(embeddings)
            save_faiss(out_dir, index=index, chunk_ids=chunk_ids)

            report = {
                "profile": args.profile,
                "strategy": strat,
                "manifest": str(args.manifest),
                "parsed_root": str(args.parsed_root),
                "chunk_count": len(chunks),
                "embedding_model": args.embedding_model,
                "embedding_cache": {
                    "db_path": str(args.cache_db),
                    "hits": stats.hits,
                    "misses": stats.misses,
                    "inserted": stats.inserted,
                },
            }
            with open(out_dir / "report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            reports.append(report)

        summary_path = (project_root / args.out_root / args.profile / "indices_report.json").resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"reports": reports}, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Indices written to: {project_root / args.out_root / args.profile}")
        return 0
    finally:
        cache.close()


if __name__ == "__main__":
    raise SystemExit(main())


