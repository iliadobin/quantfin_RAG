#!/usr/bin/env python3
"""
Parse the public PDF corpus into normalized pages and chunks (2 strategies).

Outputs under data/parsed/<doc_id>/:
  - pages.jsonl
  - chunks_fixed.jsonl
  - chunks_section_aware.jsonl
  - meta.json
  - report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# Ensure repo root on sys.path (consistent with other scripts)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ingest.artifacts import ensure_dir, write_jsonl  # noqa: E402
from ingest.manifest import load_corpus_manifest  # noqa: E402
from ingest.pipeline import (  # noqa: E402
    ParseConfig,
    ChunkingConfig,
    parse_pdf_to_pages,
    pages_to_chunks,
    build_parse_report,
)


def _abs_pdf_path(pdf_path_from_manifest: str) -> Path:
    p = Path(pdf_path_from_manifest)
    return (project_root / p).resolve() if not p.is_absolute() else p


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs/corpus_public.yaml")
    ap.add_argument("--out", default="data/parsed")
    ap.add_argument("--doc-id", default=None, help="If set, parse only this doc_id")
    ap.add_argument("--parser", default="auto", choices=["auto", "pymupdf", "pdfplumber"])
    ap.add_argument("--max-pages", type=int, default=None, help="Optional cap per document (debug)")
    ap.add_argument("--no-strip-headers", action="store_true", help="Disable header/footer stripping")
    args = ap.parse_args(argv)

    manifest = load_corpus_manifest(project_root / args.manifest)
    out_root = ensure_dir(project_root / args.out)

    parse_cfg = ParseConfig(
        preferred_parser=args.parser,
        max_pages=args.max_pages,
        strip_headers_footers=not args.no_strip_headers,
    )
    chunk_cfg = ChunkingConfig()

    docs = manifest.documents
    if args.doc_id:
        docs = [d for d in docs if d.id == args.doc_id]
        if not docs:
            raise SystemExit(f"doc_id not found in manifest: {args.doc_id}")

    reports = []
    for d in tqdm(docs, desc="Parsing corpus"):
        if not d.pdf_path:
            continue
        pdf_path = _abs_pdf_path(d.pdf_path)
        doc_dir = ensure_dir(out_root / d.id)

        pages = parse_pdf_to_pages(doc_id=d.id, pdf_path=pdf_path, cfg=parse_cfg)
        chunks_by_strategy = pages_to_chunks(pages, doc_id=d.id, cfg=chunk_cfg)
        report = build_parse_report(
            doc_id=d.id, pdf_path=pdf_path, pages=pages, chunks_by_strategy=chunks_by_strategy
        )
        reports.append(report)

        # Write artifacts
        write_jsonl(doc_dir / "pages.jsonl", pages)
        write_jsonl(doc_dir / "chunks_fixed.jsonl", chunks_by_strategy["fixed"])
        write_jsonl(doc_dir / "chunks_section_aware.jsonl", chunks_by_strategy["section_aware"])

        with open(doc_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(d.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
        with open(doc_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    with open(out_root / "corpus_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "manifest": str(args.manifest),
                "total_documents": len(docs),
                "reports": reports,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n✓ Parsed artifacts written to: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


