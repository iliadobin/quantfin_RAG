#!/usr/bin/env python3
"""
Main script to collect public corpus:
1. Build arXiv allowlist via API
2. Download anchors + arXiv PDFs
3. Generate corpus manifest with metadata/checksums
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ingest.sources.arxiv_collector import ArxivCollector
from ingest.sources.pdf_downloader import PDFDownloader


def main():
    """Run full corpus collection pipeline."""
    print("\n" + "="*80)
    print("QA Assistant - Public Corpus Collection")
    print("Target: pricing + немного risk, ≤900 pages")
    print("="*80 + "\n")
    
    # Step 1: Build arXiv allowlist
    print("STEP 1: Building arXiv allowlist via API...")
    print("-" * 80)
    
    collector = ArxivCollector("configs/arxiv_search_config.yaml")
    papers = collector.collect_allowlist()
    collector.save_allowlist(papers, "configs/arxiv_allowlist.yaml")
    if not papers:
        raise RuntimeError(
            "arXiv allowlist is empty. Loosen keywords in configs/arxiv_search_config.yaml "
            "or remove prefer_keywords constraint (now optional scoring) and retry."
        )
    
    # Step 2: Download PDFs
    print("\nSTEP 2: Downloading PDFs...")
    print("-" * 80)
    
    downloader = PDFDownloader()
    
    # Download anchors (may require manual intervention if URLs fail)
    anchor_docs = downloader.download_anchors("configs/anchors.yaml")
    
    # Download arXiv papers
    arxiv_docs = downloader.download_arxiv_papers("configs/arxiv_allowlist.yaml")
    
    # Step 3: Build manifest with page limit
    print("\nSTEP 3: Building corpus manifest...")
    print("-" * 80)
    
    all_docs = anchor_docs + arxiv_docs
    manifest = downloader.build_corpus_manifest(
        all_docs,
        max_pages=900,
        output_path="configs/corpus_public.yaml"
    )
    
    # Summary
    print("\n" + "="*80)
    print("✓ CORPUS COLLECTION COMPLETE")
    print("="*80)
    print(f"Documents collected: {manifest.total_documents}")
    print(f"Total pages: {manifest.total_pages} / 900")
    print(f"PDF directory: {downloader.pdf_dir}")
    print(f"Manifest: configs/corpus_public.yaml")
    print("\nNext steps:")
    print("  1. Review configs/corpus_public.yaml")
    print("  2. Run parsing: python scripts/parse_corpus.py")
    print("  3. Build indices: python scripts/build_indices.py")


if __name__ == "__main__":
    main()

