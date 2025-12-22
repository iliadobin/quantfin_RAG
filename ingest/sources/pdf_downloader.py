"""
PDF Downloader with metadata extraction and page counting.
Downloads anchors + arXiv papers, computes checksums, counts pages.
"""
import hashlib
import requests
import yaml
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import time
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from knowledge.models import Document, CorpusManifest


class PDFDownloader:
    """Download and process PDF documents."""
    
    def __init__(self, pdf_dir: str = "data/pdf"):
        """Initialize downloader."""
        # Project root (repo root) = two levels up from this file: ingest/sources/ -> repo/
        self.project_root = Path(__file__).resolve().parents[2]
        pdf_dir_path = Path(pdf_dir)
        self.pdf_dir = (self.project_root / pdf_dir_path).resolve() if not pdf_dir_path.is_absolute() else pdf_dir_path
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, output_path: Path, timeout: int = 30) -> bool:
        """Download file from URL."""
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                if total_size:
                    pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name)
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
                    pbar.close()
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            return True
        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            return False
    
    def compute_sha256(self, file_path: Path) -> str:
        """Compute SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def count_pdf_pages(self, file_path: Path) -> Optional[int]:
        """Count pages in PDF using PyMuPDF."""
        try:
            doc = fitz.open(file_path)
            page_count = doc.page_count
            doc.close()
            return page_count
        except Exception as e:
            print(f"  Warning: Could not count pages: {e}")
            return None
    
    def download_anchors(self, anchors_config_path: str) -> List[Document]:
        """Download anchor documents."""
        print("\n" + "=" * 80)
        print("Downloading Anchor Documents")
        print("=" * 80)
        
        with open(anchors_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        documents = []
        for anchor in config['anchors']:
            doc_id = anchor['id']
            print(f"\n[{doc_id}] {anchor['title']}")
            print(f"  URL: {anchor['url']}")
            
            # Create filename
            pdf_filename = f"{doc_id}.pdf"
            pdf_path = self.pdf_dir / pdf_filename
            
            # Download if not exists
            if pdf_path.exists():
                print(f"  ✓ Already exists: {pdf_path}")
            else:
                print(f"  Downloading...")
                success = self.download_file(anchor['url'], pdf_path)
                if not success:
                    print(f"  ✗ Skipping {doc_id} (download failed)")
                    print(f"  Note: You may need to manually download from {anchor['url']}")
                    continue
                time.sleep(1)  # be polite
            
            # Compute metadata
            if pdf_path.exists():
                checksum = self.compute_sha256(pdf_path)
                page_count = self.count_pdf_pages(pdf_path)
                file_size = pdf_path.stat().st_size
                
                print(f"  ✓ SHA256: {checksum[:16]}...")
                print(f"  ✓ Pages: {page_count}")
                print(f"  ✓ Size: {file_size / 1024:.1f} KB")
                
                doc = Document(
                    id=doc_id,
                    title=anchor['title'],
                    source_url=anchor['url'],
                    source_type=anchor['source_type'],
                    license=anchor['license'],
                    corpus_profile="public",
                    pdf_path=str(pdf_path.relative_to(self.project_root)),
                    checksum_sha256=checksum,
                    file_size_bytes=file_size,
                    page_count=page_count,
                    description=anchor.get('description'),
                    retrieved_at=datetime.now()
                )
                documents.append(doc)
        
        return documents
    
    def download_arxiv_papers(self, allowlist_path: str) -> List[Document]:
        """Download arXiv papers from allowlist."""
        print("\n" + "=" * 80)
        print("Downloading arXiv Papers")
        print("=" * 80)
        
        with open(allowlist_path, 'r') as f:
            allowlist = yaml.safe_load(f)

        if not allowlist.get("papers"):
            raise RuntimeError(f"Allowlist at {allowlist_path} has no papers. Run arXiv collector first.")
        
        documents = []
        for paper in allowlist['papers']:
            arxiv_id = paper['arxiv_id']
            print(f"\n[{arxiv_id}] {paper['title'][:60]}...")
            
            # Create filename
            pdf_filename = f"arxiv_{arxiv_id.replace('.', '_')}.pdf"
            pdf_path = self.pdf_dir / pdf_filename
            
            # Download if not exists
            if pdf_path.exists():
                print(f"  ✓ Already exists")
            else:
                print(f"  Downloading from {paper['pdf_url']}")
                success = self.download_file(paper['pdf_url'], pdf_path)
                if not success:
                    print(f"  ✗ Skipping {arxiv_id} (download failed)")
                    continue
                time.sleep(2)  # be polite to arXiv
            
            # Compute metadata
            if pdf_path.exists():
                checksum = self.compute_sha256(pdf_path)
                page_count = self.count_pdf_pages(pdf_path)
                file_size = pdf_path.stat().st_size
                
                print(f"  ✓ SHA256: {checksum[:16]}...")
                print(f"  ✓ Pages: {page_count}")
                
                # Parse year from published date
                year = None
                if 'published' in paper:
                    year = int(paper['published'][:4])
                
                doc = Document(
                    id=f"arxiv_{arxiv_id.replace('.', '_')}",
                    title=paper['title'],
                    source_url=f"https://arxiv.org/abs/{arxiv_id}",
                    source_type="arxiv",
                    license="arXiv.org perpetual non-exclusive license",
                    corpus_profile="public",
                    pdf_path=str(pdf_path.relative_to(self.project_root)),
                    checksum_sha256=checksum,
                    file_size_bytes=file_size,
                    page_count=page_count,
                    authors=paper.get('authors'),
                    year=year,
                    arxiv_id=arxiv_id,
                    primary_category=paper.get('primary_category'),
                    retrieved_at=datetime.now()
                )
                documents.append(doc)
        
        return documents
    
    def build_corpus_manifest(
        self, 
        documents: List[Document],
        max_pages: int = 900,
        output_path: str = "configs/corpus_public.yaml"
    ) -> CorpusManifest:
        """Build corpus manifest with page limit control."""
        print("\n" + "=" * 80)
        print("Building Corpus Manifest")
        print("=" * 80)
        
        # Keep anchors first (if present), then fill with smaller docs to fit the page budget
        anchors = [d for d in documents if d.source_type in ("regulatory", "industry")]
        others = [d for d in documents if d.source_type not in ("regulatory", "industry")]
        others_sorted = sorted(others, key=lambda d: d.page_count or 0)
        documents_sorted = anchors + others_sorted
        
        # Select documents up to page limit
        selected_docs = []
        total_pages = 0
        
        for doc in documents_sorted:
            if doc.page_count is None:
                print(f"Warning: {doc.id} has no page count, skipping")
                continue
            
            if total_pages + doc.page_count <= max_pages:
                selected_docs.append(doc)
                total_pages += doc.page_count
            else:
                print(f"Skipping {doc.id} ({doc.page_count}p) - would exceed {max_pages} page limit")
        
        print(f"\n✓ Selected {len(selected_docs)} documents")
        print(f"✓ Total pages: {total_pages} / {max_pages}")
        
        manifest = CorpusManifest(
            created_at=datetime.now(),
            total_documents=len(selected_docs),
            total_pages=total_pages,
            documents=selected_docs,
            build_config={
                'max_pages': max_pages,
                'embedding_model': 'intfloat/e5-small-v2',
                'target_domain': 'derivatives pricing + market risk'
            },
            notes=f"Public corpus: {len(selected_docs)} docs, {total_pages} pages"
        )
        
        # Save manifest
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            # Convert to dict for YAML serialization
            manifest_dict = manifest.model_dump(mode='json')
            yaml.dump(manifest_dict, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\n✓ Manifest saved to: {output_file}")
        
        return manifest


if __name__ == "__main__":
    downloader = PDFDownloader()
    
    # Download anchors
    anchor_docs = downloader.download_anchors("configs/anchors.yaml")
    
    # Download arXiv papers
    arxiv_docs = downloader.download_arxiv_papers("configs/arxiv_allowlist.yaml")
    
    # Build corpus manifest with page limit
    all_docs = anchor_docs + arxiv_docs
    manifest = downloader.build_corpus_manifest(
        all_docs,
        max_pages=900,
        output_path="configs/corpus_public.yaml"
    )
    
    print("\n" + "=" * 80)
    print("Corpus Collection Complete!")
    print("=" * 80)
    print(f"Total documents: {manifest.total_documents}")
    print(f"Total pages: {manifest.total_pages}")

