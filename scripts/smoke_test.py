#!/usr/bin/env python3
"""
Quick smoke test for QA Assistant.

Validates that the system is properly set up and all critical components work.
Run this before starting development or after deployment.

Usage:
    python scripts/smoke_test.py
"""
import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env (so DEEPSEEK_API_KEY is visible without manual export)
try:
    from dotenv import load_dotenv

    load_dotenv(project_root / ".env", override=False)
except Exception:
    pass


def print_header(text):
    """Print section header."""
    print(f"\n{'='*80}")
    print(f"  {text}")
    print('='*80)


def print_check(name, status, message=""):
    """Print check result."""
    symbol = "✅" if status else "❌"
    print(f"{symbol} {name:<40} {message}")


def check_imports():
    """Check that all critical imports work."""
    print_header("Import Checks")
    
    all_ok = True
    
    # Core dependencies
    checks = [
        ("yaml", "pyyaml"),
        ("pydantic", "pydantic"),
        ("fitz", "PyMuPDF"),
        ("requests", "requests"),
        ("arxiv", "arxiv"),
        ("tqdm", "tqdm"),
        ("openai", "openai"),
        ("tenacity", "tenacity"),
        ("sentence_transformers", "sentence-transformers"),
    ]
    
    for module, package in checks:
        try:
            __import__(module)
            print_check(f"Import {module}", True, f"({package})")
        except ImportError:
            print_check(f"Import {module}", False, f"Install: pip install {package}")
            all_ok = False
    
    return all_ok


def check_project_structure():
    """Check project structure."""
    print_header("Project Structure")
    
    all_ok = True
    
    required_dirs = [
        "configs",
        "ingest",
        "knowledge",
        "rag",
        "benchmarks",
        "tests",
        "scripts",
        "data",
    ]
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        exists = dir_path.exists()
        print_check(f"Directory: {dir_name}/", exists)
        if not exists:
            all_ok = False
    
    return all_ok


def check_data_directories():
    """Check data directories."""
    print_header("Data Directories")
    
    data_dirs = [
        "data/pdf",
        "data/parsed",
        "data/indices",
        "data/cache",
        "data/runs",
    ]
    
    for dir_name in data_dirs:
        dir_path = project_root / dir_name
        exists = dir_path.exists()
        print_check(f"Directory: {dir_name}/", exists)
        if not exists:
            # Try to create
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"  → Created {dir_name}/")
            except Exception as e:
                print(f"  → Failed to create: {e}")
    
    return True


def check_models():
    """Check data models."""
    print_header("Data Models")
    
    try:
        from knowledge.models import (
            Document, PageSpan, Chunk, Citation, Answer
        )
        
        # Create test instances
        span = PageSpan(start_page=1, end_page=1)
        print_check("PageSpan", True)
        
        doc = Document(
            id="test",
            title="Test",
            source_url="http://example.com",
            source_type="test",
            license="CC0"
        )
        print_check("Document", True)
        
        chunk = Chunk(
            id="chunk_1",
            doc_id="test",
            strategy="fixed",
            text="Test text",
            page_span=span,
            token_count=2,
            char_count=9
        )
        print_check("Chunk", True)
        
        citation = Citation(
            doc_id="test",
            page_span=span,
            quote="Test",
            score=0.9,
            retriever_tag="test"
        )
        print_check("Citation", True)
        
        answer = Answer(
            text="Test answer",
            citations=[citation],
            confidence=0.8
        )
        print_check("Answer", True)
        
        return True
    
    except Exception as e:
        print_check("Models", False, str(e))
        return False


def check_contracts():
    """Check RAG contracts."""
    print_header("RAG Contracts")
    
    try:
        from rag.contracts import (
            Retriever, Reranker, Generator, Pipeline
        )
        
        print_check("Retriever protocol", True)
        print_check("Reranker protocol", True)
        print_check("Generator protocol", True)
        print_check("Pipeline protocol", True)
        
        return True
    
    except Exception as e:
        print_check("Contracts", False, str(e))
        return False


def check_ingest():
    """Check ingest components."""
    print_header("Ingest Components")
    
    all_ok = True
    
    try:
        from ingest.normalization.text_normalizer import normalize_page_text
        
        test_text = "test   text\nwith\t\tspaces"
        normalized = normalize_page_text(test_text)
        
        print_check("Text normalization", True)
    except Exception as e:
        print_check("Text normalization", False, str(e))
        all_ok = False
    
    try:
        from ingest.chunking.fixed import chunk_pages_fixed, FixedChunkerConfig
        from knowledge.models import ParsedPage
        
        pages = [ParsedPage(doc_id="test", page_number=1, text="test " * 100, char_count=500)]
        cfg = FixedChunkerConfig(max_chars=100, overlap_chars=20, min_chars=50)
        chunks = list(chunk_pages_fixed(pages, doc_id="test", cfg=cfg))
        
        print_check("Fixed chunking", len(chunks) > 0, f"({len(chunks)} chunks)")
    except Exception as e:
        print_check("Fixed chunking", False, str(e))
        all_ok = False
    
    try:
        from ingest.chunking.section_aware import chunk_pages_section_aware, SectionAwareChunkerConfig
        
        pages = [ParsedPage(doc_id="test", page_number=1, text="1 Introduction\ntest " * 50, char_count=500)]
        cfg = SectionAwareChunkerConfig(max_chars=200, overlap_chars=0, min_chars=50)
        chunks = list(chunk_pages_section_aware(pages, doc_id="test", cfg=cfg))
        
        print_check("Section-aware chunking", len(chunks) > 0, f"({len(chunks)} chunks)")
    except Exception as e:
        print_check("Section-aware chunking", False, str(e))
        all_ok = False
    
    return all_ok


def check_llm_client():
    """Check LLM client."""
    print_header("LLM Client")
    
    try:
        from llm.deepseek_client import DeepSeekClient, CacheManager
        import os
        
        print_check("DeepSeekClient import", True)
        print_check("CacheManager import", True)
        
        # Check API key
        has_key = bool(os.getenv("DEEPSEEK_API_KEY"))
        print_check("DEEPSEEK_API_KEY set", has_key, 
                   "Set for API tests" if not has_key else "")
        
        return True
    
    except Exception as e:
        print_check("LLM client", False, str(e))
        return False


def run_quick_test():
    """Run a quick end-to-end test."""
    print_header("Quick End-to-End Test")
    
    try:
        from knowledge.models import ParsedPage, RetrievedChunk, Chunk, PageSpan, Citation, Answer
        from ingest.chunking.fixed import chunk_pages_fixed, FixedChunkerConfig
        
        # 1. Create mock pages
        pages = [
            ParsedPage(
                doc_id="test_doc",
                page_number=1,
                text="Delta measures option price sensitivity. " * 10,
                char_count=400
            )
        ]
        print_check("Create mock pages", True)
        
        # 2. Chunk
        cfg = FixedChunkerConfig(max_chars=200, overlap_chars=30, min_chars=50)
        chunks = list(chunk_pages_fixed(pages, doc_id="test_doc", cfg=cfg))
        print_check("Chunk pages", len(chunks) > 0, f"({len(chunks)} chunks)")
        
        # 3. Create retrieved chunk (simulating retrieval)
        if chunks:
            retrieved = RetrievedChunk(
                chunk=chunks[0],
                score=0.95,
                retriever_tag="test"
            )
            print_check("Create retrieved chunk", True)
        else:
            print_check("Create retrieved chunk", False, "No chunks created")
            return False
        
        # 4. Create citation
        citation = Citation(
            doc_id=retrieved.chunk.doc_id,
            page_span=retrieved.chunk.page_span,
            quote=retrieved.chunk.text[:50],
            score=retrieved.score,
            retriever_tag=retrieved.retriever_tag
        )
        print_check("Create citation", True)
        
        # 5. Create answer
        answer = Answer(
            text="Delta measures the sensitivity of option price to underlying price changes.",
            citations=[citation],
            confidence=0.9
        )
        print_check("Create answer", True)
        
        # 6. Format answer
        formatted = answer.format_with_citations()
        print_check("Format answer", "References:" in formatted)
        
        return True
    
    except Exception as e:
        print_check("End-to-end test", False, str(e))
        return False


def run_performance_check():
    """Run quick performance check."""
    print_header("Performance Check")
    
    try:
        from knowledge.models import ParsedPage
        from ingest.chunking.fixed import chunk_pages_fixed, FixedChunkerConfig
        
        # Create 100 pages
        pages = [
            ParsedPage(doc_id="perf", page_number=i, text="test " * 100, char_count=500)
            for i in range(1, 101)
        ]
        
        cfg = FixedChunkerConfig(max_chars=500, overlap_chars=50, min_chars=100)
        
        start = time.time()
        chunks = list(chunk_pages_fixed(pages, doc_id="perf", cfg=cfg))
        elapsed = time.time() - start
        
        target = 1.0  # 1 second
        passed = elapsed < target
        
        print_check(
            "Chunk 100 pages", 
            passed, 
            f"({elapsed:.3f}s, target: <{target}s)"
        )
        
        return passed
    
    except Exception as e:
        print_check("Performance check", False, str(e))
        return False


def main():
    """Run all smoke tests."""
    print("\n" + "="*80)
    print("  QA ASSISTANT - SMOKE TEST")
    print("="*80)
    
    start_time = time.time()
    
    results = {
        "imports": check_imports(),
        "structure": check_project_structure(),
        "data_dirs": check_data_directories(),
        "models": check_models(),
        "contracts": check_contracts(),
        "ingest": check_ingest(),
        "llm": check_llm_client(),
        "e2e": run_quick_test(),
        "performance": run_performance_check(),
    }
    
    elapsed = time.time() - start_time
    
    # Summary
    print_header("Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    print(f"Time: {elapsed:.2f}s\n")
    
    if all(results.values()):
        print("✅ ALL SMOKE TESTS PASSED!")
        print("\nSystem is ready. You can now:")
        print("  - Run full tests: python scripts/run_tests.py")
        print("  - Build indices: python scripts/build_indices.py")
        print("  - Run benchmark: python scripts/run_benchmark.py")
        print("  - Start bot: python scripts/run_telegram_bot.py")
        exit_code = 0
    else:
        print("❌ SOME TESTS FAILED")
        print("\nFailed checks:")
        for name, status in results.items():
            if not status:
                print(f"  - {name}")
        print("\nPlease fix the issues above before proceeding.")
        exit_code = 1
    
    print("="*80 + "\n")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

