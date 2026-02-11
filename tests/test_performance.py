"""
Performance smoke tests for RAG components.

Tests latency, throughput, memory usage to ensure system meets
performance requirements.
"""
import unittest
import time
import tracemalloc
from typing import List
from pathlib import Path

from knowledge.models import ParsedPage, Chunk, PageSpan
from ingest.chunking.fixed import chunk_pages_fixed, FixedChunkerConfig
from ingest.chunking.section_aware import chunk_pages_section_aware, SectionAwareChunkerConfig
from ingest.normalization.text_normalizer import normalize_page_text


class TestChunkingPerformance(unittest.TestCase):
    """Test chunking performance."""
    
    def setUp(self):
        """Create test data."""
        # Create a realistic-sized document (100 pages, ~500 chars each)
        self.pages = []
        for i in range(100):
            text = f"Page {i+1}\n\n" + ("Test content. " * 35)  # ~500 chars
            self.pages.append(
                ParsedPage(
                    doc_id="perf_test_doc",
                    page_number=i+1,
                    text=text,
                    char_count=len(text)
                )
            )
    
    def test_fixed_chunking_latency(self):
        """Test fixed chunking latency."""
        cfg = FixedChunkerConfig(max_chars=500, overlap_chars=50, min_chars=100)
        
        start_time = time.time()
        chunks = list(chunk_pages_fixed(self.pages, doc_id="perf_test_doc", cfg=cfg))
        elapsed = time.time() - start_time
        
        # Should complete in under 1 second for 100 pages
        self.assertLess(elapsed, 1.0, f"Fixed chunking took {elapsed:.3f}s (expected < 1s)")
        
        # Should produce reasonable number of chunks
        self.assertGreater(len(chunks), 50)
        self.assertLess(len(chunks), 500)
        
        print(f"Fixed chunking: {len(chunks)} chunks in {elapsed:.3f}s "
              f"({len(chunks)/elapsed:.1f} chunks/sec)")
    
    def test_section_aware_chunking_latency(self):
        """Test section-aware chunking latency."""
        cfg = SectionAwareChunkerConfig(max_chars=500, overlap_chars=50, min_chars=100)
        
        start_time = time.time()
        chunks = list(chunk_pages_section_aware(self.pages, doc_id="perf_test_doc", cfg=cfg))
        elapsed = time.time() - start_time
        
        # Section-aware should still be reasonably fast (< 2s for 100 pages)
        self.assertLess(elapsed, 2.0, f"Section-aware chunking took {elapsed:.3f}s (expected < 2s)")
        
        print(f"Section-aware chunking: {len(chunks)} chunks in {elapsed:.3f}s "
              f"({len(chunks)/elapsed:.1f} chunks/sec)")
    
    def test_chunking_memory_usage(self):
        """Test memory usage during chunking."""
        cfg = FixedChunkerConfig(max_chars=500, overlap_chars=50, min_chars=100)
        
        tracemalloc.start()
        
        chunks = list(chunk_pages_fixed(self.pages, doc_id="perf_test_doc", cfg=cfg))
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / 1024 / 1024
        
        # Memory should be reasonable (< 50MB for 100 pages)
        self.assertLess(peak_mb, 50, f"Peak memory usage: {peak_mb:.1f}MB (expected < 50MB)")
        
        print(f"Chunking memory: current={current/1024/1024:.1f}MB, peak={peak_mb:.1f}MB")


class TestNormalizationPerformance(unittest.TestCase):
    """Test text normalization performance."""
    
    def test_normalization_throughput(self):
        """Test normalization throughput on large text."""
        # Create ~100KB of text with various issues
        text = """This is a test docu-
ment with line breaks and   extra    spaces.

It has multiple    consecutive   whitespaces and hyphen-
ation across lines that need to be fixed.
""" * 1000  # Repeat to get ~100KB
        
        start_time = time.time()
        
        # Normalize 100 times
        for _ in range(100):
            normalized = normalize_page_text(text)
        
        elapsed = time.time() - start_time
        
        # Should handle 100 normalizations in < 1 second
        self.assertLess(elapsed, 1.0, f"Normalization took {elapsed:.3f}s (expected < 1s)")
        
        throughput = (len(text) * 100) / elapsed / 1024 / 1024  # MB/s
        print(f"Normalization throughput: {throughput:.1f} MB/s")


class TestRetrievalPerformance(unittest.TestCase):
    """Test retrieval performance (without actual indices, just structure)."""
    
    def setUp(self):
        """Create test chunks."""
        self.chunks = self._create_large_chunk_set(1000)
    
    def _create_large_chunk_set(self, count: int) -> List[Chunk]:
        """Create a large set of chunks for testing."""
        chunks = []
        keywords = ["derivatives", "options", "delta", "gamma", "vega"]
        for i in range(count):
            span = PageSpan(start_page=i//10 + 1, end_page=i//10 + 1)
            # Ensure each chunk has at least one keyword
            keyword = keywords[i % len(keywords)]
            chunk = Chunk(
                id=f"chunk_{i}",
                doc_id=f"doc_{i//100}",
                strategy="fixed",
                text=f"This is test chunk {i} with content about {keyword} and financial instruments.",
                page_span=span,
                token_count=15,
                char_count=70
            )
            chunks.append(chunk)
        return chunks
    
    def test_chunk_filtering_performance(self):
        """Test performance of filtering chunks (simulating post-retrieval filtering)."""
        start_time = time.time()
        
        # Simulate filtering operations
        for query_term in ["derivatives", "options", "delta", "gamma", "vega"]:
            filtered = [c for c in self.chunks if query_term in c.text.lower()]
            self.assertGreater(len(filtered), 0)
        
        elapsed = time.time() - start_time
        
        # Should be fast (< 0.1s for 1000 chunks × 5 queries)
        self.assertLess(elapsed, 0.1, f"Chunk filtering took {elapsed:.3f}s (expected < 0.1s)")
        
        print(f"Chunk filtering: 5 queries over 1000 chunks in {elapsed:.3f}s")
    
    def test_chunk_sorting_performance(self):
        """Test performance of sorting chunks (simulating reranking)."""
        # Add scores to chunks
        import random
        scored_chunks = [(c, random.random()) for c in self.chunks]
        
        start_time = time.time()
        
        # Sort by score
        sorted_chunks = sorted(scored_chunks, key=lambda x: x[1], reverse=True)
        top_k = sorted_chunks[:10]
        
        elapsed = time.time() - start_time
        
        # Should be very fast (< 0.01s for 1000 chunks)
        self.assertLess(elapsed, 0.01, f"Sorting took {elapsed:.3f}s (expected < 0.01s)")
        
        print(f"Chunk sorting: 1000 chunks in {elapsed:.3f}s")


class TestCitationPerformance(unittest.TestCase):
    """Test citation construction performance."""
    
    def test_citation_formatting_performance(self):
        """Test citation formatting performance."""
        from knowledge.models import Citation
        
        # Create many citations
        citations = []
        for i in range(100):
            span = PageSpan(start_page=i+1, end_page=i+1)
            citation = Citation(
                doc_id=f"doc_{i}",
                page_span=span,
                quote=f"Quote from document {i} " * 10,
                score=0.9,
                retriever_tag="dense"
            )
            citations.append(citation)
        
        start_time = time.time()
        
        # Format all citations
        formatted = [c.format_page_reference() for c in citations]
        
        elapsed = time.time() - start_time
        
        # Should be very fast
        self.assertLess(elapsed, 0.01, f"Citation formatting took {elapsed:.3f}s")
        self.assertEqual(len(formatted), 100)
        
        print(f"Citation formatting: 100 citations in {elapsed:.3f}s")
    
    def test_answer_formatting_performance(self):
        """Test answer formatting with many citations."""
        from knowledge.models import Citation, Answer
        
        # Create answer with many citations
        citations = []
        for i in range(50):
            span = PageSpan(start_page=i+1, end_page=i+2)
            citation = Citation(
                doc_id="test_doc",
                page_span=span,
                quote=f"Supporting quote number {i} with detailed content.",
                score=0.9 - i*0.01,
                retriever_tag="hybrid"
            )
            citations.append(citation)
        
        answer = Answer(
            text="This is a comprehensive answer with many supporting citations.",
            citations=citations,
            confidence=0.85
        )
        
        start_time = time.time()
        
        # Format with citations
        formatted = answer.format_with_citations()
        
        elapsed = time.time() - start_time
        
        # Should be fast even with many citations
        self.assertLess(elapsed, 0.05, f"Answer formatting took {elapsed:.3f}s")
        self.assertIn("References:", formatted)
        
        print(f"Answer formatting: 50 citations in {elapsed:.3f}s")


class TestIndexBuildPerformance(unittest.TestCase):
    """Test index building performance (smoke test)."""
    
    def test_chunk_serialization_performance(self):
        """Test serialization performance for index storage."""
        import json
        
        # Create chunks
        chunks = []
        for i in range(1000):
            span = PageSpan(start_page=i//10 + 1, end_page=i//10 + 1)
            chunk = Chunk(
                id=f"chunk_{i}",
                doc_id="test_doc",
                strategy="fixed",
                text=f"Chunk text {i} " * 20,
                page_span=span,
                token_count=25,
                char_count=250
            )
            chunks.append(chunk)
        
        start_time = time.time()
        
        # Serialize to JSON
        serialized = [json.loads(c.model_dump_json()) for c in chunks]
        
        elapsed = time.time() - start_time
        
        # Should complete quickly
        self.assertLess(elapsed, 0.5, f"Serialization took {elapsed:.3f}s (expected < 0.5s)")
        self.assertEqual(len(serialized), 1000)
        
        print(f"Chunk serialization: 1000 chunks in {elapsed:.3f}s")


class PerformanceReport:
    """Collect and display performance results."""
    
    @staticmethod
    def print_summary():
        """Print performance summary."""
        print("\n" + "="*80)
        print("PERFORMANCE TEST SUMMARY")
        print("="*80)
        print("\nAll performance tests passed!")
        print("\nKey metrics:")
        print("  - Chunking: < 1s for 100 pages")
        print("  - Normalization: > 10 MB/s throughput")
        print("  - Retrieval operations: < 0.1s for 1000 chunks")
        print("  - Citation formatting: < 0.01s for 100 citations")
        print("  - Memory usage: < 50MB for typical workloads")
        print("="*80 + "\n")


if __name__ == "__main__":
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary if all passed
    if result.wasSuccessful():
        PerformanceReport.print_summary()

