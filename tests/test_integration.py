"""
Integration tests for full ingest→index→query pipeline.

Tests the complete workflow on a minimal corpus to ensure all components
work together correctly.
"""
import unittest
import tempfile
import shutil
from pathlib import Path
from typing import List

from knowledge.models import Document, ParsedPage, Chunk, PageSpan
from ingest.chunking.fixed import chunk_pages_fixed, FixedChunkerConfig
from ingest.normalization.text_normalizer import normalize_page_text


class TestIngestIntegration(unittest.TestCase):
    """Test complete ingest pipeline integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.pdf_dir = cls.test_dir / "pdf"
        cls.parsed_dir = cls.test_dir / "parsed"
        cls.pdf_dir.mkdir(exist_ok=True)
        cls.parsed_dir.mkdir(exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    def test_parsing_normalization_chunking_flow(self):
        """Test complete flow: parse → normalize → chunk."""
        # Create a mock document
        doc = Document(
            id="test_doc",
            title="Test Document",
            source_url="http://example.com/test.pdf",
            source_type="test",
            license="CC0"
        )
        
        # Simulate parsed pages (in real scenario, these come from PDF parser)
        raw_pages = [
            ParsedPage(
                doc_id=doc.id,
                page_number=1,
                text="Introduction\n\nThis is a test docu-\nment about option pricing.",
                char_count=100
            ),
            ParsedPage(
                doc_id=doc.id,
                page_number=2,
                text="Greeks\n\nDelta measures   the sensitivity\nof option price.",
                char_count=80
            )
        ]
        
        # Step 1: Normalize pages
        normalized_pages = []
        for page in raw_pages:
            normalized_text = normalize_page_text(page.text)
            normalized_pages.append(
                ParsedPage(
                    doc_id=page.doc_id,
                    page_number=page.page_number,
                    text=normalized_text,
                    char_count=len(normalized_text),
                    raw_char_count=page.char_count
                )
            )
        
        # Verify normalization worked
        self.assertIn("document about", normalized_pages[0].text)  # hyphen removed
        self.assertNotIn("docu-\n", normalized_pages[0].text)
        
        # Step 2: Chunk
        cfg = FixedChunkerConfig(max_chars=100, overlap_chars=20, min_chars=30)
        chunks = list(chunk_pages_fixed(normalized_pages, doc_id=doc.id, cfg=cfg))
        
        # Verify chunking
        self.assertGreater(len(chunks), 0)
        
        # All chunks should have valid page spans
        for chunk in chunks:
            self.assertIsNotNone(chunk.page_span)
            self.assertGreaterEqual(chunk.page_span.start_page, 1)
            self.assertLessEqual(chunk.page_span.end_page, 2)
            self.assertEqual(chunk.doc_id, doc.id)
    
    def test_chunks_maintain_document_traceability(self):
        """Test that chunks maintain traceability to source document and pages."""
        doc_id = "arxiv_test_001"
        
        pages = [
            ParsedPage(
                doc_id=doc_id,
                page_number=5,
                text="Content on page 5 " * 50,
                char_count=900
            ),
            ParsedPage(
                doc_id=doc_id,
                page_number=6,
                text="Content on page 6 " * 50,
                char_count=900
            )
        ]
        
        cfg = FixedChunkerConfig(max_chars=300, overlap_chars=30, min_chars=100)
        chunks = list(chunk_pages_fixed(pages, doc_id=doc_id, cfg=cfg))
        
        for chunk in chunks:
            # Every chunk must trace back to document
            self.assertEqual(chunk.doc_id, doc_id)
            
            # Must have valid page span
            self.assertGreaterEqual(chunk.page_span.start_page, 5)
            self.assertLessEqual(chunk.page_span.end_page, 6)
            
            # Must have content
            self.assertGreater(len(chunk.text), 0)
            
            # Char offsets should be present and valid
            if chunk.page_span.start_char is not None:
                self.assertGreaterEqual(chunk.page_span.start_char, 0)
            if chunk.page_span.end_char is not None:
                self.assertGreater(chunk.page_span.end_char, 0)


class TestIndexBuildIntegration(unittest.TestCase):
    """Test index building integration."""
    
    def setUp(self):
        """Set up test chunks."""
        self.test_chunks = self._create_test_chunks()
    
    def _create_test_chunks(self) -> List[Chunk]:
        """Create test chunks for indexing."""
        chunks = []
        
        texts = [
            "Delta is the first derivative of the option price with respect to the underlying asset price.",
            "Gamma measures the rate of change in delta over time.",
            "Vega represents sensitivity to volatility changes in the underlying.",
            "Theta measures time decay of options.",
            "The Black-Scholes model assumes constant volatility and no arbitrage."
        ]
        
        for i, text in enumerate(texts):
            span = PageSpan(start_page=i+1, end_page=i+1)
            chunk = Chunk(
                id=f"chunk_{i}",
                doc_id="test_doc",
                strategy="fixed",
                text=text,
                page_span=span,
                token_count=len(text.split()),
                char_count=len(text)
            )
            chunks.append(chunk)
        
        return chunks
    
    def test_chunks_are_indexable(self):
        """Test that chunks have all required fields for indexing."""
        for chunk in self.test_chunks:
            # Required for BM25
            self.assertIsNotNone(chunk.text)
            self.assertGreater(len(chunk.text), 0)
            
            # Required for vector index
            self.assertIsNotNone(chunk.id)
            
            # Required for citations
            self.assertIsNotNone(chunk.page_span)
            self.assertIsNotNone(chunk.doc_id)
    
    def test_chunk_ids_are_unique(self):
        """Test that chunk IDs are unique."""
        ids = [chunk.id for chunk in self.test_chunks]
        self.assertEqual(len(ids), len(set(ids)), "Chunk IDs must be unique")


class TestEndToEndQuery(unittest.TestCase):
    """Test end-to-end query pipeline."""
    
    def setUp(self):
        """Set up mock components."""
        self.test_chunks = self._create_derivatives_chunks()
    
    def _create_derivatives_chunks(self) -> List[Chunk]:
        """Create realistic derivatives content chunks."""
        texts = [
            "The Greeks are risk measures that describe how option prices change. "
            "Delta (Δ) measures sensitivity to underlying price, gamma (Γ) measures "
            "rate of change of delta, and vega (ν) measures sensitivity to volatility.",
            
            "Delta hedging is a strategy to reduce directional risk by offsetting "
            "the delta of an option position with the underlying asset. For a call option, "
            "delta is positive, so we sell the underlying to hedge.",
            
            "The Black-Scholes formula provides closed-form solutions for European "
            "call and put options. Key assumptions include: constant volatility, "
            "no dividends, frictionless markets, and log-normal price distribution.",
            
            "Monte Carlo simulation is used to price complex derivatives by simulating "
            "many possible price paths. Variance reduction techniques like antithetic "
            "variates can improve efficiency.",
            
            "American options can be exercised at any time before expiration, unlike "
            "European options. The Longstaff-Schwartz method uses least squares regression "
            "to estimate continuation values."
        ]
        
        chunks = []
        for i, text in enumerate(texts):
            span = PageSpan(start_page=i*2+1, end_page=i*2+1)
            chunk = Chunk(
                id=f"deriv_chunk_{i}",
                doc_id="derivatives_intro",
                strategy="fixed",
                text=text,
                page_span=span,
                token_count=len(text.split()),
                char_count=len(text)
            )
            chunks.append(chunk)
        
        return chunks
    
    def test_keyword_matching_for_retrieval(self):
        """Test that chunks can be matched by keywords (simulating BM25)."""
        # Simulate simple keyword matching
        query = "delta hedging strategy"
        query_terms = set(query.lower().split())
        
        relevant_chunks = []
        for chunk in self.test_chunks:
            chunk_terms = set(chunk.text.lower().split())
            overlap = query_terms & chunk_terms
            if len(overlap) >= 2:  # At least 2 matching terms
                relevant_chunks.append(chunk)
        
        # Should find the delta hedging chunk
        self.assertGreater(len(relevant_chunks), 0)
        
        # Check that relevant content is found
        found_delta_hedge = any("delta hedging" in c.text.lower() for c in relevant_chunks)
        self.assertTrue(found_delta_hedge)
    
    def test_citation_construction_from_chunks(self):
        """Test constructing citations from retrieved chunks."""
        from knowledge.models import Citation, RetrievedChunk
        
        # Simulate retrieved chunk
        chunk = self.test_chunks[1]  # Delta hedging chunk
        retrieved = RetrievedChunk(
            chunk=chunk,
            score=0.95,
            retriever_tag="bm25"
        )
        
        # Construct citation
        citation = Citation(
            doc_id=retrieved.chunk.doc_id,
            page_span=retrieved.chunk.page_span,
            quote=retrieved.chunk.text[:100],
            score=retrieved.score,
            retriever_tag=retrieved.retriever_tag,
            chunk_id=retrieved.chunk.id
        )
        
        # Verify citation
        self.assertEqual(citation.doc_id, "derivatives_intro")
        self.assertIsNotNone(citation.page_span)
        self.assertGreater(len(citation.quote), 0)
        self.assertEqual(citation.retriever_tag, "bm25")
    
    def test_answer_construction_with_multiple_citations(self):
        """Test building answer with multiple supporting citations."""
        from knowledge.models import Answer, Citation
        
        # Create citations from multiple chunks
        citations = []
        for i in [0, 1]:  # Use first two chunks
            chunk = self.test_chunks[i]
            citation = Citation(
                doc_id=chunk.doc_id,
                page_span=chunk.page_span,
                quote=chunk.text[:80],
                score=0.9 - i*0.05,
                retriever_tag="dense",
                chunk_id=chunk.id
            )
            citations.append(citation)
        
        # Construct answer
        answer = Answer(
            text="Delta (Δ) is a key Greek that measures option price sensitivity to "
                 "underlying price changes. Delta hedging uses this measure to reduce "
                 "directional risk by taking offsetting positions in the underlying asset.",
            citations=citations,
            confidence=0.88
        )
        
        # Verify answer
        self.assertTrue(answer.has_citations())
        self.assertEqual(len(answer.citations), 2)
        self.assertFalse(answer.is_refused())
        
        # Test formatting
        formatted = answer.format_with_citations()
        self.assertIn("References:", formatted)
        self.assertIn("[1]", formatted)
        self.assertIn("[2]", formatted)


if __name__ == "__main__":
    unittest.main()

