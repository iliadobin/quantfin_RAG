"""
Unit tests for knowledge models.

Tests data models, validation, serialization, and helper methods.
"""
import unittest
from datetime import datetime
from knowledge.models import (
    Document, PageSpan, ParsedPage, Chunk, Citation, 
    RetrievalTrace, Answer, RetrievedChunk
)


class TestPageSpan(unittest.TestCase):
    """Test PageSpan model."""
    
    def test_single_page_span(self):
        """Test single-page span."""
        span = PageSpan(start_page=5, end_page=5)
        self.assertEqual(span.start_page, 5)
        self.assertEqual(span.end_page, 5)
        self.assertIsNone(span.start_char)
    
    def test_multi_page_span(self):
        """Test multi-page span."""
        span = PageSpan(start_page=10, end_page=15)
        self.assertEqual(span.start_page, 10)
        self.assertEqual(span.end_page, 15)
    
    def test_page_span_with_chars(self):
        """Test page span with character offsets."""
        span = PageSpan(start_page=3, end_page=3, start_char=100, end_char=500)
        self.assertEqual(span.start_char, 100)
        self.assertEqual(span.end_char, 500)


class TestDocument(unittest.TestCase):
    """Test Document model."""
    
    def test_minimal_document(self):
        """Test creating document with minimal required fields."""
        doc = Document(
            id="test_doc_1",
            title="Test Document",
            source_url="https://example.com/test.pdf",
            source_type="arxiv",
            license="CC BY 4.0"
        )
        self.assertEqual(doc.id, "test_doc_1")
        self.assertEqual(doc.corpus_profile, "public")  # default
    
    def test_document_with_metadata(self):
        """Test document with full metadata."""
        doc = Document(
            id="arxiv_2005_02347",
            title="Differential Machine Learning",
            source_url="https://arxiv.org/abs/2005.02347",
            source_type="arxiv",
            license="arXiv.org perpetual non-exclusive license",
            arxiv_id="2005.02347",
            primary_category="q-fin.CP",
            authors=["Author One", "Author Two"],
            year=2020,
            page_count=42
        )
        self.assertEqual(doc.arxiv_id, "2005.02347")
        self.assertEqual(len(doc.authors), 2)
        self.assertEqual(doc.page_count, 42)


class TestChunk(unittest.TestCase):
    """Test Chunk model."""
    
    def test_chunk_creation(self):
        """Test creating chunk with required fields."""
        span = PageSpan(start_page=5, end_page=5, start_char=0, end_char=500)
        chunk = Chunk(
            id="chunk_1",
            doc_id="doc_1",
            strategy="fixed",
            text="This is test text for a chunk.",
            page_span=span,
            token_count=10,
            char_count=500
        )
        self.assertEqual(chunk.id, "chunk_1")
        self.assertEqual(chunk.strategy, "fixed")
        self.assertEqual(len(chunk.section_path), 0)  # default empty
    
    def test_chunk_with_section_path(self):
        """Test chunk with section hierarchy."""
        span = PageSpan(start_page=10, end_page=10)
        chunk = Chunk(
            id="chunk_2",
            doc_id="doc_1",
            strategy="section_aware",
            text="Section text",
            page_span=span,
            section_path=["1 Introduction", "1.1 Background"],
            token_count=5,
            char_count=100
        )
        self.assertEqual(len(chunk.section_path), 2)
        self.assertIn("1 Introduction", chunk.section_path)


class TestCitation(unittest.TestCase):
    """Test Citation model and formatting."""
    
    def test_citation_creation(self):
        """Test creating citation."""
        span = PageSpan(start_page=15, end_page=15)
        citation = Citation(
            doc_id="arxiv_2005_02347",
            page_span=span,
            quote="The Greeks measure sensitivity to various parameters.",
            score=0.95,
            retriever_tag="dense"
        )
        self.assertEqual(citation.doc_id, "arxiv_2005_02347")
        self.assertEqual(citation.score, 0.95)
    
    def test_single_page_reference_formatting(self):
        """Test formatting single-page citation reference."""
        span = PageSpan(start_page=10, end_page=10)
        citation = Citation(
            doc_id="test_doc",
            page_span=span,
            quote="Sample quote",
            score=0.8,
            retriever_tag="bm25"
        )
        formatted = citation.format_page_reference()
        self.assertEqual(formatted, "test_doc, p. 10")
    
    def test_multi_page_reference_formatting(self):
        """Test formatting multi-page citation reference."""
        span = PageSpan(start_page=10, end_page=15)
        citation = Citation(
            doc_id="test_doc",
            page_span=span,
            quote="Multi-page quote",
            score=0.9,
            retriever_tag="hybrid"
        )
        formatted = citation.format_page_reference()
        self.assertEqual(formatted, "test_doc, pp. 10-15")


class TestAnswer(unittest.TestCase):
    """Test Answer model and methods."""
    
    def test_answer_with_citations(self):
        """Test answer with supporting citations."""
        span = PageSpan(start_page=5, end_page=5)
        citation = Citation(
            doc_id="doc_1",
            page_span=span,
            quote="Supporting evidence",
            score=0.9,
            retriever_tag="dense"
        )
        answer = Answer(
            text="Delta measures the rate of change of option price with respect to underlying.",
            citations=[citation],
            confidence=0.95
        )
        
        self.assertTrue(answer.has_citations())
        self.assertFalse(answer.is_refused())
        self.assertEqual(len(answer.citations), 1)
    
    def test_refused_answer(self):
        """Test refused answer."""
        answer = Answer(
            text="I cannot answer this question.",
            citations=[],
            confidence=0.0,
            refusal_reason="Insufficient evidence in corpus"
        )
        
        self.assertTrue(answer.is_refused())
        self.assertFalse(answer.has_citations())
        self.assertIsNotNone(answer.refusal_reason)
    
    def test_answer_formatting_with_citations(self):
        """Test formatted answer with inline citations."""
        span1 = PageSpan(start_page=10, end_page=10)
        span2 = PageSpan(start_page=15, end_page=16)
        
        citations = [
            Citation(
                doc_id="doc_1",
                page_span=span1,
                quote="Delta is the first derivative of option price with respect to spot price.",
                score=0.95,
                retriever_tag="dense"
            ),
            Citation(
                doc_id="doc_1",
                page_span=span2,
                quote="Gamma measures the rate of change of delta.",
                score=0.90,
                retriever_tag="dense"
            )
        ]
        
        answer = Answer(
            text="Delta and gamma are first-order and second-order Greeks respectively.",
            citations=citations,
            confidence=0.92
        )
        
        formatted = answer.format_with_citations()
        self.assertIn("References:", formatted)
        self.assertIn("[1]", formatted)
        self.assertIn("[2]", formatted)
        self.assertIn("doc_1, p. 10", formatted)
        self.assertIn("doc_1, pp. 15-16", formatted)


class TestRetrievalTrace(unittest.TestCase):
    """Test RetrievalTrace model."""
    
    def test_basic_trace(self):
        """Test basic retrieval trace."""
        trace = RetrievalTrace(
            query="What is delta hedging?",
            retrieved_chunks_count=10,
            final_chunks_count=5,
            retrieval_time_ms=150.5
        )
        
        self.assertEqual(trace.query, "What is delta hedging?")
        self.assertEqual(trace.retrieved_chunks_count, 10)
        self.assertEqual(trace.final_chunks_count, 5)
    
    def test_trace_with_query_expansion(self):
        """Test trace with expanded queries."""
        trace = RetrievalTrace(
            query="Greeks",
            expanded_queries=[
                "What are option Greeks?",
                "Delta gamma vega theta rho",
                "Greek sensitivities for options"
            ],
            retrieved_chunks_count=30,
            reranked_chunks_count=10,
            final_chunks_count=5
        )
        
        self.assertEqual(len(trace.expanded_queries), 3)
        self.assertEqual(trace.reranked_chunks_count, 10)


class TestParsedPage(unittest.TestCase):
    """Test ParsedPage model."""
    
    def test_parsed_page_creation(self):
        """Test creating parsed page."""
        page = ParsedPage(
            doc_id="doc_1",
            page_number=1,
            text="This is page content.",
            char_count=21
        )
        
        self.assertEqual(page.page_number, 1)
        self.assertEqual(page.char_count, 21)
        self.assertIsNone(page.extraction_method)


class TestRetrievedChunk(unittest.TestCase):
    """Test RetrievedChunk model."""
    
    def test_retrieved_chunk_wraps_chunk(self):
        """Test that RetrievedChunk properly wraps Chunk."""
        span = PageSpan(start_page=1, end_page=1)
        chunk = Chunk(
            id="chunk_1",
            doc_id="doc_1",
            strategy="fixed",
            text="Test text",
            page_span=span,
            token_count=3,
            char_count=9
        )
        
        retrieved = RetrievedChunk(
            chunk=chunk,
            score=0.85,
            retriever_tag="bm25"
        )
        
        self.assertEqual(retrieved.chunk.id, "chunk_1")
        self.assertEqual(retrieved.score, 0.85)
        self.assertEqual(retrieved.retriever_tag, "bm25")


if __name__ == "__main__":
    unittest.main()

