import unittest

from ingest.normalization.text_normalizer import normalize_page_text
from ingest.chunking.fixed import chunk_pages_fixed, FixedChunkerConfig
from ingest.chunking.section_aware import chunk_pages_section_aware, SectionAwareChunkerConfig
from knowledge.models import ParsedPage


class TestTextNormalization(unittest.TestCase):
    def test_hyphenation_fix(self):
        raw = "volatil-\nity model"
        norm = normalize_page_text(raw)
        self.assertIn("volatility model", norm)

    def test_whitespace_collapse(self):
        raw = "a\t\tb   c\r\nd"
        norm = normalize_page_text(raw)
        self.assertEqual(norm, "a b c\nd")


class TestChunking(unittest.TestCase):
    def test_fixed_chunker_produces_page_spans(self):
        pages = [
            ParsedPage(doc_id="d", page_number=1, text="x " * 2000, char_count=4000),
        ]
        cfg = FixedChunkerConfig(max_chars=500, overlap_chars=50, min_chars=100)
        chunks = list(chunk_pages_fixed(pages, doc_id="d", cfg=cfg))
        self.assertGreaterEqual(len(chunks), 2)
        for ch in chunks:
            self.assertEqual(ch.page_span.start_page, 1)
            self.assertEqual(ch.page_span.end_page, 1)
            self.assertIsNotNone(ch.page_span.start_char)
            self.assertIsNotNone(ch.page_span.end_char)

    def test_section_aware_detects_headings_and_sets_section_path(self):
        pages = [
            ParsedPage(
                doc_id="d",
                page_number=1,
                text="1 Introduction\n" + ("hello " * 200) + "\n2 Model\n" + ("world " * 200),
                char_count=1,
            )
        ]
        cfg = SectionAwareChunkerConfig(max_chars=800, overlap_chars=0, min_chars=200)
        chunks = list(chunk_pages_section_aware(pages, doc_id="d", cfg=cfg))
        self.assertTrue(any(ch.section_path for ch in chunks))
        self.assertTrue(any("1 Introduction" in " > ".join(ch.section_path) for ch in chunks))
    
    def test_fixed_chunker_overlap(self):
        """Test that fixed chunker creates proper overlap."""
        pages = [
            ParsedPage(doc_id="d", page_number=1, text="a" * 1000, char_count=1000),
        ]
        cfg = FixedChunkerConfig(max_chars=300, overlap_chars=50, min_chars=100)
        chunks = list(chunk_pages_fixed(pages, doc_id="d", cfg=cfg))
        
        # Verify overlap exists between consecutive chunks
        self.assertGreaterEqual(len(chunks), 2)
        for i in range(len(chunks) - 1):
            curr_end = chunks[i].page_span.end_char
            next_start = chunks[i+1].page_span.start_char
            if curr_end and next_start:
                overlap = curr_end - next_start
                self.assertGreater(overlap, 0, "Expected overlap between chunks")
    
    def test_multi_page_chunking(self):
        """Test chunking across multiple pages."""
        pages = [
            ParsedPage(doc_id="d", page_number=1, text="Page 1 " * 100, char_count=700),
            ParsedPage(doc_id="d", page_number=2, text="Page 2 " * 100, char_count=700),
            ParsedPage(doc_id="d", page_number=3, text="Page 3 " * 100, char_count=700),
        ]
        cfg = FixedChunkerConfig(max_chars=500, overlap_chars=50, min_chars=100)
        chunks = list(chunk_pages_fixed(pages, doc_id="d", cfg=cfg))
        
        # Should have chunks from multiple pages
        page_nums = set()
        for chunk in chunks:
            page_nums.add(chunk.page_span.start_page)
        
        self.assertGreater(len(page_nums), 1, "Expected chunks from multiple pages")
    
    def test_section_aware_nested_sections(self):
        """Test section-aware chunking with nested headings."""
        text = """1 Introduction
This is the introduction section with some content.

1.1 Background
Background subsection content goes here.

1.2 Motivation  
Motivation subsection with more detailed text.

2 Methodology
Main methodology section begins here.

2.1 Data Collection
Details about data collection methods.
"""
        pages = [ParsedPage(doc_id="d", page_number=1, text=text, char_count=len(text))]
        cfg = SectionAwareChunkerConfig(max_chars=200, overlap_chars=0, min_chars=50)
        chunks = list(chunk_pages_section_aware(pages, doc_id="d", cfg=cfg))
        
        # Verify we detected sections
        self.assertGreater(len(chunks), 0)
        
        # Check that some chunks have section paths
        chunks_with_sections = [c for c in chunks if len(c.section_path) > 0]
        self.assertGreater(len(chunks_with_sections), 0)
    
    def test_chunk_char_count_matches_text_length(self):
        """Test that chunk char_count matches actual text length."""
        pages = [
            ParsedPage(doc_id="d", page_number=1, text="test text", char_count=9),
        ]
        cfg = FixedChunkerConfig(max_chars=500, overlap_chars=0, min_chars=5)
        chunks = list(chunk_pages_fixed(pages, doc_id="d", cfg=cfg))
        
        for chunk in chunks:
            self.assertEqual(chunk.char_count, len(chunk.text))


if __name__ == "__main__":
    unittest.main()


