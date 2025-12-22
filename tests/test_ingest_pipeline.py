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


if __name__ == "__main__":
    unittest.main()


