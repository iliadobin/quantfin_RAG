"""
Unit tests for RAG contracts (protocols).

Tests that implementations properly follow the protocol interfaces.
"""
import unittest
from typing import List
from knowledge.models import (
    Chunk, RetrievedChunk, Citation, Answer, PageSpan
)
from rag.contracts import Retriever, Reranker, Generator, Pipeline


class MockRetriever:
    """Mock retriever for testing protocol compliance."""
    
    def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[RetrievedChunk]:
        """Mock retrieve method."""
        # Create mock chunks
        chunks = []
        for i in range(min(top_k, 3)):
            span = PageSpan(start_page=i+1, end_page=i+1)
            chunk = Chunk(
                id=f"chunk_{i}",
                doc_id="test_doc",
                strategy="mock",
                text=f"Mock text for query: {query}",
                page_span=span,
                token_count=10,
                char_count=50
            )
            chunks.append(RetrievedChunk(
                chunk=chunk,
                score=0.9 - i*0.1,
                retriever_tag="mock"
            ))
        return chunks
    
    def batch_retrieve(self, queries: List[str], top_k: int = 10, **kwargs) -> List[List[RetrievedChunk]]:
        """Mock batch retrieve method."""
        return [self.retrieve(q, top_k, **kwargs) for q in queries]


class MockReranker:
    """Mock reranker for testing protocol compliance."""
    
    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int = 5, **kwargs) -> List[RetrievedChunk]:
        """Mock rerank method - just sorts by score and limits."""
        sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
        return sorted_chunks[:top_k]


class MockGenerator:
    """Mock generator for testing protocol compliance."""
    
    def generate(self, query: str, chunks: List[RetrievedChunk], **kwargs) -> Answer:
        """Mock generate method."""
        # Create mock citations from chunks
        citations = []
        for retrieved in chunks[:2]:  # Use first 2 chunks
            citation = Citation(
                doc_id=retrieved.chunk.doc_id,
                page_span=retrieved.chunk.page_span,
                quote=retrieved.chunk.text[:50],
                score=retrieved.score,
                retriever_tag=retrieved.retriever_tag
            )
            citations.append(citation)
        
        return Answer(
            text=f"Mock answer for: {query}",
            citations=citations,
            confidence=0.85
        )


class MockPipeline:
    """Mock pipeline for testing protocol compliance."""
    
    def __init__(self):
        self.retriever = MockRetriever()
        self.reranker = MockReranker()
        self.generator = MockGenerator()
        self._name = "mock_pipeline_v1"
        self._version = "1.0.0"
    
    def run(self, query: str, corpus_profile: str = "public", **kwargs) -> Answer:
        """Mock pipeline run."""
        # Retrieve
        chunks = self.retriever.retrieve(query, top_k=10)
        
        # Rerank
        chunks = self.reranker.rerank(query, chunks, top_k=5)
        
        # Generate
        answer = self.generator.generate(query, chunks)
        
        return answer
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> str:
        return self._version


class TestRetrieverProtocol(unittest.TestCase):
    """Test Retriever protocol compliance."""
    
    def setUp(self):
        self.retriever = MockRetriever()
    
    def test_implements_retrieve(self):
        """Test that retriever implements retrieve method."""
        self.assertTrue(hasattr(self.retriever, 'retrieve'))
        self.assertTrue(callable(self.retriever.retrieve))
    
    def test_retrieve_returns_chunks(self):
        """Test that retrieve returns list of RetrievedChunk."""
        results = self.retriever.retrieve("test query", top_k=5)
        self.assertIsInstance(results, list)
        self.assertTrue(all(isinstance(r, RetrievedChunk) for r in results))
    
    def test_retrieve_respects_top_k(self):
        """Test that retrieve respects top_k parameter."""
        results = self.retriever.retrieve("test query", top_k=2)
        self.assertLessEqual(len(results), 2)
    
    def test_batch_retrieve(self):
        """Test batch retrieve method."""
        queries = ["query 1", "query 2", "query 3"]
        results = self.retriever.batch_retrieve(queries, top_k=3)
        
        self.assertEqual(len(results), 3)
        self.assertTrue(all(isinstance(r, list) for r in results))


class TestRerankerProtocol(unittest.TestCase):
    """Test Reranker protocol compliance."""
    
    def setUp(self):
        self.reranker = MockReranker()
        self.retriever = MockRetriever()
    
    def test_implements_rerank(self):
        """Test that reranker implements rerank method."""
        self.assertTrue(hasattr(self.reranker, 'rerank'))
        self.assertTrue(callable(self.reranker.rerank))
    
    def test_rerank_returns_chunks(self):
        """Test that rerank returns list of RetrievedChunk."""
        chunks = self.retriever.retrieve("test query", top_k=10)
        reranked = self.reranker.rerank("test query", chunks, top_k=5)
        
        self.assertIsInstance(reranked, list)
        self.assertTrue(all(isinstance(r, RetrievedChunk) for r in reranked))
    
    def test_rerank_respects_top_k(self):
        """Test that rerank respects top_k parameter."""
        chunks = self.retriever.retrieve("test query", top_k=10)
        reranked = self.reranker.rerank("test query", chunks, top_k=3)
        
        self.assertLessEqual(len(reranked), 3)


class TestGeneratorProtocol(unittest.TestCase):
    """Test Generator protocol compliance."""
    
    def setUp(self):
        self.generator = MockGenerator()
        self.retriever = MockRetriever()
    
    def test_implements_generate(self):
        """Test that generator implements generate method."""
        self.assertTrue(hasattr(self.generator, 'generate'))
        self.assertTrue(callable(self.generator.generate))
    
    def test_generate_returns_answer(self):
        """Test that generate returns Answer."""
        chunks = self.retriever.retrieve("test query", top_k=5)
        answer = self.generator.generate("test query", chunks)
        
        self.assertIsInstance(answer, Answer)
    
    def test_generate_includes_citations(self):
        """Test that generated answer includes citations."""
        chunks = self.retriever.retrieve("test query", top_k=5)
        answer = self.generator.generate("test query", chunks)
        
        self.assertIsInstance(answer.citations, list)
        self.assertTrue(all(isinstance(c, Citation) for c in answer.citations))


class TestPipelineProtocol(unittest.TestCase):
    """Test Pipeline protocol compliance."""
    
    def setUp(self):
        self.pipeline = MockPipeline()
    
    def test_implements_run(self):
        """Test that pipeline implements run method."""
        self.assertTrue(hasattr(self.pipeline, 'run'))
        self.assertTrue(callable(self.pipeline.run))
    
    def test_run_returns_answer(self):
        """Test that run returns Answer."""
        answer = self.pipeline.run("What is delta hedging?")
        self.assertIsInstance(answer, Answer)
    
    def test_pipeline_has_name(self):
        """Test that pipeline has name property."""
        self.assertTrue(hasattr(self.pipeline, 'name'))
        self.assertIsInstance(self.pipeline.name, str)
        self.assertEqual(self.pipeline.name, "mock_pipeline_v1")
    
    def test_pipeline_has_version(self):
        """Test that pipeline has version property."""
        self.assertTrue(hasattr(self.pipeline, 'version'))
        self.assertIsInstance(self.pipeline.version, str)
        self.assertEqual(self.pipeline.version, "1.0.0")
    
    def test_run_with_corpus_profile(self):
        """Test run with corpus_profile parameter."""
        answer = self.pipeline.run("test query", corpus_profile="public")
        self.assertIsInstance(answer, Answer)


class TestProtocolInteroperability(unittest.TestCase):
    """Test that protocol components work together."""
    
    def test_full_pipeline_flow(self):
        """Test complete pipeline flow with all components."""
        retriever = MockRetriever()
        reranker = MockReranker()
        generator = MockGenerator()
        
        query = "What are option Greeks?"
        
        # Step 1: Retrieve
        chunks = retriever.retrieve(query, top_k=10)
        self.assertGreater(len(chunks), 0)
        
        # Step 2: Rerank
        reranked = reranker.rerank(query, chunks, top_k=5)
        self.assertLessEqual(len(reranked), 5)
        
        # Step 3: Generate
        answer = generator.generate(query, reranked)
        self.assertIsInstance(answer, Answer)
        self.assertGreater(len(answer.citations), 0)
    
    def test_pipeline_integration(self):
        """Test that full pipeline integrates all components."""
        pipeline = MockPipeline()
        
        answer = pipeline.run("What is delta?")
        
        # Verify answer structure
        self.assertIsInstance(answer, Answer)
        self.assertIsNotNone(answer.text)
        self.assertGreater(len(answer.citations), 0)
        self.assertGreater(answer.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()

