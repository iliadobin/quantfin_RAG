"""
RAGv2: Hybrid retrieval with reranking.

Combines BM25 + dense retrieval, then reranks with cross-encoder.
"""
import time
import logging
from typing import Optional

from knowledge.models import Answer, RetrievalTrace
from llm.deepseek_client import DeepSeekClient
from rag.retrievers.hybrid_retriever import HybridRetriever
from rag.rerankers.cross_encoder_reranker import CrossEncoderReranker
from rag.generators.citation_generator import CitationGenerator
from rag.guardrails.unanswerable_detector import UnanswerableDetector


logger = logging.getLogger(__name__)


class RAGv2Hybrid:
    """
    RAG v2: Hybrid retrieval + reranking.
    
    Pipeline:
    1. Hybrid retrieval (BM25 + dense with RRF fusion)
    2. Rerank with cross-encoder
    3. Generate answer with citations
    4. Optional: unanswerable detection
    """
    
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        llm_client: DeepSeekClient,
        use_unanswerable_detection: bool = True,
        retrieval_top_k: int = 20,
        rerank_top_k: int = 10,
        model: Optional[str] = None
    ):
        """
        Initialize RAGv2 pipeline.
        
        Args:
            hybrid_retriever: Hybrid retriever instance
            reranker: Cross-encoder reranker
            llm_client: DeepSeek client
            use_unanswerable_detection: Whether to detect unanswerable questions
            retrieval_top_k: Number of chunks to retrieve before reranking
            rerank_top_k: Number of chunks after reranking
            model: LLM model to use
        """
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.llm_client = llm_client
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k
        self.model = model
        
        # Components
        self.generator = CitationGenerator(llm_client, model=model)
        
        if use_unanswerable_detection:
            self.unanswerable_detector = UnanswerableDetector()
        else:
            self.unanswerable_detector = None
    
    def run(
        self,
        query: str,
        corpus_profile: str = "public",
        **kwargs
    ) -> Answer:
        """
        Run RAGv2 pipeline.
        
        Args:
            query: User query
            corpus_profile: Corpus to use
            **kwargs: Additional parameters
            
        Returns:
            Answer with citations and trace
        """
        start_time = time.time()
        
        # Override defaults
        retrieval_top_k = kwargs.get('retrieval_top_k', self.retrieval_top_k)
        rerank_top_k = kwargs.get('rerank_top_k', self.rerank_top_k)
        
        # Pre-retrieval check
        if self.unanswerable_detector:
            refusal_reason = self.unanswerable_detector.detect_pre_retrieval(query)
            if refusal_reason:
                logger.info(f"Query refused pre-retrieval: {refusal_reason}")
                return Answer(
                    text="I cannot answer this question as it appears to be outside the scope of the available documents.",
                    citations=[],
                    confidence=0.0,
                    refusal_reason=refusal_reason,
                    trace=RetrievalTrace(
                        query=query,
                        metadata={"pipeline": "rag_v2", "stage": "pre_retrieval_check"}
                    )
                )
        
        # Step 1: Hybrid retrieval
        retrieval_start = time.time()
        retrieved_chunks = self.hybrid_retriever.retrieve(query, top_k=retrieval_top_k)
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        logger.info(f"Hybrid retrieval: {len(retrieved_chunks)} chunks in {retrieval_time:.1f}ms")
        
        # Post-retrieval check
        if self.unanswerable_detector:
            refusal_reason = self.unanswerable_detector.detect_post_retrieval(query, retrieved_chunks)
            if refusal_reason:
                logger.info(f"Query refused post-retrieval: {refusal_reason}")
                return Answer(
                    text="I cannot find sufficient relevant information in the documents to answer this question confidently.",
                    citations=[],
                    confidence=0.0,
                    refusal_reason=refusal_reason,
                    trace=RetrievalTrace(
                        query=query,
                        retrieved_chunks_count=len(retrieved_chunks),
                        retrieval_time_ms=retrieval_time,
                        metadata={"pipeline": "rag_v2", "stage": "post_retrieval_check"}
                    )
                )
        
        # Step 2: Rerank
        rerank_start = time.time()
        reranked_chunks = self.reranker.rerank(query, retrieved_chunks, top_k=rerank_top_k)
        rerank_time = (time.time() - rerank_start) * 1000
        
        logger.info(f"Reranked to {len(reranked_chunks)} chunks in {rerank_time:.1f}ms")
        
        # Step 3: Generate answer
        generation_start = time.time()
        answer = self.generator.generate(query, reranked_chunks)
        generation_time = (time.time() - generation_start) * 1000
        
        logger.info(f"Generated answer in {generation_time:.1f}ms with {len(answer.citations)} citations")
        
        # Add trace
        total_time = (time.time() - start_time) * 1000
        answer.trace = RetrievalTrace(
            query=query,
            retrieved_chunks_count=len(retrieved_chunks),
            reranked_chunks_count=len(reranked_chunks),
            final_chunks_count=len(reranked_chunks),
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            metadata={
                "pipeline": "rag_v2",
                "corpus_profile": corpus_profile,
                "rerank_time_ms": rerank_time,
                "total_time_ms": total_time
            }
        )
        
        return answer
    
    @property
    def name(self) -> str:
        """Pipeline name."""
        return "RAGv2_Hybrid"
    
    @property
    def version(self) -> str:
        """Pipeline version."""
        return "2.0"

