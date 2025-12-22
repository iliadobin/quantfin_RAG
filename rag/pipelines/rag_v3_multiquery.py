"""
RAGv3: Multi-query with template-based expansion.

Uses rule-based query expansion (cheap, no LLM) and fusion.
"""
import time
import logging
from typing import Optional

from knowledge.models import Answer, RetrievalTrace
from llm.deepseek_client import DeepSeekClient
from rag.retrievers.multi_query_retriever import MultiQueryRetriever
from rag.rerankers.cross_encoder_reranker import CrossEncoderReranker
from rag.generators.citation_generator import CitationGenerator
from rag.guardrails.unanswerable_detector import UnanswerableDetector


logger = logging.getLogger(__name__)


class RAGv3MultiQuery:
    """
    RAG v3: Multi-query with fusion.
    
    Pipeline:
    1. Expand query using templates (pricing, Greeks, methods, etc.)
    2. Retrieve for each query variant
    3. Fuse results with RRF
    4. Optional: rerank
    5. Generate answer with citations
    """
    
    def __init__(
        self,
        multi_query_retriever: MultiQueryRetriever,
        llm_client: DeepSeekClient,
        reranker: Optional[CrossEncoderReranker] = None,
        use_unanswerable_detection: bool = True,
        retrieval_top_k: int = 15,
        rerank_top_k: int = 10,
        model: Optional[str] = None
    ):
        """
        Initialize RAGv3 pipeline.
        
        Args:
            multi_query_retriever: Multi-query retriever
            llm_client: DeepSeek client
            reranker: Optional cross-encoder reranker
            use_unanswerable_detection: Whether to detect unanswerable questions
            retrieval_top_k: Number of chunks after fusion
            rerank_top_k: Number of chunks after reranking (if used)
            model: LLM model to use
        """
        self.multi_query_retriever = multi_query_retriever
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
        Run RAGv3 pipeline.
        
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
        use_rerank = kwargs.get('use_rerank', self.reranker is not None)
        
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
                        metadata={"pipeline": "rag_v3", "stage": "pre_retrieval_check"}
                    )
                )
        
        # Step 1: Multi-query retrieval with fusion
        retrieval_start = time.time()
        fused_chunks = self.multi_query_retriever.retrieve(query, top_k=retrieval_top_k)
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        # Get expanded queries from metadata (if available)
        expanded_queries = []
        if fused_chunks and hasattr(self.multi_query_retriever, '_expand_query'):
            expanded_queries = self.multi_query_retriever._expand_query(query)
        
        logger.info(f"Multi-query retrieval: {len(expanded_queries)} queries, {len(fused_chunks)} fused chunks in {retrieval_time:.1f}ms")
        
        # Post-retrieval check
        if self.unanswerable_detector:
            refusal_reason = self.unanswerable_detector.detect_post_retrieval(query, fused_chunks)
            if refusal_reason:
                logger.info(f"Query refused post-retrieval: {refusal_reason}")
                return Answer(
                    text="I cannot find sufficient relevant information in the documents to answer this question confidently.",
                    citations=[],
                    confidence=0.0,
                    refusal_reason=refusal_reason,
                    trace=RetrievalTrace(
                        query=query,
                        expanded_queries=expanded_queries,
                        retrieved_chunks_count=len(fused_chunks),
                        retrieval_time_ms=retrieval_time,
                        metadata={"pipeline": "rag_v3", "stage": "post_retrieval_check"}
                    )
                )
        
        # Step 2: Optional reranking
        rerank_time = 0.0
        final_chunks = fused_chunks
        
        if use_rerank and self.reranker:
            rerank_start = time.time()
            final_chunks = self.reranker.rerank(query, fused_chunks, top_k=rerank_top_k)
            rerank_time = (time.time() - rerank_start) * 1000
            logger.info(f"Reranked to {len(final_chunks)} chunks in {rerank_time:.1f}ms")
        
        # Step 3: Generate answer
        generation_start = time.time()
        answer = self.generator.generate(query, final_chunks)
        generation_time = (time.time() - generation_start) * 1000
        
        logger.info(f"Generated answer in {generation_time:.1f}ms with {len(answer.citations)} citations")
        
        # Add trace
        total_time = (time.time() - start_time) * 1000
        answer.trace = RetrievalTrace(
            query=query,
            expanded_queries=expanded_queries,
            retrieved_chunks_count=len(fused_chunks),
            reranked_chunks_count=len(final_chunks) if use_rerank else 0,
            final_chunks_count=len(final_chunks),
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            metadata={
                "pipeline": "rag_v3",
                "corpus_profile": corpus_profile,
                "rerank_time_ms": rerank_time if use_rerank else 0,
                "total_time_ms": total_time,
                "used_rerank": use_rerank
            }
        )
        
        return answer
    
    @property
    def name(self) -> str:
        """Pipeline name."""
        return "RAGv3_MultiQuery"
    
    @property
    def version(self) -> str:
        """Pipeline version."""
        return "3.0"

