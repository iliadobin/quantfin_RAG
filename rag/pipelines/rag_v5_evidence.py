"""
RAGv5: Evidence validation pipeline.

Generates answer, then validates that all claims are supported by evidence.
Uses rule-based validation primarily, with optional LLM validation.
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
from rag.guardrails.evidence_validator import EvidenceValidator


logger = logging.getLogger(__name__)


class RAGv5Evidence:
    """
    RAG v5: Evidence validation pipeline.
    
    Pipeline:
    1. Hybrid retrieval + reranking
    2. Generate answer with citations
    3. Validate evidence (rule-based or LLM)
    4. Refuse or adjust confidence if validation fails
    """
    
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        llm_client: DeepSeekClient,
        use_llm_validation: bool = False,
        use_unanswerable_detection: bool = True,
        retrieval_top_k: int = 20,
        rerank_top_k: int = 10,
        min_citation_coverage: float = 0.5,
        model: Optional[str] = None
    ):
        """
        Initialize RAGv5 pipeline.
        
        Args:
            hybrid_retriever: Hybrid retriever
            reranker: Cross-encoder reranker
            llm_client: DeepSeek client
            use_llm_validation: Whether to use LLM for validation (costs tokens)
            use_unanswerable_detection: Whether to detect unanswerable questions
            retrieval_top_k: Number of chunks before reranking
            rerank_top_k: Number of chunks after reranking
            min_citation_coverage: Minimum citation coverage for validation
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
        
        self.evidence_validator = EvidenceValidator(
            llm_client=llm_client if use_llm_validation else None,
            use_llm=use_llm_validation,
            min_citation_coverage=min_citation_coverage
        )
        
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
        Run RAGv5 pipeline.
        
        Args:
            query: User query
            corpus_profile: Corpus to use
            **kwargs: Additional parameters
            
        Returns:
            Answer with validated citations and trace
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
                        metadata={"pipeline": "rag_v5", "stage": "pre_retrieval_check"}
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
                        metadata={"pipeline": "rag_v5", "stage": "post_retrieval_check"}
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
        
        # Step 4: Validate evidence
        validation_start = time.time()
        validated_answer = self.evidence_validator.validate(
            query,
            answer,
            chunks=reranked_chunks
        )
        validation_time = (time.time() - validation_start) * 1000
        
        logger.info(f"Evidence validation in {validation_time:.1f}ms: {validated_answer.metadata.get('validation', 'unknown')}")
        
        # Add trace
        total_time = (time.time() - start_time) * 1000
        validated_answer.trace = RetrievalTrace(
            query=query,
            retrieved_chunks_count=len(retrieved_chunks),
            reranked_chunks_count=len(reranked_chunks),
            final_chunks_count=len(reranked_chunks),
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            metadata={
                "pipeline": "rag_v5",
                "corpus_profile": corpus_profile,
                "rerank_time_ms": rerank_time,
                "validation_time_ms": validation_time,
                "total_time_ms": total_time,
                "validation_result": validated_answer.metadata.get("validation")
            }
        )
        
        return validated_answer
    
    @property
    def name(self) -> str:
        """Pipeline name."""
        return "RAGv5_Evidence"
    
    @property
    def version(self) -> str:
        """Pipeline version."""
        return "5.0"

