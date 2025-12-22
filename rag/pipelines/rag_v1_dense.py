"""
RAGv1: Dense retrieval pipeline.

Simple but effective: dense retrieval + generation with citations.
"""
import time
import logging
from typing import Optional, List

from knowledge.models import Answer, RetrievalTrace, RetrievedChunk
from llm.deepseek_client import DeepSeekClient
from rag.retrievers.dense_retriever import DenseRetriever
from rag.generators.citation_generator import CitationGenerator
from rag.guardrails.unanswerable_detector import UnanswerableDetector


logger = logging.getLogger(__name__)


class RAGv1Dense:
    """
    RAG v1: Dense retrieval pipeline.
    
    Pipeline:
    1. Dense retrieval (sentence transformer + FAISS)
    2. Generate answer with citations
    3. Optional: unanswerable detection
    """
    
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        llm_client: DeepSeekClient,
        use_unanswerable_detection: bool = True,
        top_k: int = 10,
        model: Optional[str] = None
    ):
        """
        Initialize RAGv1 pipeline.
        
        Args:
            dense_retriever: Dense retriever instance
            llm_client: DeepSeek client
            use_unanswerable_detection: Whether to detect unanswerable questions
            top_k: Number of chunks to retrieve
            model: LLM model to use
        """
        self.dense_retriever = dense_retriever
        self.llm_client = llm_client
        self.top_k = top_k
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
        Run RAGv1 pipeline.
        
        Args:
            query: User query
            corpus_profile: Corpus to use (currently unused, for future)
            **kwargs: Additional parameters (top_k, etc.)
            
        Returns:
            Answer with citations and trace
        """
        start_time = time.time()
        
        # Override defaults with kwargs
        top_k = kwargs.get('top_k', self.top_k)
        
        # Pre-retrieval unanswerable check
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
                        metadata={"pipeline": "rag_v1", "stage": "pre_retrieval_check"}
                    )
                )
        
        # Step 1: Dense retrieval
        retrieval_start = time.time()
        chunks = self.dense_retriever.retrieve(query, top_k=top_k)
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        logger.info(f"Retrieved {len(chunks)} chunks in {retrieval_time:.1f}ms")
        
        # Post-retrieval unanswerable check
        if self.unanswerable_detector:
            refusal_reason = self.unanswerable_detector.detect_post_retrieval(query, chunks)
            if refusal_reason:
                logger.info(f"Query refused post-retrieval: {refusal_reason}")
                return Answer(
                    text="I cannot find sufficient relevant information in the documents to answer this question confidently.",
                    citations=[],
                    confidence=0.0,
                    refusal_reason=refusal_reason,
                    trace=RetrievalTrace(
                        query=query,
                        retrieved_chunks_count=len(chunks),
                        final_chunks_count=len(chunks),
                        retrieval_time_ms=retrieval_time,
                        metadata={"pipeline": "rag_v1", "stage": "post_retrieval_check"}
                    )
                )
        
        # Step 2: Generate answer
        generation_start = time.time()
        answer = self.generator.generate(query, chunks)
        generation_time = (time.time() - generation_start) * 1000
        
        logger.info(f"Generated answer in {generation_time:.1f}ms with {len(answer.citations)} citations")
        
        # Add trace
        total_time = (time.time() - start_time) * 1000
        answer.trace = RetrievalTrace(
            query=query,
            retrieved_chunks_count=len(chunks),
            final_chunks_count=len(chunks),
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            metadata={
                "pipeline": "rag_v1",
                "corpus_profile": corpus_profile,
                "total_time_ms": total_time
            }
        )
        
        return answer
    
    @property
    def name(self) -> str:
        """Pipeline name."""
        return "RAGv1_Dense"
    
    @property
    def version(self) -> str:
        """Pipeline version."""
        return "1.0"

