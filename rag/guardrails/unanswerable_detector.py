"""
Unanswerable question detector.

Detects questions that cannot/should not be answered from the corpus.
"""
import logging
from typing import List, Optional, Set

from knowledge.models import Answer, RetrievedChunk


logger = logging.getLogger(__name__)


# Keywords indicating unanswerable questions
UNANSWERABLE_PATTERNS = [
    # Questions about future/predictions
    "will",
    "predict",
    "forecast",
    "in 2024",
    "in 2025",
    "next year",
    
    # Questions requiring real-time data
    "current price",
    "today",
    "right now",
    "latest",
    
    # Questions outside domain
    "recipe",
    "weather",
    "sports",
    "celebrity",
    
    # Personal questions
    "what should i",
    "should i buy",
    "should i sell",
    "my portfolio",
]


# Keywords indicating the query might be about specific things not in corpus
OUT_OF_SCOPE_INDICATORS = [
    "cryptocurrency",
    "bitcoin",
    "ethereum",
    "stock recommendation",
    "political",
]


class UnanswerableDetector:
    """
    Detects unanswerable questions before/after retrieval.
    
    Uses rule-based checks to identify questions that:
    1. Are clearly out of scope
    2. Require information not in the corpus
    3. Ask for predictions or advice
    """
    
    def __init__(
        self,
        min_retrieval_score: float = 0.1,
        max_results_check: int = 5
    ):
        """
        Initialize detector.
        
        Args:
            min_retrieval_score: Minimum score threshold for top result
            max_results_check: Number of top results to check
        """
        self.min_retrieval_score = min_retrieval_score
        self.max_results_check = max_results_check
    
    def _check_query_patterns(self, query: str) -> Optional[str]:
        """
        Check if query matches unanswerable patterns.
        
        Args:
            query: User query
            
        Returns:
            Refusal reason if unanswerable, None otherwise
        """
        query_lower = query.lower()
        
        # Check unanswerable patterns
        for pattern in UNANSWERABLE_PATTERNS:
            if pattern in query_lower:
                logger.info(f"Query matches unanswerable pattern: {pattern}")
                return "out_of_scope"
        
        # Check out-of-scope indicators
        for indicator in OUT_OF_SCOPE_INDICATORS:
            if indicator in query_lower:
                logger.info(f"Query matches out-of-scope indicator: {indicator}")
                return "out_of_scope"
        
        return None
    
    def _check_retrieval_quality(
        self,
        chunks: List[RetrievedChunk]
    ) -> Optional[str]:
        """
        Check if retrieval quality is sufficient.
        
        Args:
            chunks: Retrieved chunks
            
        Returns:
            Refusal reason if insufficient, None otherwise
        """
        if not chunks:
            logger.info("No chunks retrieved")
            return "no_relevant_context"
        
        # Check top result score
        top_score = chunks[0].score if chunks else 0.0
        if top_score < self.min_retrieval_score:
            logger.info(f"Top retrieval score too low: {top_score:.3f}")
            return "low_retrieval_confidence"
        
        return None
    
    def detect_pre_retrieval(self, query: str) -> Optional[str]:
        """
        Detect unanswerable questions before retrieval.
        
        Fast check to avoid unnecessary retrieval/generation.
        
        Args:
            query: User query
            
        Returns:
            Refusal reason if unanswerable, None otherwise
        """
        return self._check_query_patterns(query)
    
    def detect_post_retrieval(
        self,
        query: str,
        chunks: List[RetrievedChunk]
    ) -> Optional[str]:
        """
        Detect unanswerable questions after retrieval.
        
        Args:
            query: User query
            chunks: Retrieved chunks
            
        Returns:
            Refusal reason if unanswerable, None otherwise
        """
        # Check retrieval quality
        retrieval_reason = self._check_retrieval_quality(chunks)
        if retrieval_reason:
            return retrieval_reason
        
        # Could add more sophisticated checks here
        # e.g., semantic similarity between query and top chunks
        
        return None
    
    def validate(
        self,
        query: str,
        answer: Answer,
        chunks: Optional[List[RetrievedChunk]] = None,
        **kwargs
    ) -> Answer:
        """
        Validate answer for unanswerable questions.
        
        This is called as a guardrail in the pipeline.
        
        Args:
            query: Original query
            answer: Generated answer
            chunks: Retrieved chunks
            **kwargs: Additional parameters
            
        Returns:
            Answer (potentially refused if unanswerable)
        """
        # If already refused, pass through
        if answer.is_refused():
            return answer
        
        # Pre-retrieval check
        pre_reason = self.detect_pre_retrieval(query)
        if pre_reason:
            return Answer(
                text="I cannot answer this question as it appears to be outside the scope of the available documents.",
                citations=[],
                confidence=0.0,
                refusal_reason=pre_reason,
                trace=answer.trace,
                metadata={
                    **answer.metadata,
                    "unanswerable_check": "failed_pre_retrieval"
                }
            )
        
        # Post-retrieval check
        if chunks:
            post_reason = self.detect_post_retrieval(query, chunks)
            if post_reason:
                return Answer(
                    text="I cannot find sufficient relevant information in the documents to answer this question confidently.",
                    citations=[],
                    confidence=0.0,
                    refusal_reason=post_reason,
                    trace=answer.trace,
                    metadata={
                        **answer.metadata,
                        "unanswerable_check": "failed_post_retrieval"
                    }
                )
        
        # Passed checks
        return Answer(
            text=answer.text,
            citations=answer.citations,
            confidence=answer.confidence,
            refusal_reason=answer.refusal_reason,
            trace=answer.trace,
            metadata={
                **answer.metadata,
                "unanswerable_check": "passed"
            }
        )

