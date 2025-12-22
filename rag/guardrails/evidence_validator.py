"""
Evidence validator - ensures claims are supported by retrieved context.
"""
import re
import logging
from typing import List, Optional, Dict, Any

from knowledge.models import Answer, RetrievedChunk
from llm.deepseek_client import DeepSeekClient
from rag.generators import prompts


logger = logging.getLogger(__name__)


class EvidenceValidator:
    """
    Validates that answer claims are supported by evidence.
    
    Can use rules-based checks or LLM-based validation.
    """
    
    def __init__(
        self,
        llm_client: Optional[DeepSeekClient] = None,
        use_llm: bool = False,
        min_citation_coverage: float = 0.5,
        min_confidence_threshold: float = 0.3
    ):
        """
        Initialize evidence validator.
        
        Args:
            llm_client: DeepSeek client (required if use_llm=True)
            use_llm: Whether to use LLM for validation (costs tokens)
            min_citation_coverage: Minimum ratio of sentences with citations
            min_confidence_threshold: Minimum confidence to pass
        """
        self.llm_client = llm_client
        self.use_llm = use_llm
        self.min_citation_coverage = min_citation_coverage
        self.min_confidence_threshold = min_confidence_threshold
        
        if use_llm and not llm_client:
            raise ValueError("llm_client required when use_llm=True")
    
    def _count_sentences(self, text: str) -> int:
        """Count sentences in text (simple heuristic)."""
        # Split by period, question mark, exclamation
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])
    
    def _count_citations(self, text: str) -> int:
        """Count citation markers in text."""
        pattern = r'\[\d+\]'
        return len(re.findall(pattern, text))
    
    def _rule_based_validation(self, answer: Answer) -> Answer:
        """
        Rule-based validation.
        
        Checks:
        1. Answer has citations
        2. Citation coverage is sufficient
        3. Confidence meets threshold
        
        Args:
            answer: Answer to validate
            
        Returns:
            Validated answer (potentially marked as refused)
        """
        # If already refused, pass through
        if answer.is_refused():
            return answer
        
        # Check for citations
        if not answer.citations:
            logger.warning("Answer has no citations - refusing")
            return Answer(
                text=answer.text,
                citations=[],
                confidence=0.0,
                refusal_reason="no_evidence",
                trace=answer.trace,
                metadata={
                    **answer.metadata,
                    "validation": "failed_no_citations"
                }
            )
        
        # Check citation coverage
        num_sentences = self._count_sentences(answer.text)
        num_citations = self._count_citations(answer.text)
        
        if num_sentences > 0:
            coverage = num_citations / num_sentences
        else:
            coverage = 0.0
        
        if coverage < self.min_citation_coverage:
            logger.warning(f"Low citation coverage: {coverage:.2f} < {self.min_citation_coverage}")
            return Answer(
                text=answer.text,
                citations=answer.citations,
                confidence=answer.confidence * 0.5,  # Penalize confidence
                refusal_reason=None,  # Don't refuse, just lower confidence
                trace=answer.trace,
                metadata={
                    **answer.metadata,
                    "validation": "low_coverage",
                    "citation_coverage": coverage
                }
            )
        
        # Check confidence threshold
        if answer.confidence < self.min_confidence_threshold:
            logger.warning(f"Low confidence: {answer.confidence:.2f}")
            return Answer(
                text="I'm not confident in answering this question based on the available documents.",
                citations=answer.citations,
                confidence=answer.confidence,
                refusal_reason="low_confidence",
                trace=answer.trace,
                metadata={
                    **answer.metadata,
                    "validation": "failed_confidence"
                }
            )
        
        # Validation passed
        return Answer(
            text=answer.text,
            citations=answer.citations,
            confidence=answer.confidence,
            refusal_reason=answer.refusal_reason,
            trace=answer.trace,
            metadata={
                **answer.metadata,
                "validation": "passed",
                "citation_coverage": coverage
            }
        )
    
    def _llm_based_validation(
        self,
        answer: Answer,
        chunks: List[RetrievedChunk]
    ) -> Answer:
        """
        LLM-based validation (costs tokens).
        
        Uses LLM to check if claims are supported by context.
        
        Args:
            answer: Answer to validate
            chunks: Context chunks used
            
        Returns:
            Validated answer
        """
        if answer.is_refused():
            return answer
        
        # Build validation prompt
        validation_prompt = prompts.build_validation_prompt(
            answer.text,
            chunks,
            max_chunks=10
        )
        
        try:
            response = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": prompts.SYSTEM_PROMPT_VALIDATION},
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=0.0,
                max_tokens=1024
            )
            
            validation_result = response["content"]
            
            # Parse validation (simple check for "Supported: no")
            if "supported: no" in validation_result.lower():
                return Answer(
                    text=answer.text,
                    citations=answer.citations,
                    confidence=answer.confidence * 0.3,
                    refusal_reason="evidence_validation_failed",
                    trace=answer.trace,
                    metadata={
                        **answer.metadata,
                        "validation": "failed_llm",
                        "validation_result": validation_result
                    }
                )
            
            # Validation passed
            return Answer(
                text=answer.text,
                citations=answer.citations,
                confidence=answer.confidence,
                refusal_reason=answer.refusal_reason,
                trace=answer.trace,
                metadata={
                    **answer.metadata,
                    "validation": "passed_llm",
                    "validation_result": validation_result
                }
            )
            
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            # Fall back to rule-based
            return self._rule_based_validation(answer)
    
    def validate(
        self,
        query: str,
        answer: Answer,
        chunks: Optional[List[RetrievedChunk]] = None,
        **kwargs
    ) -> Answer:
        """
        Validate answer.
        
        Args:
            query: Original query (unused currently)
            answer: Answer to validate
            chunks: Context chunks (required for LLM validation)
            **kwargs: Additional parameters
            
        Returns:
            Validated answer
        """
        if self.use_llm:
            if not chunks:
                logger.warning("LLM validation requires chunks, falling back to rule-based")
                return self._rule_based_validation(answer)
            return self._llm_based_validation(answer, chunks)
        else:
            return self._rule_based_validation(answer)

