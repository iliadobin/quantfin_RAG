"""
Answer generator with automatic citation extraction and mapping.
"""
import re
import logging
from typing import List, Optional, Dict, Any

from knowledge.models import RetrievedChunk, Answer, Citation, PageSpan
from llm.deepseek_client import DeepSeekClient
from rag.generators import prompts


logger = logging.getLogger(__name__)


class CitationGenerator:
    """
    Generator that produces answers with citations.
    
    Uses LLM to generate answer with inline citations [1], [2], etc.,
    then maps them to actual source chunks.
    """
    
    def __init__(
        self,
        llm_client: DeepSeekClient,
        model: Optional[str] = None,
        max_context_chunks: int = 10,
        temperature: float = 0.0,
        max_tokens: int = 2048
    ):
        """
        Initialize citation generator.
        
        Args:
            llm_client: DeepSeek client instance
            model: Model to use (default: client's default)
            max_context_chunks: Maximum chunks to include in context
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
        """
        self.llm_client = llm_client
        self.model = model
        self.max_context_chunks = max_context_chunks
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def _extract_citations(self, text: str) -> List[int]:
        """
        Extract citation numbers from answer text.
        
        Args:
            text: Answer text with citations like [1], [2]
            
        Returns:
            List of unique citation numbers
        """
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, text)
        return sorted(set(int(m) for m in matches))
    
    def _build_citations(
        self,
        citation_numbers: List[int],
        chunks: List[RetrievedChunk]
    ) -> List[Citation]:
        """
        Build Citation objects from citation numbers and chunks.
        
        Args:
            citation_numbers: Citation numbers from answer
            chunks: Retrieved chunks used in context
            
        Returns:
            List of Citation objects
        """
        citations = []
        
        for num in citation_numbers:
            idx = num - 1  # Citations are 1-indexed
            if 0 <= idx < len(chunks):
                retrieved = chunks[idx]
                chunk = retrieved.chunk
                
                # Extract quote (first 200 chars of chunk)
                quote = chunk.text[:200].strip()
                if len(chunk.text) > 200:
                    quote += "..."
                
                citation = Citation(
                    doc_id=chunk.doc_id,
                    page_span=chunk.page_span,
                    quote=quote,
                    score=retrieved.score,
                    retriever_tag=retrieved.retriever_tag,
                    chunk_id=chunk.id
                )
                citations.append(citation)
        
        return citations
    
    def _detect_refusal(self, text: str) -> Optional[str]:
        """
        Detect if answer is a refusal.
        
        Args:
            text: Generated answer text
            
        Returns:
            Refusal reason if detected, None otherwise
        """
        refusal_phrases = [
            "cannot answer",
            "don't have enough information",
            "insufficient information",
            "not found in",
            "no information about",
            "unable to answer"
        ]
        
        text_lower = text.lower()
        for phrase in refusal_phrases:
            if phrase in text_lower:
                return "insufficient_context"
        
        return None
    
    def _compute_confidence(
        self,
        answer_text: str,
        citations: List[Citation],
        chunks: List[RetrievedChunk]
    ) -> float:
        """
        Compute confidence score for answer.
        
        Simple heuristic based on:
        - Number of citations
        - Average retrieval score of cited chunks
        - Answer length (very short = low confidence)
        
        Args:
            answer_text: Generated answer
            citations: Extracted citations
            chunks: Retrieved chunks
            
        Returns:
            Confidence score 0-1
        """
        if not answer_text or len(answer_text) < 20:
            return 0.0
        
        if not citations:
            return 0.3  # No citations = low confidence
        
        # Average score of cited chunks
        avg_score = sum(c.score for c in citations) / len(citations)
        
        # Normalize (assuming scores are roughly 0-1 for dense, 0-10+ for BM25)
        # Use sigmoid-like scaling
        confidence = min(1.0, avg_score / 2.0) if avg_score < 2.0 else min(1.0, avg_score / 10.0)
        
        # Boost if multiple citations
        if len(citations) >= 2:
            confidence = min(1.0, confidence * 1.2)
        
        return confidence
    
    def generate(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        **kwargs
    ) -> Answer:
        """
        Generate answer with citations.
        
        Args:
            query: User query
            chunks: Retrieved chunks
            **kwargs: Additional generation parameters
            
        Returns:
            Answer with citations and metadata
        """
        if not chunks:
            return Answer(
                text="No relevant information found in the corpus.",
                citations=[],
                confidence=0.0,
                refusal_reason="no_context",
                metadata={"reason": "empty_retrieval"}
            )
        
        # Build prompt
        user_prompt = prompts.build_qa_prompt(
            query,
            chunks,
            max_chunks=self.max_context_chunks
        )
        
        # Generate answer
        try:
            response = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": prompts.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens)
            )
            
            answer_text = response["content"]
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return Answer(
                text="Error generating answer.",
                citations=[],
                confidence=0.0,
                refusal_reason="generation_error",
                metadata={"error": str(e)}
            )
        
        # Extract citations
        citation_numbers = self._extract_citations(answer_text)
        citations = self._build_citations(citation_numbers, chunks)
        
        # Check for refusal
        refusal_reason = self._detect_refusal(answer_text)
        
        # Compute confidence
        confidence = self._compute_confidence(answer_text, citations, chunks)
        if refusal_reason:
            confidence = 0.0
        
        return Answer(
            text=answer_text,
            citations=citations,
            confidence=confidence,
            refusal_reason=refusal_reason,
            metadata={
                "model": response.get("model"),
                "usage": response.get("usage"),
                "num_context_chunks": len(chunks),
                "num_citations": len(citations)
            }
        )
    
    def batch_generate(
        self,
        queries: List[str],
        chunks_list: List[List[RetrievedChunk]],
        **kwargs
    ) -> List[Answer]:
        """
        Batch generate answers.
        
        Args:
            queries: List of queries
            chunks_list: List of chunk lists (one per query)
            **kwargs: Generation parameters
            
        Returns:
            List of answers
        """
        answers = []
        for query, chunks in zip(queries, chunks_list):
            answer = self.generate(query, chunks, **kwargs)
            answers.append(answer)
        return answers

