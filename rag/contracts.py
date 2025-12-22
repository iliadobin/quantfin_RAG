"""
Contracts (protocols) for RAG components.

All RAG pipelines implement these interfaces for consistency and testability.
"""
from typing import Protocol, List, Dict, Any, Optional
from knowledge.models import RetrievedChunk, Citation, Answer, RetrievalTrace


class Retriever(Protocol):
    """Protocol for retrievers."""
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[RetrievedChunk]:
        """
        Retrieve top-k chunks for a query.
        
        Args:
            query: User query string
            top_k: Number of chunks to retrieve
            **kwargs: Additional retriever-specific parameters
            
        Returns:
            List of retrieved chunks with scores
        """
        ...
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
        **kwargs
    ) -> List[List[RetrievedChunk]]:
        """
        Retrieve for multiple queries (batch mode for efficiency).
        
        Args:
            queries: List of query strings
            top_k: Number of chunks per query
            **kwargs: Additional parameters
            
        Returns:
            List of retrieval results, one per query
        """
        ...


class Reranker(Protocol):
    """Protocol for rerankers."""
    
    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 5,
        **kwargs
    ) -> List[RetrievedChunk]:
        """
        Rerank chunks based on query-chunk relevance.
        
        Args:
            query: User query
            chunks: Chunks to rerank
            top_k: Number of top chunks to return
            **kwargs: Additional reranker-specific parameters
            
        Returns:
            Reranked chunks with updated scores
        """
        ...


class Generator(Protocol):
    """Protocol for answer generators."""
    
    def generate(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        **kwargs
    ) -> Answer:
        """
        Generate answer from query and retrieved chunks.
        
        Args:
            query: User query
            chunks: Retrieved and possibly reranked chunks
            **kwargs: Generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            Answer with citations and metadata
        """
        ...


class Guardrail(Protocol):
    """Protocol for guardrails (validation, safety checks)."""
    
    def validate(
        self,
        query: str,
        answer: Answer,
        **kwargs
    ) -> Answer:
        """
        Validate and potentially modify/reject answer.
        
        Args:
            query: Original query
            answer: Generated answer
            **kwargs: Validation parameters
            
        Returns:
            Validated answer (may be modified or refused)
        """
        ...


class Pipeline(Protocol):
    """Protocol for complete RAG pipelines."""
    
    def run(
        self,
        query: str,
        corpus_profile: str = "public",
        **kwargs
    ) -> Answer:
        """
        Run complete RAG pipeline.
        
        Args:
            query: User query
            corpus_profile: Corpus to use (default: "public")
            **kwargs: Pipeline-specific parameters
            
        Returns:
            Answer with full trace
        """
        ...
    
    @property
    def name(self) -> str:
        """Pipeline name for identification."""
        ...
    
    @property
    def version(self) -> str:
        """Pipeline version string."""
        ...

