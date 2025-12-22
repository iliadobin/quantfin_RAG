"""
Cross-encoder reranker using sentence transformers.

Uses local models for cost efficiency (no API calls).
"""
import logging
from typing import List, Optional
from sentence_transformers import CrossEncoder

from knowledge.models import RetrievedChunk


logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranker for relevance scoring.
    
    Uses local cross-encoder model to rerank retrieved chunks.
    Cost-efficient (no API calls).
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu"
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Cross-encoder model name
            device: Device for inference ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name, device=device)
    
    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 5,
        **kwargs
    ) -> List[RetrievedChunk]:
        """
        Rerank chunks using cross-encoder.
        
        Args:
            query: User query
            chunks: Retrieved chunks to rerank
            top_k: Number of top chunks to return
            **kwargs: Additional parameters
            
        Returns:
            Reranked chunks with updated scores
        """
        if not chunks:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [(query, chunk.chunk.text) for chunk in chunks]
        
        # Compute relevance scores
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Create reranked results
        reranked = []
        for chunk, score in zip(chunks, scores):
            reranked.append(RetrievedChunk(
                chunk=chunk.chunk,
                score=float(score),
                retriever_tag=f"{chunk.retriever_tag}+rerank",
                metadata={
                    **chunk.metadata,
                    "original_score": chunk.score,
                    "rerank_score": float(score)
                }
            ))
        
        # Sort by rerank score and take top-k
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]
    
    def batch_rerank(
        self,
        queries: List[str],
        chunks_list: List[List[RetrievedChunk]],
        top_k: int = 5,
        **kwargs
    ) -> List[List[RetrievedChunk]]:
        """
        Batch rerank for multiple queries.
        
        Args:
            queries: List of queries
            chunks_list: List of chunk lists (one per query)
            top_k: Number of results per query
            **kwargs: Additional parameters
            
        Returns:
            List of reranked results
        """
        results = []
        for query, chunks in zip(queries, chunks_list):
            reranked = self.rerank(query, chunks, top_k=top_k, **kwargs)
            results.append(reranked)
        return results

