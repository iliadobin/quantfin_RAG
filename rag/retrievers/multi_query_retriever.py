"""
Multi-query retriever with template-based query expansion.

Uses rule-based expansion for cost efficiency (no LLM calls).
"""
import logging
from typing import List, Dict, Optional, Union
from collections import defaultdict

from knowledge.models import RetrievedChunk
from rag.retrievers.dense_retriever import DenseRetriever
from rag.retrievers.bm25_retriever import BM25Retriever
from rag.retrievers.hybrid_retriever import HybridRetriever


logger = logging.getLogger(__name__)


# Template-based query expansion for derivatives/pricing domain
QUERY_EXPANSION_TEMPLATES = {
    "pricing": [
        "{query}",
        "How to price {query}",
        "Pricing formula for {query}",
        "Valuation of {query}",
    ],
    "greeks": [
        "{query}",
        "Greeks for {query}",
        "Delta and gamma of {query}",
        "Sensitivity analysis {query}",
    ],
    "assumptions": [
        "{query}",
        "Assumptions for {query}",
        "Requirements for {query}",
        "Conditions for {query}",
    ],
    "methods": [
        "{query}",
        "Methods for {query}",
        "Approaches to {query}",
        "Techniques for {query}",
    ],
}


class MultiQueryRetriever:
    """
    Multi-query retriever with template-based expansion.
    
    Expands query using domain-specific templates (cheap, no LLM),
    then fuses results from multiple queries.
    """
    
    def __init__(
        self,
        base_retriever: Union[DenseRetriever, BM25Retriever, HybridRetriever],
        expansion_strategy: str = "pricing",
        max_queries: int = 3,
        fusion_method: str = "rrf"
    ):
        """
        Initialize multi-query retriever.
        
        Args:
            base_retriever: Base retriever (dense, BM25, or hybrid)
            expansion_strategy: Template set to use
            max_queries: Maximum number of query variants
            fusion_method: How to fuse results ('rrf' or 'max_score')
        """
        self.base_retriever = base_retriever
        self.expansion_strategy = expansion_strategy
        self.max_queries = max_queries
        self.fusion_method = fusion_method
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Expand query using templates.
        
        Args:
            query: Original query
            
        Returns:
            List of expanded queries (including original)
        """
        templates = QUERY_EXPANSION_TEMPLATES.get(
            self.expansion_strategy,
            ["{query}"]
        )
        
        # Generate variants
        variants = []
        for template in templates[:self.max_queries]:
            expanded = template.format(query=query)
            if expanded not in variants:
                variants.append(expanded)
        
        return variants
    
    def _fuse_results_rrf(
        self,
        all_results: List[List[RetrievedChunk]],
        top_k: int,
        rrf_k: int = 60
    ) -> List[RetrievedChunk]:
        """
        Fuse results using reciprocal rank fusion.
        
        Args:
            all_results: List of result lists (one per query variant)
            top_k: Number of final results
            rrf_k: RRF constant
            
        Returns:
            Fused results
        """
        fusion_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, RetrievedChunk] = {}
        
        for results in all_results:
            for rank, retrieved in enumerate(results, start=1):
                chunk_id = retrieved.chunk.id
                rrf_score = 1.0 / (rrf_k + rank)
                fusion_scores[chunk_id] += rrf_score
                
                if chunk_id not in chunk_map:
                    chunk_map[chunk_id] = retrieved
        
        # Sort by fused score
        sorted_ids = sorted(
            fusion_scores.keys(),
            key=lambda cid: fusion_scores[cid],
            reverse=True
        )[:top_k]
        
        # Build results
        results = []
        for chunk_id in sorted_ids:
            retrieved = chunk_map[chunk_id]
            results.append(RetrievedChunk(
                chunk=retrieved.chunk,
                score=fusion_scores[chunk_id],
                retriever_tag="multi_query",
                metadata={
                    "fusion_score": fusion_scores[chunk_id],
                    "original_retriever": retrieved.retriever_tag
                }
            ))
        
        return results
    
    def _fuse_results_max_score(
        self,
        all_results: List[List[RetrievedChunk]],
        top_k: int
    ) -> List[RetrievedChunk]:
        """
        Fuse by taking max score per chunk.
        
        Args:
            all_results: List of result lists
            top_k: Number of final results
            
        Returns:
            Fused results
        """
        max_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievedChunk] = {}
        
        for results in all_results:
            for retrieved in results:
                chunk_id = retrieved.chunk.id
                score = retrieved.score
                
                if chunk_id not in max_scores or score > max_scores[chunk_id]:
                    max_scores[chunk_id] = score
                    chunk_map[chunk_id] = retrieved
        
        # Sort by max score
        sorted_ids = sorted(
            max_scores.keys(),
            key=lambda cid: max_scores[cid],
            reverse=True
        )[:top_k]
        
        # Build results
        results = []
        for chunk_id in sorted_ids:
            retrieved = chunk_map[chunk_id]
            results.append(RetrievedChunk(
                chunk=retrieved.chunk,
                score=max_scores[chunk_id],
                retriever_tag="multi_query",
                metadata={
                    "max_score": max_scores[chunk_id],
                    "original_retriever": retrieved.retriever_tag
                }
            ))
        
        return results
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        per_query_k: Optional[int] = None,
        **kwargs
    ) -> List[RetrievedChunk]:
        """
        Retrieve using multi-query expansion and fusion.
        
        Args:
            query: Original user query
            top_k: Number of final results
            per_query_k: Number of results per expanded query (default: top_k)
            **kwargs: Additional parameters
            
        Returns:
            Fused retrieval results
        """
        # Expand query
        expanded_queries = self._expand_query(query)
        logger.debug(f"Expanded query into {len(expanded_queries)} variants")
        
        # Retrieve for each variant
        per_k = per_query_k or top_k
        all_results = self.base_retriever.batch_retrieve(
            expanded_queries,
            top_k=per_k,
            **kwargs
        )
        
        # Fuse
        if self.fusion_method == "rrf":
            return self._fuse_results_rrf(all_results, top_k)
        else:
            return self._fuse_results_max_score(all_results, top_k)
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
        **kwargs
    ) -> List[List[RetrievedChunk]]:
        """
        Batch multi-query retrieval.
        
        Args:
            queries: List of queries
            top_k: Number of results per query
            **kwargs: Additional parameters
            
        Returns:
            List of fused results per query
        """
        return [self.retrieve(q, top_k=top_k, **kwargs) for q in queries]

