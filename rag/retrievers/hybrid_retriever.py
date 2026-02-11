"""
Hybrid retriever combining BM25 and dense retrieval with reciprocal rank fusion.
"""
import logging
from typing import List, Dict, Optional
from collections import defaultdict

from knowledge.models import RetrievedChunk
from rag.retrievers.bm25_retriever import BM25Retriever
from rag.retrievers.dense_retriever import DenseRetriever


logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever using reciprocal rank fusion (RRF).
    
    Combines BM25 (lexical) and dense (semantic) retrieval for better coverage.
    RRF formula: score = sum(1 / (k + rank_i)) for each retriever
    """
    
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            bm25_retriever: BM25 retriever instance
            dense_retriever: Dense retriever instance
            bm25_weight: Weight for BM25 scores (0-1)
            dense_weight: Weight for dense scores (0-1)
            rrf_k: RRF constant (typically 60)
        """
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k
    
    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[RetrievedChunk],
        dense_results: List[RetrievedChunk],
        top_k: int
    ) -> List[RetrievedChunk]:
        """
        Combine results using reciprocal rank fusion.
        
        Args:
            bm25_results: Results from BM25
            dense_results: Results from dense retriever
            top_k: Number of results to return
            
        Returns:
            Fused and sorted results
        """
        # Compute RRF scores
        fusion_scores: Dict[str, float] = defaultdict(float)
        bm25_contrib: Dict[str, float] = defaultdict(float)
        dense_contrib: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, RetrievedChunk] = {}
        
        # Add BM25 scores
        for rank, retrieved in enumerate(bm25_results, start=1):
            chunk_id = retrieved.chunk.id
            rrf_score = self.bm25_weight / (self.rrf_k + rank)
            fusion_scores[chunk_id] += rrf_score
            bm25_contrib[chunk_id] += rrf_score
            
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = retrieved
        
        # Add dense scores
        for rank, retrieved in enumerate(dense_results, start=1):
            chunk_id = retrieved.chunk.id
            rrf_score = self.dense_weight / (self.rrf_k + rank)
            fusion_scores[chunk_id] += rrf_score
            dense_contrib[chunk_id] += rrf_score
            
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = retrieved
        
        # Sort by fused score
        sorted_ids = sorted(
            fusion_scores.keys(),
            key=lambda cid: fusion_scores[cid],
            reverse=True
        )[:top_k]
        
        # Build final results
        results = []
        for chunk_id in sorted_ids:
            retrieved = chunk_map[chunk_id]
            # Create new RetrievedChunk with hybrid score
            results.append(RetrievedChunk(
                chunk=retrieved.chunk,
                score=fusion_scores[chunk_id],
                retriever_tag="hybrid",
                metadata={
                    "bm25_score": bm25_contrib.get(chunk_id, 0.0),
                    "dense_score": dense_contrib.get(chunk_id, 0.0),
                    "original_retriever": retrieved.retriever_tag
                }
            ))
        
        return results
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        bm25_top_k: Optional[int] = None,
        dense_top_k: Optional[int] = None,
        **kwargs
    ) -> List[RetrievedChunk]:
        """
        Retrieve using hybrid fusion.
        
        Args:
            query: User query
            top_k: Number of final results
            bm25_top_k: Number of BM25 results (default: 2*top_k)
            dense_top_k: Number of dense results (default: 2*top_k)
            **kwargs: Additional parameters
            
        Returns:
            Fused retrieval results
        """
        # Retrieve from both
        bm25_k = bm25_top_k or (2 * top_k)
        dense_k = dense_top_k or (2 * top_k)
        
        bm25_results = self.bm25_retriever.retrieve(query, top_k=bm25_k)
        dense_results = self.dense_retriever.retrieve(query, top_k=dense_k)
        
        logger.debug(f"Hybrid retrieval: BM25={len(bm25_results)}, Dense={len(dense_results)}")
        
        # Fuse
        return self._reciprocal_rank_fusion(bm25_results, dense_results, top_k)
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
        **kwargs
    ) -> List[List[RetrievedChunk]]:
        """
        Batch hybrid retrieval.
        
        Args:
            queries: List of queries
            top_k: Number of results per query
            **kwargs: Additional parameters
            
        Returns:
            List of fused results per query
        """
        bm25_k = kwargs.get('bm25_top_k', 2 * top_k)
        dense_k = kwargs.get('dense_top_k', 2 * top_k)
        
        # Batch retrieve from both
        bm25_results = self.bm25_retriever.batch_retrieve(queries, top_k=bm25_k)
        dense_results = self.dense_retriever.batch_retrieve(queries, top_k=dense_k)
        
        # Fuse each query's results
        all_results = []
        for bm25_res, dense_res in zip(bm25_results, dense_results):
            fused = self._reciprocal_rank_fusion(bm25_res, dense_res, top_k)
            all_results.append(fused)
        
        return all_results

