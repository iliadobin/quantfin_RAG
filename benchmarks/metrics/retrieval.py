"""
Retrieval metrics for DS2 (RetrievalQrels).

Implements standard IR metrics without requiring LLM calls.
"""
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import math


class RetrievalMetrics(BaseModel):
    """Container for retrieval metrics."""
    
    # Core metrics
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    ndcg_at_20: float = 0.0
    
    mrr: float = 0.0  # Mean Reciprocal Rank
    
    # Additional stats
    mean_average_precision: float = 0.0
    total_queries: int = 0
    total_relevant_docs: int = 0
    
    # Per-query details (optional)
    per_query_metrics: List[Dict[str, Any]] = Field(default_factory=list)


def compute_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float:
    """
    Compute Recall@k.
    
    Args:
        retrieved_ids: List of retrieved chunk IDs (in rank order)
        relevant_ids: List of gold-standard relevant chunk IDs
        k: Cutoff position
        
    Returns:
        Recall@k score [0, 1]
    """
    if not relevant_ids:
        return 0.0
    
    retrieved_at_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    hits = len(retrieved_at_k & relevant_set)
    return hits / len(relevant_set)


def compute_dcg_at_k(
    retrieved_ids: List[str],
    relevance_map: Dict[str, int],
    k: int
) -> float:
    """
    Compute DCG@k (Discounted Cumulative Gain).
    
    Args:
        retrieved_ids: List of retrieved chunk IDs (in rank order)
        relevance_map: Dict mapping chunk_id -> relevance score (0, 1, 2)
        k: Cutoff position
        
    Returns:
        DCG@k score
    """
    dcg = 0.0
    for i, chunk_id in enumerate(retrieved_ids[:k], 1):
        rel = relevance_map.get(chunk_id, 0)
        # DCG formula: sum of rel / log2(rank + 1)
        dcg += rel / math.log2(i + 1)
    return dcg


def compute_ndcg_at_k(
    retrieved_ids: List[str],
    relevance_map: Dict[str, int],
    k: int
) -> float:
    """
    Compute nDCG@k (Normalized DCG).
    
    Args:
        retrieved_ids: List of retrieved chunk IDs (in rank order)
        relevance_map: Dict mapping chunk_id -> relevance score
        k: Cutoff position
        
    Returns:
        nDCG@k score [0, 1]
    """
    # Compute DCG@k for retrieved results
    dcg = compute_dcg_at_k(retrieved_ids, relevance_map, k)
    
    # Compute ideal DCG@k (sort by relevance descending)
    sorted_relevances = sorted(relevance_map.values(), reverse=True)
    ideal_ids = [f"ideal_{i}" for i in range(len(sorted_relevances))]
    ideal_relevance_map = {ideal_ids[i]: rel for i, rel in enumerate(sorted_relevances)}
    idcg = compute_dcg_at_k(ideal_ids, ideal_relevance_map, k)
    
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg


def compute_reciprocal_rank(
    retrieved_ids: List[str],
    relevant_ids: List[str]
) -> float:
    """
    Compute Reciprocal Rank (RR).
    
    Args:
        retrieved_ids: List of retrieved chunk IDs (in rank order)
        relevant_ids: List of gold-standard relevant chunk IDs
        
    Returns:
        RR score (1/rank of first relevant doc, or 0 if none found)
    """
    relevant_set = set(relevant_ids)
    
    for i, chunk_id in enumerate(retrieved_ids, 1):
        if chunk_id in relevant_set:
            return 1.0 / i
    
    return 0.0


def compute_average_precision(
    retrieved_ids: List[str],
    relevant_ids: List[str]
) -> float:
    """
    Compute Average Precision (AP).
    
    Args:
        retrieved_ids: List of retrieved chunk IDs (in rank order)
        relevant_ids: List of gold-standard relevant chunk IDs
        
    Returns:
        AP score [0, 1]
    """
    if not relevant_ids:
        return 0.0
    
    relevant_set = set(relevant_ids)
    num_relevant = len(relevant_set)
    
    num_hits = 0
    sum_precisions = 0.0
    
    for i, chunk_id in enumerate(retrieved_ids, 1):
        if chunk_id in relevant_set:
            num_hits += 1
            precision_at_i = num_hits / i
            sum_precisions += precision_at_i
    
    if num_hits == 0:
        return 0.0
    
    return sum_precisions / num_relevant


def compute_retrieval_metrics(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    k_values: List[int] = [5, 10, 20]
) -> RetrievalMetrics:
    """
    Compute retrieval metrics across all queries.
    
    Args:
        predictions: List of predictions, each with:
            - query_id: str
            - retrieved_ids: List[str] (chunk IDs in rank order)
        ground_truth: List of ground truth, each with:
            - query_id: str
            - qrels: Dict[str, int] (chunk_id -> relevance score)
            
    Returns:
        RetrievalMetrics with aggregate scores
    """
    # Build mapping from query_id to ground truth
    gt_map = {item['query_id']: item['qrels'] for item in ground_truth}
    
    # Accumulators for metrics
    recall_scores = {k: [] for k in k_values}
    ndcg_scores = {k: [] for k in k_values}
    rr_scores = []
    ap_scores = []
    
    per_query_results = []
    total_relevant = 0
    
    for pred in predictions:
        query_id = pred['query_id']
        retrieved_ids = pred['retrieved_ids']
        
        # Get ground truth qrels
        qrels = gt_map.get(query_id, {})
        relevant_ids = [cid for cid, rel in qrels.items() if rel > 0]
        total_relevant += len(relevant_ids)
        
        # Compute metrics for this query
        query_metrics = {'query_id': query_id}
        
        # Recall@k
        for k in k_values:
            recall = compute_recall_at_k(retrieved_ids, relevant_ids, k)
            recall_scores[k].append(recall)
            query_metrics[f'recall_at_{k}'] = recall
        
        # nDCG@k
        for k in k_values:
            ndcg = compute_ndcg_at_k(retrieved_ids, qrels, k)
            ndcg_scores[k].append(ndcg)
            query_metrics[f'ndcg_at_{k}'] = ndcg
        
        # RR
        rr = compute_reciprocal_rank(retrieved_ids, relevant_ids)
        rr_scores.append(rr)
        query_metrics['rr'] = rr
        
        # AP
        ap = compute_average_precision(retrieved_ids, relevant_ids)
        ap_scores.append(ap)
        query_metrics['ap'] = ap
        
        per_query_results.append(query_metrics)
    
    # Compute averages
    num_queries = len(predictions)
    
    def safe_mean(scores):
        return sum(scores) / len(scores) if scores else 0.0
    
    metrics = RetrievalMetrics(
        recall_at_5=safe_mean(recall_scores.get(5, [])),
        recall_at_10=safe_mean(recall_scores.get(10, [])),
        recall_at_20=safe_mean(recall_scores.get(20, [])),
        ndcg_at_5=safe_mean(ndcg_scores.get(5, [])),
        ndcg_at_10=safe_mean(ndcg_scores.get(10, [])),
        ndcg_at_20=safe_mean(ndcg_scores.get(20, [])),
        mrr=safe_mean(rr_scores),
        mean_average_precision=safe_mean(ap_scores),
        total_queries=num_queries,
        total_relevant_docs=total_relevant,
        per_query_metrics=per_query_results
    )
    
    return metrics

