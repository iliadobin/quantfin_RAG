"""
Citation metrics for evaluating citation quality and evidence coverage.

These metrics evaluate:
- Whether citations actually support the claims
- Whether all claims have supporting evidence
- Citation precision and recall
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from knowledge.models import Citation, Answer


class CitationMetrics(BaseModel):
    """Container for citation metrics."""
    
    # Core metrics
    citation_precision: float = 0.0  # Fraction of citations that support claims
    citation_recall: float = 0.0     # Fraction of gold citations found
    evidence_coverage: float = 0.0   # Fraction of claims with evidence
    
    # Detailed stats
    total_predicted_citations: int = 0
    total_gold_citations: int = 0
    correct_citations: int = 0
    
    claims_with_evidence: int = 0
    claims_without_evidence: int = 0
    
    # Per-example details
    per_example_metrics: List[Dict[str, Any]] = Field(default_factory=list)


def extract_claims_from_text(text: str) -> List[str]:
    """
    Extract factual claims from answer text.
    
    Simple heuristic: split by sentence boundaries.
    In production, could use more sophisticated NLP.
    
    Args:
        text: Answer text
        
    Returns:
        List of claim strings
    """
    # Simple sentence splitting
    import re
    sentences = re.split(r'[.!?]+', text)
    claims = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    return claims


def citation_matches_gold(
    pred_citation: Citation,
    gold_citations: List[Citation],
    page_tolerance: int = 0
) -> bool:
    """
    Check if predicted citation matches any gold citation.
    
    Args:
        pred_citation: Predicted citation
        gold_citations: List of gold citations
        page_tolerance: Allow ±N pages for match
        
    Returns:
        True if match found
    """
    for gold in gold_citations:
        # Must be same document
        if pred_citation.doc_id != gold.doc_id:
            continue
        
        # Check page overlap with tolerance
        pred_start = pred_citation.page_span.start_page
        pred_end = pred_citation.page_span.end_page
        gold_start = gold.page_span.start_page
        gold_end = gold.page_span.end_page
        
        # Check if ranges overlap (with tolerance)
        if (pred_start - page_tolerance <= gold_end and 
            pred_end + page_tolerance >= gold_start):
            return True
    
    return False


def compute_citation_precision(
    predicted_citations: List[Citation],
    gold_citations: List[Citation],
    page_tolerance: int = 1
) -> float:
    """
    Compute citation precision: fraction of predicted citations that are correct.
    
    Args:
        predicted_citations: Citations from system
        gold_citations: Gold-standard citations
        page_tolerance: Allow ±N pages for match
        
    Returns:
        Precision score [0, 1]
    """
    if not predicted_citations:
        return 0.0
    
    correct = sum(
        1 for pred in predicted_citations
        if citation_matches_gold(pred, gold_citations, page_tolerance)
    )
    
    return correct / len(predicted_citations)


def compute_citation_recall(
    predicted_citations: List[Citation],
    gold_citations: List[Citation],
    page_tolerance: int = 1
) -> float:
    """
    Compute citation recall: fraction of gold citations found.
    
    Args:
        predicted_citations: Citations from system
        gold_citations: Gold-standard citations
        page_tolerance: Allow ±N pages for match
        
    Returns:
        Recall score [0, 1]
    """
    if not gold_citations:
        return 1.0  # No gold citations to find
    
    found = sum(
        1 for gold in gold_citations
        if citation_matches_gold(gold, predicted_citations, page_tolerance)
    )
    
    return found / len(gold_citations)


def compute_evidence_coverage(
    answer_text: str,
    citations: List[Citation],
    min_claims_threshold: int = 1
) -> float:
    """
    Compute evidence coverage: fraction of claims with supporting citations.
    
    This is a heuristic metric. Ideally would use LLM to map claims to citations,
    but we use simpler heuristics here for efficiency.
    
    Args:
        answer_text: Generated answer text
        citations: Citations provided with answer
        min_claims_threshold: Minimum number of claims to count
        
    Returns:
        Coverage score [0, 1]
    """
    claims = extract_claims_from_text(answer_text)
    
    if len(claims) < min_claims_threshold:
        # Too short to evaluate
        return 1.0 if citations else 0.0
    
    # Heuristic: if we have at least one citation per 2 claims, good coverage
    # In production, would do more sophisticated claim→citation matching
    expected_citations = max(1, len(claims) // 2)
    
    if len(citations) >= expected_citations:
        return 1.0
    else:
        return len(citations) / expected_citations


def compute_citation_metrics(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    page_tolerance: int = 1
) -> CitationMetrics:
    """
    Compute citation metrics across all examples.
    
    Args:
        predictions: List of predictions, each with:
            - example_id: str
            - answer: Answer object with citations
        ground_truth: List of ground truth, each with:
            - example_id: str
            - gold_citations: List[Citation]
            - gold_answer: str (optional, for claim counting)
            
    Returns:
        CitationMetrics with aggregate scores
    """
    # Build mapping from example_id to ground truth
    gt_map = {item['example_id']: item for item in ground_truth}
    
    # Accumulators
    precision_scores = []
    recall_scores = []
    coverage_scores = []
    
    total_pred_citations = 0
    total_gold_citations = 0
    total_correct_citations = 0
    
    total_claims_with_evidence = 0
    total_claims_without_evidence = 0
    
    per_example_results = []
    
    for pred in predictions:
        example_id = pred['example_id']
        answer: Answer = pred['answer']
        
        # Get ground truth
        gt = gt_map.get(example_id, {})
        gold_citations = gt.get('gold_citations', [])
        
        # Compute precision
        precision = compute_citation_precision(
            answer.citations, gold_citations, page_tolerance
        )
        precision_scores.append(precision)
        
        # Compute recall
        recall = compute_citation_recall(
            answer.citations, gold_citations, page_tolerance
        )
        recall_scores.append(recall)
        
        # Compute coverage
        coverage = compute_evidence_coverage(answer.text, answer.citations)
        coverage_scores.append(coverage)
        
        # Update totals
        total_pred_citations += len(answer.citations)
        total_gold_citations += len(gold_citations)
        
        correct = sum(
            1 for pred_cit in answer.citations
            if citation_matches_gold(pred_cit, gold_citations, page_tolerance)
        )
        total_correct_citations += correct
        
        # Count claims
        claims = extract_claims_from_text(answer.text)
        has_citations = len(answer.citations) > 0
        if has_citations:
            total_claims_with_evidence += len(claims)
        else:
            total_claims_without_evidence += len(claims)
        
        per_example_results.append({
            'example_id': example_id,
            'precision': precision,
            'recall': recall,
            'coverage': coverage,
            'num_predicted_citations': len(answer.citations),
            'num_gold_citations': len(gold_citations),
            'num_claims': len(claims)
        })
    
    # Compute averages
    def safe_mean(scores):
        return sum(scores) / len(scores) if scores else 0.0
    
    metrics = CitationMetrics(
        citation_precision=safe_mean(precision_scores),
        citation_recall=safe_mean(recall_scores),
        evidence_coverage=safe_mean(coverage_scores),
        total_predicted_citations=total_pred_citations,
        total_gold_citations=total_gold_citations,
        correct_citations=total_correct_citations,
        claims_with_evidence=total_claims_with_evidence,
        claims_without_evidence=total_claims_without_evidence,
        per_example_metrics=per_example_results
    )
    
    return metrics

