"""
Hallucination and refusal metrics for DS3 (UnanswerableAndTraps).

Evaluates:
- Whether system correctly refuses unanswerable questions
- Whether system avoids hallucinated answers
- Robustness to traps and ambiguous queries
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from knowledge.models import Answer


class HallucinationMetrics(BaseModel):
    """Container for hallucination/refusal metrics."""
    
    # Core metrics
    refusal_accuracy: float = 0.0      # Correctly refused unanswerable questions
    false_refusal_rate: float = 0.0    # Incorrectly refused answerable questions
    hallucination_rate: float = 0.0    # Answered without evidence when should refuse
    
    # Detailed stats
    total_unanswerable: int = 0
    correct_refusals: int = 0
    incorrect_answers: int = 0  # Answered when should refuse
    
    total_answerable: int = 0
    correct_answers: int = 0
    false_refusals: int = 0
    
    # Trap-specific metrics
    trap_resistance_by_type: Dict[str, float] = Field(default_factory=dict)
    
    # Per-example details
    per_example_metrics: List[Dict[str, Any]] = Field(default_factory=list)


def is_valid_refusal(answer: Answer) -> bool:
    """
    Check if answer is a valid refusal.
    
    Valid refusal: has refusal_reason and/or low confidence and/or no citations.
    
    Args:
        answer: Answer object
        
    Returns:
        True if this is a refusal
    """
    # Explicit refusal reason
    if answer.refusal_reason:
        return True
    
    # Very low confidence with no citations
    if answer.confidence < 0.3 and not answer.citations:
        return True
    
    # Answer text contains refusal patterns
    refusal_patterns = [
        "I cannot answer",
        "I don't have information",
        "This is outside",
        "cannot be answered",
        "insufficient information",
        "not available in",
        "unclear",
        "ambiguous"
    ]
    
    for pattern in refusal_patterns:
        if pattern.lower() in answer.text.lower():
            return True
    
    return False


def is_hallucinated_answer(answer: Answer, require_citations: bool = True) -> bool:
    """
    Check if answer appears to be hallucinated.
    
    Hallucination indicators:
    - Has substantive answer but no citations
    - High confidence but no evidence
    
    Args:
        answer: Answer object
        require_citations: Whether citations are required
        
    Returns:
        True if likely hallucinated
    """
    # If it's a refusal, not a hallucination
    if is_valid_refusal(answer):
        return False
    
    # If no citations but substantive answer, likely hallucination
    if require_citations and not answer.citations and len(answer.text) > 50:
        return True
    
    return False


def compute_refusal_accuracy(
    predictions: List[Answer],
    expected_refusals: List[bool]
) -> float:
    """
    Compute refusal accuracy for unanswerable questions.
    
    Args:
        predictions: List of Answer objects
        expected_refusals: List of bools (True = should refuse)
        
    Returns:
        Accuracy [0, 1]
    """
    if not predictions:
        return 0.0
    
    correct = 0
    for pred, should_refuse in zip(predictions, expected_refusals):
        did_refuse = is_valid_refusal(pred)
        if did_refuse == should_refuse:
            correct += 1
    
    return correct / len(predictions)


def compute_hallucination_rate(
    predictions: List[Answer],
    expected_refusals: List[bool]
) -> float:
    """
    Compute hallucination rate: fraction of unanswerable questions
    where system provided an answer without evidence.
    
    Args:
        predictions: List of Answer objects
        expected_refusals: List of bools (True = should refuse)
        
    Returns:
        Hallucination rate [0, 1]
    """
    unanswerable_preds = [
        pred for pred, should_refuse in zip(predictions, expected_refusals)
        if should_refuse
    ]
    
    if not unanswerable_preds:
        return 0.0
    
    hallucinated = sum(
        1 for pred in unanswerable_preds
        if is_hallucinated_answer(pred)
    )
    
    return hallucinated / len(unanswerable_preds)


def compute_hallucination_metrics(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]]
) -> HallucinationMetrics:
    """
    Compute hallucination/refusal metrics across all examples.
    
    Args:
        predictions: List of predictions, each with:
            - example_id: str
            - answer: Answer object
        ground_truth: List of ground truth, each with:
            - example_id: str
            - should_refuse: bool
            - trap_type: str (optional)
            
    Returns:
        HallucinationMetrics with aggregate scores
    """
    # Build mapping from example_id to ground truth
    gt_map = {item['example_id']: item for item in ground_truth}
    
    # Separate unanswerable and answerable
    unanswerable_results = []
    answerable_results = []
    
    # Track trap resistance by type
    trap_stats: Dict[str, Dict[str, int]] = {}
    
    per_example_results = []
    
    for pred in predictions:
        example_id = pred['example_id']
        answer: Answer = pred['answer']
        
        # Get ground truth
        gt = gt_map.get(example_id, {})
        should_refuse = gt.get('should_refuse', False)
        trap_type = gt.get('trap_type', 'unknown')
        
        did_refuse = is_valid_refusal(answer)
        is_hallucinated = is_hallucinated_answer(answer)
        
        # Record result
        result = {
            'example_id': example_id,
            'should_refuse': should_refuse,
            'did_refuse': did_refuse,
            'is_hallucinated': is_hallucinated,
            'trap_type': trap_type,
            'correct': did_refuse == should_refuse
        }
        per_example_results.append(result)
        
        # Update trap statistics
        if trap_type not in trap_stats:
            trap_stats[trap_type] = {'total': 0, 'correct': 0}
        trap_stats[trap_type]['total'] += 1
        if result['correct']:
            trap_stats[trap_type]['correct'] += 1
        
        # Categorize
        if should_refuse:
            unanswerable_results.append(result)
        else:
            answerable_results.append(result)
    
    # Compute metrics
    total_unanswerable = len(unanswerable_results)
    correct_refusals = sum(1 for r in unanswerable_results if r['did_refuse'])
    incorrect_answers = sum(1 for r in unanswerable_results if not r['did_refuse'])
    
    total_answerable = len(answerable_results)
    correct_answers = sum(1 for r in answerable_results if not r['did_refuse'])
    false_refusals = sum(1 for r in answerable_results if r['did_refuse'])
    
    refusal_accuracy = correct_refusals / total_unanswerable if total_unanswerable > 0 else 0.0
    false_refusal_rate = false_refusals / total_answerable if total_answerable > 0 else 0.0
    
    # Hallucination rate: of the unanswerable questions, how many were answered without evidence
    hallucinated_count = sum(1 for r in unanswerable_results if r['is_hallucinated'])
    hallucination_rate = hallucinated_count / total_unanswerable if total_unanswerable > 0 else 0.0
    
    # Trap resistance by type
    trap_resistance = {
        trap_type: stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        for trap_type, stats in trap_stats.items()
    }
    
    metrics = HallucinationMetrics(
        refusal_accuracy=refusal_accuracy,
        false_refusal_rate=false_refusal_rate,
        hallucination_rate=hallucination_rate,
        total_unanswerable=total_unanswerable,
        correct_refusals=correct_refusals,
        incorrect_answers=incorrect_answers,
        total_answerable=total_answerable,
        correct_answers=correct_answers,
        false_refusals=false_refusals,
        trap_resistance_by_type=trap_resistance,
        per_example_metrics=per_example_results
    )
    
    return metrics

