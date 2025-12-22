"""
Metrics for benchmark evaluation.

Includes:
- Retrieval metrics (Recall@k, nDCG@k, MRR)
- Citation metrics (precision, coverage)
- Hallucination/refusal metrics
- LLM-judge for answer quality
"""
from .retrieval import RetrievalMetrics, compute_retrieval_metrics
from .citation import CitationMetrics, compute_citation_metrics
from .hallucination import HallucinationMetrics, compute_hallucination_metrics
from .llm_judge import LLMJudge, judge_answer_quality

__all__ = [
    'RetrievalMetrics',
    'compute_retrieval_metrics',
    'CitationMetrics',
    'compute_citation_metrics',
    'HallucinationMetrics',
    'compute_hallucination_metrics',
    'LLMJudge',
    'judge_answer_quality',
]

