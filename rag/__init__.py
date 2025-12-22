"""
RAG (Retrieval-Augmented Generation) system for quantitative finance QA.

This package provides:
- Multiple retrieval strategies (dense, BM25, hybrid, multi-query)
- Reranking with cross-encoders
- Answer generation with citations
- Evidence validation and guardrails
- 5 complete RAG pipelines (v1-v5)
"""

from rag.contracts import Retriever, Reranker, Generator, Guardrail, Pipeline
from rag.pipelines import (
    RAGv1Dense,
    RAGv2Hybrid,
    RAGv3MultiQuery,
    RAGv4ParentChild,
    RAGv5Evidence
)

__version__ = "1.0.0"

__all__ = [
    # Contracts
    'Retriever',
    'Reranker',
    'Generator',
    'Guardrail',
    'Pipeline',
    
    # Pipelines
    'RAGv1Dense',
    'RAGv2Hybrid',
    'RAGv3MultiQuery',
    'RAGv4ParentChild',
    'RAGv5Evidence',
]

