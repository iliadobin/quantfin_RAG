"""
Baselines for benchmarking (non-RAG).

These are intentionally simple reference systems used as comparison points
against the RAG pipelines.
"""

from baselines.llm_direct import (
    LLMDirectBaseline,
    DeepSeekChatDirectBaseline,
    DeepSeekReasonerDirectBaseline,
)

__all__ = [
    "LLMDirectBaseline",
    "DeepSeekChatDirectBaseline",
    "DeepSeekReasonerDirectBaseline",
]


