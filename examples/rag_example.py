#!/usr/bin/env python3
"""
Example usage of RAG pipelines.

This example shows how to initialize and use all 5 RAG pipelines.
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm.deepseek_client import DeepSeekClient
from rag.retrievers import DenseRetriever, BM25Retriever, HybridRetriever, MultiQueryRetriever
from rag.rerankers import CrossEncoderReranker
from rag.pipelines import (
    RAGv1Dense, RAGv2Hybrid, RAGv3MultiQuery, 
    RAGv4ParentChild, RAGv5Evidence
)


def setup_components():
    """Initialize all RAG components."""
    
    # Paths (adjust to your setup)
    index_dir = "data/indices/public/fixed"
    chunks_path = f"{index_dir}/chunks.jsonl"
    
    # Check if indices exist
    if not Path(index_dir).exists():
        print(f"⚠️  Index directory not found: {index_dir}")
        print("Run: python scripts/build_indices.py --strategy fixed")
        return None
    
    print("Initializing components...")
    
    # 1. LLM Client
    llm = DeepSeekClient(cache_enabled=True)
    print("✓ DeepSeek client initialized")
    
    # 2. Retrievers
    dense = DenseRetriever(
        index_dir=index_dir,
        chunks_jsonl_path=chunks_path,
        model_name="intfloat/e5-small-v2"
    )
    print("✓ Dense retriever loaded")
    
    bm25 = BM25Retriever(
        index_dir=index_dir,
        chunks_jsonl_path=chunks_path
    )
    print("✓ BM25 retriever loaded")
    
    hybrid = HybridRetriever(bm25, dense)
    print("✓ Hybrid retriever initialized")
    
    multi_query = MultiQueryRetriever(
        hybrid,
        expansion_strategy="pricing",
        max_queries=3
    )
    print("✓ Multi-query retriever initialized")
    
    # 3. Reranker
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    print("✓ Cross-encoder reranker loaded")
    
    return {
        'llm': llm,
        'dense': dense,
        'bm25': bm25,
        'hybrid': hybrid,
        'multi_query': multi_query,
        'reranker': reranker
    }


def demo_pipeline(pipeline, query: str):
    """Demo a single pipeline."""
    print(f"\n{'='*60}")
    print(f"Pipeline: {pipeline.name} v{pipeline.version}")
    print(f"Query: {query}")
    print('='*60)
    
    answer = pipeline.run(query)
    
    print(f"\nAnswer ({answer.confidence:.2f} confidence):")
    print(answer.text)
    
    if answer.citations:
        print(f"\nCitations ({len(answer.citations)}):")
        for i, cit in enumerate(answer.citations[:3], 1):  # Show first 3
            print(f"  [{i}] {cit.format_page_reference()}")
            print(f"      \"{cit.quote[:80]}...\"")
    
    if answer.trace:
        print(f"\nTrace:")
        print(f"  Retrieved: {answer.trace.retrieved_chunks_count} chunks")
        print(f"  Retrieval time: {answer.trace.retrieval_time_ms:.1f}ms")
        print(f"  Generation time: {answer.trace.generation_time_ms:.1f}ms")
        print(f"  Total time: {answer.trace.metadata.get('total_time_ms', 0):.1f}ms")
    
    if answer.is_refused():
        print(f"\n⚠️  Refused: {answer.refusal_reason}")


def main():
    """Run examples for all pipelines."""
    
    # Check for API key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("⚠️  DEEPSEEK_API_KEY not set")
        print("Set with: export DEEPSEEK_API_KEY='your-key'")
        return
    
    # Setup
    components = setup_components()
    if not components:
        return
    
    # Example query
    query = "What is the Black-Scholes formula for a European call option?"
    
    print(f"\n{'#'*60}")
    print("RAG PIPELINES DEMO")
    print(f"{'#'*60}")
    
    # Pipeline 1: Dense
    pipeline_v1 = RAGv1Dense(
        dense_retriever=components['dense'],
        llm_client=components['llm'],
        top_k=10
    )
    demo_pipeline(pipeline_v1, query)
    
    # Pipeline 2: Hybrid + Rerank
    pipeline_v2 = RAGv2Hybrid(
        hybrid_retriever=components['hybrid'],
        reranker=components['reranker'],
        llm_client=components['llm'],
        retrieval_top_k=20,
        rerank_top_k=10
    )
    demo_pipeline(pipeline_v2, query)
    
    # Pipeline 3: Multi-Query
    pipeline_v3 = RAGv3MultiQuery(
        multi_query_retriever=components['multi_query'],
        llm_client=components['llm'],
        reranker=components['reranker'],
        retrieval_top_k=15,
        rerank_top_k=10
    )
    demo_pipeline(pipeline_v3, query)
    
    # Pipeline 4: Parent-Child
    pipeline_v4 = RAGv4ParentChild(
        child_retriever=components['dense'],
        llm_client=components['llm'],
        parent_strategy="page",
        child_top_k=15,
        parent_top_k=5
    )
    demo_pipeline(pipeline_v4, query)
    
    # Pipeline 5: Evidence Validation
    pipeline_v5 = RAGv5Evidence(
        hybrid_retriever=components['hybrid'],
        reranker=components['reranker'],
        llm_client=components['llm'],
        use_llm_validation=False,  # Use rule-based for demo
        retrieval_top_k=20,
        rerank_top_k=10
    )
    demo_pipeline(pipeline_v5, query)
    
    # Show token stats
    print(f"\n{'='*60}")
    print("Token Usage Summary")
    print('='*60)
    stats = components['llm'].get_stats()
    print(f"Total input tokens: {stats['total_input_tokens']}")
    print(f"Total output tokens: {stats['total_output_tokens']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache misses: {stats['cache_misses']}")
    print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

