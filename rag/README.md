# RAG Pipelines Documentation

This directory contains the complete RAG (Retrieval-Augmented Generation) system for the quantitative finance QA assistant.

## Architecture Overview

```
User Query
    ↓
[Retriever] → Get relevant chunks from corpus
    ↓
[Reranker] → (Optional) Reorder by relevance
    ↓
[Generator] → Generate answer with citations
    ↓
[Guardrails] → Validate evidence, detect unanswerable
    ↓
Answer + Citations
```

## Components

### 1. Retrievers (`retrievers/`)

- **DenseRetriever**: Semantic search using sentence transformers + FAISS
- **BM25Retriever**: Lexical search using BM25
- **HybridRetriever**: Combines BM25 + dense with reciprocal rank fusion (RRF)
- **MultiQueryRetriever**: Expands query with templates, fuses results

### 2. Rerankers (`rerankers/`)

- **CrossEncoderReranker**: Re-scores chunks using cross-encoder model (local, no API calls)

### 3. Generators (`generators/`)

- **CitationGenerator**: Generates answers with inline citations [1], [2], etc.
- Prompts optimized for DeepSeek API with cache-friendly structure

### 4. Guardrails (`guardrails/`)

- **EvidenceValidator**: Ensures claims are supported by citations
- **UnanswerableDetector**: Detects out-of-scope or unanswerable questions

## Pipelines

### RAGv1: Dense Retrieval

**Description**: Simple but effective baseline using dense retrieval.

**Steps**:
1. Dense retrieval (sentence transformer + FAISS)
2. Generate answer with citations
3. Optional unanswerable detection

**Use case**: Fast, semantic search. Good for well-defined questions.

**Example**:
```python
from rag.pipelines import RAGv1Dense
from rag.retrievers import DenseRetriever
from llm import DeepSeekClient

retriever = DenseRetriever(
    index_dir="data/indices/public/fixed",
    chunks_jsonl_path="data/indices/public/fixed/chunks.jsonl"
)
llm = DeepSeekClient()

pipeline = RAGv1Dense(retriever, llm)
answer = pipeline.run("What is the Black-Scholes formula?")
print(answer.text)
print(f"Citations: {len(answer.citations)}")
```

### RAGv2: Hybrid + Rerank

**Description**: Combines lexical (BM25) and semantic (dense) retrieval, then reranks.

**Steps**:
1. Hybrid retrieval (BM25 + dense with RRF fusion)
2. Rerank with cross-encoder
3. Generate answer with citations
4. Optional unanswerable detection

**Use case**: Best recall and precision. Recommended for production.

**Example**:
```python
from rag.pipelines import RAGv2Hybrid
from rag.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from rag.rerankers import CrossEncoderReranker
from llm import DeepSeekClient

bm25 = BM25Retriever(index_dir="...", chunks_jsonl_path="...")
dense = DenseRetriever(index_dir="...", chunks_jsonl_path="...")
hybrid = HybridRetriever(bm25, dense)
reranker = CrossEncoderReranker()
llm = DeepSeekClient()

pipeline = RAGv2Hybrid(hybrid, reranker, llm)
answer = pipeline.run("How to price American options?")
```

### RAGv3: Multi-Query + Fusion

**Description**: Expands query using domain templates (cheap, no LLM), fuses results.

**Steps**:
1. Expand query using templates (pricing, Greeks, methods, etc.)
2. Retrieve for each query variant
3. Fuse results with RRF
4. Optional reranking
5. Generate answer with citations

**Use case**: Improves coverage for multi-aspect questions. Token-efficient.

**Template strategies**:
- `pricing`: "How to price X", "Pricing formula for X", "Valuation of X"
- `greeks`: "Greeks for X", "Delta and gamma of X"
- `assumptions`: "Assumptions for X", "Requirements for X"
- `methods`: "Methods for X", "Approaches to X"

**Example**:
```python
from rag.pipelines import RAGv3MultiQuery
from rag.retrievers import HybridRetriever, MultiQueryRetriever

hybrid = HybridRetriever(bm25, dense)
multi_query = MultiQueryRetriever(
    hybrid, 
    expansion_strategy="pricing",
    max_queries=3
)

pipeline = RAGv3MultiQuery(multi_query, llm)
answer = pipeline.run("European call option pricing")
```

### RAGv4: Parent-Child

**Description**: Retrieves small chunks for precision, expands to parent context (page/section) for generation.

**Steps**:
1. Retrieve small "child" chunks for precision
2. Expand to "parent" context (page or section)
3. Generate answer from parent context with child-level citations

**Use case**: Better context for complex topics requiring surrounding information.

**Example**:
```python
from rag.pipelines import RAGv4ParentChild

pipeline = RAGv4ParentChild(
    child_retriever=dense,  # or hybrid
    llm_client=llm,
    parent_strategy="page",  # 'page' or 'section'
    child_top_k=15,
    parent_top_k=5
)

answer = pipeline.run("Explain Greeks in detail")
```

### RAGv5: Evidence Validation

**Description**: Generates answer, then validates that all claims are supported by evidence.

**Steps**:
1. Hybrid retrieval + reranking
2. Generate answer with citations
3. Validate evidence (rule-based or LLM)
4. Refuse or adjust confidence if validation fails

**Use case**: Maximum reliability. Ensures all claims have supporting evidence.

**Validation modes**:
- **Rule-based** (default): Fast, checks citation coverage
- **LLM-based** (optional): More accurate but costs tokens

**Example**:
```python
from rag.pipelines import RAGv5Evidence

pipeline = RAGv5Evidence(
    hybrid_retriever=hybrid,
    reranker=reranker,
    llm_client=llm,
    use_llm_validation=False,  # Use rule-based (cheaper)
    min_citation_coverage=0.5
)

answer = pipeline.run("What is VaR?")
print(f"Validation: {answer.metadata.get('validation')}")
print(f"Confidence: {answer.confidence}")
```

## Token Optimization

All pipelines are designed for token efficiency:

1. **Prompt caching**: Consistent system prompts for DeepSeek cache hits
2. **Local models**: Embeddings and reranking use local models (no API calls)
3. **Template-based expansion**: RAGv3 uses rules instead of LLM for query expansion
4. **Rule-based guardrails**: Evidence validation uses rules by default (LLM optional)
5. **Result caching**: LLM responses cached with deterministic keys

## Configuration

Pipelines can be configured via YAML files in `configs/pipelines/`:

```yaml
# Example: rag_v2.yaml
pipeline: RAGv2Hybrid
retrieval:
  top_k: 20
  bm25_weight: 0.5
  dense_weight: 0.5
rerank:
  top_k: 10
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
generation:
  model: "deepseek-chat"
  temperature: 0.0
  max_tokens: 2048
guardrails:
  unanswerable_detection: true
```

## Performance Benchmarks

Expected performance on public corpus (~700 pages):

| Pipeline | Retrieval | Generation | Total | Token Cost |
|----------|-----------|------------|-------|------------|
| RAGv1 | ~100ms | ~3s | ~3.1s | ~500 tokens |
| RAGv2 | ~150ms | ~3s | ~3.2s | ~500 tokens |
| RAGv3 | ~300ms | ~3s | ~3.3s | ~500 tokens |
| RAGv4 | ~200ms | ~3s | ~3.2s | ~600 tokens |
| RAGv5 | ~150ms | ~3.5s | ~3.7s | ~500-800 tokens* |

*RAGv5 with LLM validation adds ~300 tokens

## Testing

Run tests for RAG components:

```bash
# Unit tests
pytest tests/test_retrievers.py
pytest tests/test_generators.py

# Integration test
pytest tests/test_rag_pipelines.py

# Smoke test with real index
python scripts/test_rag_smoke.py
```

## Next Steps

1. **Benchmark**: Run all pipelines on DS1-DS5 datasets
2. **Tune**: Optimize top_k, weights, thresholds based on metrics
3. **Deploy**: Integrate with Telegram bot
4. **Monitor**: Track token usage, latency, cache hit rates

