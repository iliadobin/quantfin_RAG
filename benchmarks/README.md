# Benchmark Infrastructure

Comprehensive benchmark system for evaluating RAG pipelines on quantitative finance Q&A tasks.

## Overview

The benchmark infrastructure provides:

- **5 Dataset Types (DS1-DS5)**: Covering different aspects of RAG system evaluation
- **Multiple Metrics**: Retrieval, citation, hallucination, and LLM-judge metrics
- **Flexible Runner**: Matrix evaluation of pipelines × datasets
- **Report Generation**: Markdown, HTML, CSV, and JSON outputs
- **Token Optimization**: Cache-friendly design to minimize API costs

## Dataset Types

### DS1: Factual QA with Citations
- **Purpose**: Evaluate factual accuracy and citation quality
- **Size**: 200 examples (target)
- **Metrics**: Citation precision/recall, evidence coverage, LLM-judge

### DS2: Retrieval Quality (Qrels)
- **Purpose**: Evaluate retrieval effectiveness
- **Size**: 200 examples (target)
- **Metrics**: Recall@k, nDCG@k, MRR (no LLM needed)

### DS3: Unanswerable and Hallucination Traps
- **Purpose**: Test refusal behavior and robustness
- **Size**: 200 examples (target)
- **Metrics**: Refusal accuracy, hallucination rate, trap resistance

### DS4: Multi-Hop Reasoning
- **Purpose**: Evaluate complex reasoning across sources
- **Size**: 100 examples (target)
- **Metrics**: Citation metrics, LLM-judge

### DS5: Structured Extraction
- **Purpose**: Test structured output generation
- **Size**: 100 examples (target)
- **Metrics**: Schema compliance, field accuracy

## Quick Start

### 1. Generate Example Datasets

```bash
python scripts/generate_example_datasets.py
```

This creates small example datasets in `benchmarks/datasets/examples/` for testing.

### 2. Run Quick Test

```bash
python scripts/run_benchmark.py --quick
```

This runs a quick test on a subset of data (fast iteration).

### 3. Run Full Benchmark

```bash
python scripts/run_benchmark.py --config configs/benchmarks/full_evaluation.yaml
```

This runs comprehensive evaluation on all pipelines and datasets.

### 4. Run Specific Pipeline + Dataset

```bash
python scripts/run_benchmark.py --pipeline rag_v1_dense --dataset ds1
```

## Directory Structure

```
benchmarks/
├── schemas.py              # Pydantic schemas for DS1-DS5
├── runner.py               # Benchmark runner
├── reports.py              # Report generation
├── metrics/
│   ├── retrieval.py        # Retrieval metrics (Recall, nDCG, MRR)
│   ├── citation.py         # Citation metrics
│   ├── hallucination.py    # Hallucination/refusal metrics
│   └── llm_judge.py        # LLM-as-a-judge
├── datasets/
│   ├── generator.py        # Dataset generation utilities
│   ├── validator.py        # Dataset validation
│   ├── loader.py           # Load/save datasets
│   └── examples/           # Example datasets (generated)
└── README.md               # This file
```

## Creating Production Datasets

The example datasets are minimal (2-5 examples each). For production:

1. **Manual Curation**: Domain experts create examples
2. **LLM-Assisted**: Use LLM to generate candidates, then human review
3. **Extraction**: Mine from real user interactions

Target sizes:
- DS1: 200 examples
- DS2: 200 examples
- DS3: 200 examples
- DS4: 100 examples
- DS5: 100 examples

### Example: Creating DS1

```python
from benchmarks.datasets.generator import DatasetGenerator
from benchmarks.datasets.loader import save_dataset

gen = DatasetGenerator()
dataset = gen.create_empty_ds1(
    description="Factual QA about derivatives pricing"
)

# Add examples
example = gen.generate_ds1_example(
    example_id="ds1_001",
    question="What is the Black-Scholes formula?",
    gold_answer="The Black-Scholes formula is...",
    doc_id="arxiv_1234_5678",
    start_page=10,
    end_page=10,
    quote="The Black-Scholes formula gives...",
    topic="black_scholes",
    difficulty="medium"
)

dataset.examples.append(example)

# Validate and save
from benchmarks.datasets.validator import DatasetValidator
validator = DatasetValidator()
report = validator.validate_ds1(dataset)
print(report.get_summary())

save_dataset(dataset, "benchmarks/datasets/ds1_production.json")
```

## Configuration

Benchmark behavior is controlled by YAML config files in `configs/benchmarks/`:

### default_config.yaml
Standard configuration for regular runs.

### quick_test.yaml
Fast iteration mode with limited examples.

### full_evaluation.yaml
Comprehensive evaluation with LLM-judge on all datasets.

### Custom Config Example

```yaml
version: "1.0"

corpus_profile: "public"

datasets:
  - ds1
  - ds2

pipelines:
  - rag_v1_dense
  - rag_v2_hybrid

pipeline_params:
  top_k: 10
  temperature: 0.0

evaluation:
  compute_llm_judge: false

output:
  output_dir: "data/runs"
  generate_reports: true
```

## Metrics Reference

### Retrieval Metrics (DS2)

- **Recall@k**: Fraction of relevant items in top-k
- **nDCG@k**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank

No LLM calls required.

### Citation Metrics (DS1, DS4, DS5)

- **Citation Precision**: Fraction of citations that are correct
- **Citation Recall**: Fraction of gold citations found
- **Evidence Coverage**: Fraction of claims with supporting evidence

Uses heuristics, optionally LLM for claim→citation mapping.

### Hallucination Metrics (DS3)

- **Refusal Accuracy**: Correctly refused unanswerable questions
- **Hallucination Rate**: Answered without evidence when should refuse
- **False Refusal Rate**: Incorrectly refused answerable questions

Rule-based detection.

### LLM Judge (DS1, DS4, DS5)

Scores (0-5) for:
- **Correctness**: Factual accuracy
- **Completeness**: Coverage of question
- **Relevance**: Addresses what was asked

Requires LLM API calls (costs tokens).

## Token Budget Management

The benchmark is designed for token efficiency:

### Where LLM is NOT needed
- DS2 evaluation (pure retrieval metrics)
- Reranking (use local cross-encoder)
- Query expansion (template-based)
- Embeddings (local model)

### Where LLM is needed
- Answer generation (required)
- LLM judge (optional, configurable)

### Optimization strategies
1. **Prompt caching**: Static prefix + dynamic suffix
2. **Batching**: Combine multiple evaluations
3. **Sampling**: Judge only subset during dev
4. **Caching**: Cache predictions for repeated runs

Config example:
```yaml
evaluation:
  compute_llm_judge: true
  judge_sample_rate: 0.2  # Only 20% during iteration
  
token_budget:
  max_total_tokens: 1000000
  warn_threshold: 800000
```

## Output Formats

The benchmark generates multiple report formats:

### JSON (Raw Data)
Complete results with all predictions and metrics.
```
data/runs/run_20240101_120000_full.json
```

### Markdown (Readable)
Summary tables in markdown format.
```
data/runs/run_20240101_120000_summary.md
```

### HTML (Interactive)
HTML report with tables.
```
data/runs/run_20240101_120000_report.html
```

### CSV (Analysis)
CSV for Excel/pandas analysis.
```
data/runs/run_20240101_120000_summary.csv
```

## Extending the Benchmark

### Adding a New Dataset Type

1. Define schema in `benchmarks/schemas.py`
2. Add generator methods in `benchmarks/datasets/generator.py`
3. Add validator in `benchmarks/datasets/validator.py`
4. Add runner method in `benchmarks/runner.py`
5. Create example dataset

### Adding a New Metric

1. Create metric module in `benchmarks/metrics/`
2. Define metric computation function
3. Add to runner's metric computation logic
4. Update report generation to include new metric

## Best Practices

### Dataset Quality
- Have domain experts review examples
- Validate citations against actual documents
- Include diverse difficulty levels
- Balance topics/categories

### Running Benchmarks
- Start with quick test mode
- Use prediction caching for iteration
- Monitor token usage
- Run full evaluation less frequently

### Reproducibility
- Fix random seeds in config
- Use deterministic operations
- Save full config with results
- Version datasets and pipelines

## Troubleshooting

### "Pipeline not found"
Ensure pipeline is implemented and imported in `scripts/run_benchmark.py`.

### "Dataset not found"
Check dataset files exist in `benchmarks/datasets/examples/` or specified directory.

### Out of memory
Reduce batch sizes or dataset sizes in config.

### High token usage
- Disable LLM judge during development
- Reduce judge_sample_rate
- Use smaller datasets for iteration

## Related Documentation

- [RAG Implementation Summary](../RAG_IMPLEMENTATION_SUMMARY.md)
- [Pipeline Configs](../configs/pipelines/)
- [Epic D Plan](../.cursor/plans/quantfinance_qa_assistant_*.plan.md)

