#!/usr/bin/env python3
"""
Run benchmark evaluation on RAG pipelines.

Usage:
    python scripts/run_benchmark.py --config configs/benchmarks/default_config.yaml
    python scripts/run_benchmark.py --config configs/benchmarks/quick_test.yaml
    python scripts/run_benchmark.py --pipeline rag_v1_dense --dataset ds1
"""
import sys
import argparse
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.runner import BenchmarkRunner
from benchmarks.reports import ReportGenerator
from benchmarks.datasets.loader import load_dataset, load_all_datasets


def load_config(config_path: Path) -> dict:
    """Load benchmark configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_pipelines(
    pipeline_names: list,
    corpus_profile: str = "public",
    index_strategy: str = "fixed",
):
    """
    Load pipeline instances by name.

    This function wires minimal dependencies for RAG pipelines based on the
    locally built indices (see `python scripts/build_indices.py`).
    """
    from llm.deepseek_client import DeepSeekClient
    from rag.retrievers import DenseRetriever, BM25Retriever, HybridRetriever, MultiQueryRetriever
    from rag.rerankers import CrossEncoderReranker
    from rag.pipelines.rag_v1_dense import RAGv1Dense
    from rag.pipelines.rag_v2_hybrid import RAGv2Hybrid
    from rag.pipelines.rag_v3_multiquery import RAGv3MultiQuery

    from baselines.llm_direct import DeepSeekChatDirectBaseline, DeepSeekReasonerDirectBaseline

    # Index paths
    index_dir = project_root / "data" / "indices" / corpus_profile / index_strategy
    chunks_path = index_dir / "chunks.jsonl"

    # Build shared components lazily (only if needed)
    llm = None
    dense = None
    bm25 = None
    hybrid = None
    multi_query = None
    reranker = None

    def ensure_llm():
        nonlocal llm
        if llm is None:
            llm = DeepSeekClient(cache_enabled=True)
        return llm

    def ensure_indices_exist():
        if not index_dir.exists():
            raise FileNotFoundError(
                f"Index directory not found: {index_dir}. "
                f"Build indices first, e.g. `python scripts/build_indices.py --strategy {index_strategy}`."
            )
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunk store not found: {chunks_path}")

    def ensure_dense():
        nonlocal dense
        ensure_indices_exist()
        if dense is None:
            dense = DenseRetriever(
                index_dir=str(index_dir),
                chunks_jsonl_path=str(chunks_path),
                model_name="intfloat/e5-small-v2",
                device="cpu",
            )
        return dense

    def ensure_bm25():
        nonlocal bm25
        ensure_indices_exist()
        if bm25 is None:
            bm25 = BM25Retriever(
                index_dir=str(index_dir),
                chunks_jsonl_path=str(chunks_path),
            )
        return bm25

    def ensure_hybrid():
        nonlocal hybrid
        if hybrid is None:
            hybrid = HybridRetriever(ensure_bm25(), ensure_dense())
        return hybrid

    def ensure_multi_query():
        nonlocal multi_query
        if multi_query is None:
            multi_query = MultiQueryRetriever(
                ensure_hybrid(),
                expansion_strategy="pricing",
                max_queries=3,
            )
        return multi_query

    def ensure_reranker():
        nonlocal reranker
        if reranker is None:
            reranker = CrossEncoderReranker(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        return reranker

    pipelines = []
    for name in pipeline_names:
        if name == "rag_v1_dense":
            pipelines.append(
                RAGv1Dense(
                    dense_retriever=ensure_dense(),
                    llm_client=ensure_llm(),
                )
            )
        elif name == "rag_v2_hybrid":
            pipelines.append(
                RAGv2Hybrid(
                    hybrid_retriever=ensure_hybrid(),
                    reranker=ensure_reranker(),
                    llm_client=ensure_llm(),
                )
            )
        elif name == "rag_v3_multiquery":
            pipelines.append(
                RAGv3MultiQuery(
                    multi_query_retriever=ensure_multi_query(),
                    llm_client=ensure_llm(),
                    reranker=ensure_reranker(),
                )
            )
        elif name == "baseline_llm_chat":
            pipelines.append(DeepSeekChatDirectBaseline())
        elif name == "baseline_llm_reasoner":
            pipelines.append(DeepSeekReasonerDirectBaseline())
        else:
            # Try advanced pipelines if present
            if name == "rag_v4_parent_child":
                try:
                    from rag.pipelines.rag_v4_parent_child import RAGv4ParentChild

                    pipelines.append(
                        RAGv4ParentChild(
                            child_retriever=ensure_dense(),
                            llm_client=ensure_llm(),
                        )
                    )
                except ImportError:
                    print("Warning: rag_v4_parent_child not available, skipping")
            elif name == "rag_v5_evidence":
                try:
                    from rag.pipelines.rag_v5_evidence import RAGv5Evidence

                    pipelines.append(
                        RAGv5Evidence(
                            hybrid_retriever=ensure_hybrid(),
                            reranker=ensure_reranker(),
                            llm_client=ensure_llm(),
                            use_llm_validation=False,
                        )
                    )
                except ImportError:
                    print("Warning: rag_v5_evidence not available, skipping")
            else:
                print(f"Warning: Pipeline '{name}' not found, skipping")

    return pipelines


def load_datasets_from_config(dataset_names: list, datasets_dir: Path):
    """Load datasets by name."""
    datasets = {}
    
    for ds_name in dataset_names:
        ds_name_lower = ds_name.lower()
        
        # Try to find dataset file
        pattern = f"{ds_name_lower}_*.json"
        files = list(datasets_dir.glob(pattern))
        
        if files:
            dataset = load_dataset(files[0], ds_name_lower)
            datasets[ds_name_lower] = dataset
        else:
            print(f"Warning: Dataset '{ds_name}' not found in {datasets_dir}, skipping")
    
    return datasets


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="Run RAG benchmark evaluation")
    
    parser.add_argument(
        '--config',
        type=Path,
        default=project_root / "configs" / "benchmarks" / "default_config.yaml",
        help="Path to benchmark config file"
    )
    
    parser.add_argument(
        '--datasets-dir',
        type=Path,
        default=project_root / "benchmarks" / "datasets" / "examples",
        help="Directory containing dataset files"
    )
    
    parser.add_argument(
        '--pipeline',
        type=str,
        help="Run single pipeline (overrides config)"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help="Run single dataset (overrides config)"
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help="Run quick test mode (loads quick_test.yaml)"
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.quick:
        config_path = project_root / "configs" / "benchmarks" / "quick_test.yaml"
    else:
        config_path = args.config
    
    print(f"Loading config: {config_path}")
    config = load_config(config_path)
    
    # Override with CLI args
    if args.pipeline:
        config['pipelines'] = [args.pipeline]
    if args.dataset:
        config['datasets'] = [args.dataset]
    
    # Load pipelines
    print(f"\nLoading pipelines: {config['pipelines']}")
    pipelines = load_pipelines(
        config["pipelines"],
        corpus_profile=config.get("corpus_profile", "public"),
        index_strategy=config.get("index_strategy", "fixed"),
    )
    
    if not pipelines:
        print("Error: No pipelines loaded. Check pipeline names and implementation.")
        return 1
    
    print(f"✓ Loaded {len(pipelines)} pipeline(s)")
    
    # Load datasets
    print(f"\nLoading datasets from: {args.datasets_dir}")
    datasets = load_datasets_from_config(config['datasets'], args.datasets_dir)
    
    if not datasets:
        print("Error: No datasets loaded. Check dataset names and paths.")
        return 1
    
    print(f"✓ Loaded {len(datasets)} dataset(s)")
    for ds_name, ds in datasets.items():
        print(f"  - {ds_name}: {ds.total_examples} examples")
    
    # Initialize runner
    output_dir = project_root / config['output']['output_dir']
    runner = BenchmarkRunner(
        output_dir=output_dir,
        llm_client=None,  # TODO: pass LLM client if judge enabled
        cache_predictions=config['cache']['enable_prediction_cache']
    )
    
    # Run benchmark matrix
    print("\n" + "=" * 60)
    print("Starting benchmark evaluation...")
    print("=" * 60)
    
    matrix = runner.run_matrix(
        pipelines=pipelines,
        datasets=datasets,
        corpus_profile=config['corpus_profile'],
        compute_judge=config['evaluation']['compute_llm_judge'],
        **config['pipeline_params']
    )
    
    # Generate reports
    if config['output']['generate_reports']:
        print("\n" + "=" * 60)
        report_gen = ReportGenerator(output_dir)
        report_paths = report_gen.generate_complete_report(matrix)
        
        print("\n✓ Benchmark complete!")
        print(f"\nView results:")
        print(f"  - Markdown: {report_paths['markdown']}")
        print(f"  - HTML: {report_paths['html']}")
        print(f"  - CSV: {report_paths['csv']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

