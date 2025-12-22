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


def load_pipelines(pipeline_names: list):
    """Load pipeline instances by name."""
    # Import pipeline classes
    from rag.pipelines.rag_v1_dense import RAGv1Dense
    from rag.pipelines.rag_v2_hybrid import RAGv2Hybrid
    from rag.pipelines.rag_v3_multiquery import RAGv3MultiQuery
    
    pipeline_classes = {
        'rag_v1_dense': RAGv1Dense,
        'rag_v2_hybrid': RAGv2Hybrid,
        'rag_v3_multiquery': RAGv3MultiQuery,
    }
    
    # Try to import advanced pipelines (may not exist yet)
    try:
        from rag.pipelines.rag_v4_parent_child import RAGv4ParentChild
        pipeline_classes['rag_v4_parent_child'] = RAGv4ParentChild
    except ImportError:
        pass
    
    try:
        from rag.pipelines.rag_v5_evidence import RAGv5Evidence
        pipeline_classes['rag_v5_evidence'] = RAGv5Evidence
    except ImportError:
        pass
    
    pipelines = []
    for name in pipeline_names:
        if name in pipeline_classes:
            # Initialize pipeline
            # Note: In production, would pass proper config/dependencies
            pipeline = pipeline_classes[name]()
            pipelines.append(pipeline)
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
    pipelines = load_pipelines(config['pipelines'])
    
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

