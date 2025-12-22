#!/usr/bin/env python3
"""
Quick test script to verify benchmark infrastructure.

Tests:
- Schema validation
- Dataset generation
- Metric computation
- Report generation
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_schemas():
    """Test dataset schemas."""
    print("Testing schemas...")
    
    from benchmarks.schemas import (
        DS1Example, DS1Dataset,
        DS2Example, DS2Dataset,
        DS3Example, DS3Dataset
    )
    from knowledge.models import PageSpan, Citation
    
    # Test DS1
    citation = Citation(
        doc_id="test_doc",
        page_span=PageSpan(start_page=1, end_page=1),
        quote="Test quote",
        score=1.0,
        retriever_tag="test"
    )
    
    ex1 = DS1Example(
        id="test_001",
        question="What is Delta?",
        gold_answer="Delta is a Greek letter.",
        gold_citations=[citation]
    )
    
    ds1 = DS1Dataset(
        name="TestDS1",
        description="Test dataset",
        examples=[ex1]
    )
    
    assert ds1.total_examples == 1
    assert ex1.id == "test_001"
    
    print("  ✓ Schemas OK")


def test_metrics():
    """Test metric computation."""
    print("Testing metrics...")
    
    from benchmarks.metrics import compute_retrieval_metrics
    
    # Mock retrieval data
    predictions = [
        {'query_id': 'q1', 'retrieved_ids': ['c1', 'c2', 'c3', 'c4', 'c5']},
        {'query_id': 'q2', 'retrieved_ids': ['c10', 'c11', 'c1', 'c2']},
    ]
    
    ground_truth = [
        {'query_id': 'q1', 'qrels': {'c1': 2, 'c2': 1, 'c5': 2}},
        {'query_id': 'q2', 'qrels': {'c1': 2, 'c2': 2}},
    ]
    
    metrics = compute_retrieval_metrics(predictions, ground_truth)
    
    assert metrics.recall_at_5 > 0
    assert metrics.total_queries == 2
    
    print(f"  ✓ Retrieval metrics: Recall@5={metrics.recall_at_5:.3f}, nDCG@5={metrics.ndcg_at_5:.3f}")


def test_generator():
    """Test dataset generator."""
    print("Testing generator...")
    
    from benchmarks.datasets.generator import DatasetGenerator
    
    gen = DatasetGenerator()
    
    # Generate DS1 example
    ex = gen.generate_ds1_example(
        example_id="test_001",
        question="Test question?",
        gold_answer="Test answer",
        doc_id="test_doc",
        start_page=1,
        end_page=1,
        quote="Test quote"
    )
    
    assert ex.id == "test_001"
    assert ex.question == "Test question?"
    assert len(ex.gold_citations) == 1
    
    print("  ✓ Generator OK")


def test_validator():
    """Test dataset validator."""
    print("Testing validator...")
    
    from benchmarks.datasets.generator import DatasetGenerator
    from benchmarks.datasets.validator import DatasetValidator
    
    gen = DatasetGenerator()
    ds = gen.create_empty_ds1("Test dataset")
    
    # Add valid example
    ex = gen.generate_ds1_example(
        example_id="test_001",
        question="Valid question with at least 10 chars?",
        gold_answer="Valid answer with sufficient length",
        doc_id="test_doc",
        start_page=1,
        end_page=1,
        quote="Test quote"
    )
    ds.examples.append(ex)
    
    validator = DatasetValidator()
    report = validator.validate_ds1(ds)
    
    assert report.total_examples == 1
    print(f"  ✓ Validator: {len(report.errors)} errors, {len(report.warnings)} warnings")


def test_loader():
    """Test dataset loading/saving."""
    print("Testing loader...")
    
    from benchmarks.datasets.generator import DatasetGenerator
    from benchmarks.datasets.loader import save_dataset, load_dataset
    import tempfile
    
    gen = DatasetGenerator()
    ds = gen.create_empty_ds1("Test dataset")
    
    ex = gen.generate_ds1_example(
        example_id="test_001",
        question="Test?",
        gold_answer="Answer",
        doc_id="doc",
        start_page=1,
        end_page=1,
        quote="Quote"
    )
    ds.examples.append(ex)
    
    # Save
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = Path(f.name)
    
    save_dataset(ds, temp_path)
    
    # Load
    loaded_ds = load_dataset(temp_path, 'ds1')
    
    assert loaded_ds.total_examples == 1
    assert loaded_ds.examples[0].id == "test_001"
    
    # Cleanup
    temp_path.unlink()
    
    print("  ✓ Loader OK")


def test_report_generation():
    """Test report generation."""
    print("Testing report generation...")
    
    from benchmarks.runner import BenchmarkResult, BenchmarkMatrix
    from benchmarks.reports import ReportGenerator
    import tempfile
    from datetime import datetime
    
    # Create mock result
    result = BenchmarkResult(
        run_id="test_run",
        pipeline_name="test_pipeline",
        pipeline_version="1.0",
        dataset_name="test_dataset",
        dataset_type="DS1",
        corpus_profile="public",
        started_at=datetime.now().isoformat(),
        completed_at=datetime.now().isoformat(),
        total_time_seconds=1.5,
        retrieval_metrics={
            'recall_at_5': 0.85,
            'recall_at_10': 0.92,
            'ndcg_at_10': 0.78,
            'mrr': 0.65
        }
    )
    
    matrix = BenchmarkMatrix(
        run_id="test_run",
        created_at=datetime.now().isoformat(),
        results=[result],
        total_pipelines=1,
        total_datasets=1,
        total_examples=10,
        total_time_seconds=1.5
    )
    
    # Generate reports
    with tempfile.TemporaryDirectory() as tmpdir:
        report_gen = ReportGenerator(Path(tmpdir))
        
        # Markdown
        md = report_gen.generate_markdown_summary(matrix)
        assert "test_pipeline" in md
        assert "test_dataset" in md
        
        # HTML
        html = report_gen.generate_html_report(matrix)
        assert "<table>" in html
        assert "test_pipeline" in html
    
    print("  ✓ Report generation OK")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Benchmark Infrastructure Test Suite")
    print("=" * 60)
    print()
    
    try:
        test_schemas()
        test_metrics()
        test_generator()
        test_validator()
        test_loader()
        test_report_generation()
        
        print()
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Generate example datasets:")
        print("     python scripts/generate_example_datasets.py")
        print()
        print("  2. Run quick test benchmark:")
        print("     python scripts/run_benchmark.py --quick")
        print()
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

