"""
Benchmark runner for systematic evaluation of RAG pipelines.

Runs pipelines on datasets and collects metrics in a structured way.
Supports:
- Running single pipeline on single dataset
- Running multiple pipelines on multiple datasets (matrix)
- Result caching and incremental runs
- Token usage tracking
"""
from typing import List, Dict, Any, Optional, Protocol
from pathlib import Path
import json
import time
from datetime import datetime
from pydantic import BaseModel, Field

from knowledge.models import Answer
from benchmarks.schemas import (
    DS1Dataset, DS2Dataset, DS3Dataset, DS4Dataset, DS5Dataset
)
from benchmarks.metrics import (
    compute_retrieval_metrics, compute_citation_metrics,
    compute_hallucination_metrics, judge_answer_quality
)


class BenchmarkResult(BaseModel):
    """Result from running a single pipeline on a single dataset."""
    
    # Identifiers
    run_id: str
    pipeline_name: str
    pipeline_version: str
    dataset_name: str
    dataset_type: str
    corpus_profile: str
    
    # Timing
    started_at: str
    completed_at: str
    total_time_seconds: float
    
    # Metrics (filled based on dataset type)
    retrieval_metrics: Optional[Dict[str, Any]] = None
    citation_metrics: Optional[Dict[str, Any]] = None
    hallucination_metrics: Optional[Dict[str, Any]] = None
    llm_judge_metrics: Optional[Dict[str, Any]] = None
    
    # Raw predictions (for debugging/analysis)
    predictions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Token usage
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    
    # Metadata
    config: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None


class BenchmarkMatrix(BaseModel):
    """Results from running multiple pipelines on multiple datasets."""
    
    run_id: str
    created_at: str
    
    results: List[BenchmarkResult]
    
    # Summary statistics
    total_pipelines: int = 0
    total_datasets: int = 0
    total_examples: int = 0
    total_time_seconds: float = 0.0
    
    def get_result(
        self,
        pipeline_name: str,
        dataset_name: str
    ) -> Optional[BenchmarkResult]:
        """Get result for specific pipeline + dataset."""
        for result in self.results:
            if (result.pipeline_name == pipeline_name and 
                result.dataset_name == dataset_name):
                return result
        return None


class BenchmarkRunner:
    """
    Runner for benchmarking RAG pipelines.
    
    Coordinates:
    - Pipeline execution on datasets
    - Metric computation
    - Result storage
    """
    
    def __init__(
        self,
        output_dir: Path,
        llm_client = None,
        cache_predictions: bool = True
    ):
        """
        Initialize runner.
        
        Args:
            output_dir: Directory for storing results
            llm_client: LLM client for judge (optional)
            cache_predictions: Whether to cache predictions
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm_client = llm_client
        self.cache_predictions = cache_predictions
        
        # Cache for predictions
        self.prediction_cache: Dict[str, List[Dict[str, Any]]] = {}
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{timestamp}"
    
    def _cache_key(
        self,
        pipeline_name: str,
        dataset_name: str,
        corpus_profile: str
    ) -> str:
        """Generate cache key."""
        return f"{pipeline_name}_{dataset_name}_{corpus_profile}"
    
    def run_pipeline_on_ds1(
        self,
        pipeline,
        dataset: DS1Dataset,
        corpus_profile: str = "public",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run pipeline on DS1 (Factual QA).
        
        Returns:
            List of predictions with answers
        """
        predictions = []
        
        for ex in dataset.examples:
            # Run pipeline
            answer: Answer = pipeline.run(
                query=ex.question,
                corpus_profile=corpus_profile,
                **kwargs
            )
            
            predictions.append({
                'example_id': ex.id,
                'question': ex.question,
                'answer': answer,
                'gold_answer': ex.gold_answer,
                'gold_citations': ex.gold_citations
            })
        
        return predictions
    
    def run_pipeline_on_ds2(
        self,
        pipeline,
        dataset: DS2Dataset,
        corpus_profile: str = "public",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run pipeline on DS2 (Retrieval).
        
        Returns only retrieval results (not full answer generation).
        """
        predictions = []
        
        for ex in dataset.examples:
            # For DS2, we only need retrieval step
            # Most pipelines expose retriever; get it if available
            if hasattr(pipeline, 'retriever'):
                retrieved = pipeline.retriever.retrieve(
                    query=ex.query,
                    top_k=kwargs.get('top_k', 20)
                )
                retrieved_ids = [chunk.chunk.id for chunk in retrieved]
            else:
                # Fallback: run full pipeline and extract retrieval from trace
                answer = pipeline.run(ex.query, corpus_profile, **kwargs)
                retrieved_ids = []
                if answer.trace and hasattr(answer.trace, 'retrieved_chunks'):
                    retrieved_ids = [c.id for c in answer.trace.retrieved_chunks]
            
            predictions.append({
                'query_id': ex.id,
                'retrieved_ids': retrieved_ids
            })
        
        return predictions
    
    def run_pipeline_on_ds3(
        self,
        pipeline,
        dataset: DS3Dataset,
        corpus_profile: str = "public",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run pipeline on DS3 (Unanswerable/Traps).
        
        Returns:
            List of predictions with answers
        """
        predictions = []
        
        for ex in dataset.examples:
            answer: Answer = pipeline.run(
                query=ex.question,
                corpus_profile=corpus_profile,
                **kwargs
            )
            
            predictions.append({
                'example_id': ex.id,
                'answer': answer,
                'should_refuse': True,  # All DS3 examples should be refused
                'trap_type': ex.trap_type
            })
        
        return predictions
    
    def run_pipeline_on_ds4(
        self,
        pipeline,
        dataset: DS4Dataset,
        corpus_profile: str = "public",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Run pipeline on DS4 (Multi-Hop)."""
        # Similar to DS1
        predictions = []
        
        for ex in dataset.examples:
            answer: Answer = pipeline.run(
                query=ex.question,
                corpus_profile=corpus_profile,
                **kwargs
            )
            
            predictions.append({
                'example_id': ex.id,
                'question': ex.question,
                'answer': answer,
                'gold_answer': ex.gold_answer,
                'gold_citations': ex.gold_citations
            })
        
        return predictions
    
    def run_pipeline_on_ds5(
        self,
        pipeline,
        dataset: DS5Dataset,
        corpus_profile: str = "public",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Run pipeline on DS5 (Structured Extraction)."""
        predictions = []
        
        for ex in dataset.examples:
            answer: Answer = pipeline.run(
                query=ex.question,
                corpus_profile=corpus_profile,
                **kwargs
            )
            
            # Try to parse answer as JSON
            try:
                structured_answer = json.loads(answer.text)
            except:
                structured_answer = {}
            
            predictions.append({
                'example_id': ex.id,
                'question': ex.question,
                'answer': answer,
                'structured_answer': structured_answer,
                'gold_output': ex.structured_output.gold_output,
                'required_fields': ex.required_fields
            })
        
        return predictions
    
    def run_single(
        self,
        pipeline,
        dataset: Any,
        dataset_type: str,
        corpus_profile: str = "public",
        run_id: Optional[str] = None,
        compute_judge: bool = False,
        **kwargs
    ) -> BenchmarkResult:
        """
        Run single pipeline on single dataset.
        
        Args:
            pipeline: Pipeline instance
            dataset: Dataset instance
            dataset_type: 'ds1', 'ds2', 'ds3', 'ds4', or 'ds5'
            corpus_profile: Corpus profile to use
            run_id: Optional run ID (generated if not provided)
            compute_judge: Whether to compute LLM judge scores
            **kwargs: Additional pipeline parameters
            
        Returns:
            BenchmarkResult with metrics
        """
        if not run_id:
            run_id = self._generate_run_id()
        
        print(f"Running {pipeline.name} on {dataset.name}...")
        
        started_at = datetime.now()
        start_time = time.time()
        
        # Check cache
        cache_key = self._cache_key(pipeline.name, dataset.name, corpus_profile)
        
        if self.cache_predictions and cache_key in self.prediction_cache:
            print(f"  Using cached predictions")
            predictions = self.prediction_cache[cache_key]
        else:
            # Run pipeline on dataset
            if dataset_type == 'ds1':
                predictions = self.run_pipeline_on_ds1(pipeline, dataset, corpus_profile, **kwargs)
            elif dataset_type == 'ds2':
                predictions = self.run_pipeline_on_ds2(pipeline, dataset, corpus_profile, **kwargs)
            elif dataset_type == 'ds3':
                predictions = self.run_pipeline_on_ds3(pipeline, dataset, corpus_profile, **kwargs)
            elif dataset_type == 'ds4':
                predictions = self.run_pipeline_on_ds4(pipeline, dataset, corpus_profile, **kwargs)
            elif dataset_type == 'ds5':
                predictions = self.run_pipeline_on_ds5(pipeline, dataset, corpus_profile, **kwargs)
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
            
            # Cache predictions
            if self.cache_predictions:
                self.prediction_cache[cache_key] = predictions
        
        completed_at = datetime.now()
        total_time = time.time() - start_time
        
        print(f"  Completed in {total_time:.2f}s")
        
        # Compute metrics based on dataset type
        result = BenchmarkResult(
            run_id=run_id,
            pipeline_name=pipeline.name,
            pipeline_version=pipeline.version,
            dataset_name=dataset.name,
            dataset_type=dataset_type.upper(),
            corpus_profile=corpus_profile,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            total_time_seconds=total_time,
            predictions=predictions,
            config=kwargs
        )
        
        # Compute appropriate metrics
        print(f"  Computing metrics...")
        
        if dataset_type == 'ds1':
            # Citation metrics
            citation_metrics = compute_citation_metrics(
                predictions,
                [{'example_id': ex.id, 'gold_citations': ex.gold_citations} 
                 for ex in dataset.examples]
            )
            result.citation_metrics = citation_metrics.model_dump()
            
            # LLM judge (optional)
            if compute_judge and self.llm_client:
                judge_metrics = judge_answer_quality(
                    predictions,
                    [{'example_id': ex.id, 'gold_answer': ex.gold_answer}
                     for ex in dataset.examples],
                    self.llm_client
                )
                result.llm_judge_metrics = judge_metrics.model_dump()
        
        elif dataset_type == 'ds2':
            # Retrieval metrics
            retrieval_metrics = compute_retrieval_metrics(
                predictions,
                [{'query_id': ex.id, 'qrels': {q.chunk_id: q.relevance for q in ex.qrels}}
                 for ex in dataset.examples]
            )
            result.retrieval_metrics = retrieval_metrics.model_dump()
        
        elif dataset_type == 'ds3':
            # Hallucination metrics
            hallucination_metrics = compute_hallucination_metrics(
                predictions,
                [{'example_id': ex.id, 'should_refuse': True, 'trap_type': ex.trap_type}
                 for ex in dataset.examples]
            )
            result.hallucination_metrics = hallucination_metrics.model_dump()
        
        elif dataset_type in ['ds4', 'ds5']:
            # For DS4/DS5: use citation + judge
            citation_metrics = compute_citation_metrics(
                predictions,
                [{'example_id': ex.id, 'gold_citations': ex.gold_citations}
                 for ex in dataset.examples]
            )
            result.citation_metrics = citation_metrics.model_dump()
        
        print(f"  Done!")
        
        return result
    
    def run_matrix(
        self,
        pipelines: List[Any],
        datasets: Dict[str, Any],
        corpus_profile: str = "public",
        compute_judge: bool = False,
        **kwargs
    ) -> BenchmarkMatrix:
        """
        Run multiple pipelines on multiple datasets.
        
        Args:
            pipelines: List of pipeline instances
            datasets: Dict of {dataset_type: dataset_instance}
            corpus_profile: Corpus profile
            compute_judge: Whether to compute LLM judge
            **kwargs: Additional pipeline parameters
            
        Returns:
            BenchmarkMatrix with all results
        """
        run_id = self._generate_run_id()
        created_at = datetime.now()
        
        print(f"\nBenchmark Matrix Run: {run_id}")
        print(f"Pipelines: {[p.name for p in pipelines]}")
        print(f"Datasets: {list(datasets.keys())}")
        print("=" * 60)
        
        results = []
        total_examples = 0
        total_time = 0.0
        
        for pipeline in pipelines:
            for dataset_type, dataset in datasets.items():
                result = self.run_single(
                    pipeline=pipeline,
                    dataset=dataset,
                    dataset_type=dataset_type,
                    corpus_profile=corpus_profile,
                    run_id=run_id,
                    compute_judge=compute_judge,
                    **kwargs
                )
                results.append(result)
                
                total_examples += len(result.predictions)
                total_time += result.total_time_seconds
        
        matrix = BenchmarkMatrix(
            run_id=run_id,
            created_at=created_at.isoformat(),
            results=results,
            total_pipelines=len(pipelines),
            total_datasets=len(datasets),
            total_examples=total_examples,
            total_time_seconds=total_time
        )
        
        # Save matrix
        output_file = self.output_dir / f"{run_id}_matrix.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(matrix.model_dump(), f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 60)
        print(f"Benchmark complete!")
        print(f"  Total examples: {total_examples}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Results saved to: {output_file}")
        
        return matrix

