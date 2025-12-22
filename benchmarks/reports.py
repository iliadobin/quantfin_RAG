"""
Report generation and formatting for benchmark results.

Provides multiple output formats:
- JSON (raw data)
- Markdown tables (readable reports)
- HTML (interactive reports)
- CSV (for analysis in Excel/pandas)
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import csv
from datetime import datetime

from benchmarks.runner import BenchmarkResult, BenchmarkMatrix


class ReportGenerator:
    """
    Generate reports from benchmark results.
    
    Supports multiple output formats for different use cases.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for report outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================================
    # Summary tables
    # ==========================================================================
    
    def _extract_metric_summary(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Extract key metrics from result for summary table."""
        summary = {
            'pipeline': result.pipeline_name,
            'dataset': result.dataset_name,
            'dataset_type': result.dataset_type,
            'num_examples': len(result.predictions),
            'time_seconds': f"{result.total_time_seconds:.2f}",
        }
        
        # Extract metrics based on dataset type
        if result.retrieval_metrics:
            summary.update({
                'recall@5': f"{result.retrieval_metrics['recall_at_5']:.3f}",
                'recall@10': f"{result.retrieval_metrics['recall_at_10']:.3f}",
                'ndcg@10': f"{result.retrieval_metrics['ndcg_at_10']:.3f}",
                'mrr': f"{result.retrieval_metrics['mrr']:.3f}",
            })
        
        if result.citation_metrics:
            summary.update({
                'citation_precision': f"{result.citation_metrics['citation_precision']:.3f}",
                'citation_recall': f"{result.citation_metrics['citation_recall']:.3f}",
                'evidence_coverage': f"{result.citation_metrics['evidence_coverage']:.3f}",
            })
        
        if result.hallucination_metrics:
            summary.update({
                'refusal_accuracy': f"{result.hallucination_metrics['refusal_accuracy']:.3f}",
                'hallucination_rate': f"{result.hallucination_metrics['hallucination_rate']:.3f}",
            })
        
        if result.llm_judge_metrics:
            summary.update({
                'judge_correctness': f"{result.llm_judge_metrics['mean_correctness']:.2f}",
                'judge_completeness': f"{result.llm_judge_metrics['mean_completeness']:.2f}",
                'judge_overall': f"{result.llm_judge_metrics['mean_overall']:.2f}",
            })
        
        return summary
    
    # ==========================================================================
    # Markdown format
    # ==========================================================================
    
    def generate_markdown_summary(self, matrix: BenchmarkMatrix) -> str:
        """
        Generate markdown summary table.
        
        Args:
            matrix: Benchmark matrix results
            
        Returns:
            Markdown formatted string
        """
        lines = [
            f"# Benchmark Results: {matrix.run_id}",
            f"",
            f"**Date**: {matrix.created_at}",
            f"**Pipelines**: {matrix.total_pipelines}",
            f"**Datasets**: {matrix.total_datasets}",
            f"**Total Examples**: {matrix.total_examples}",
            f"**Total Time**: {matrix.total_time_seconds:.2f}s",
            f"",
            f"---",
            f""
        ]
        
        # Group results by dataset type
        by_dataset_type: Dict[str, List[BenchmarkResult]] = {}
        for result in matrix.results:
            ds_type = result.dataset_type
            if ds_type not in by_dataset_type:
                by_dataset_type[ds_type] = []
            by_dataset_type[ds_type].append(result)
        
        # Generate table for each dataset type
        for ds_type in sorted(by_dataset_type.keys()):
            results = by_dataset_type[ds_type]
            lines.append(f"## {ds_type} Results")
            lines.append("")
            
            # Build table
            summaries = [self._extract_metric_summary(r) for r in results]
            
            if summaries:
                # Get all unique keys across summaries
                all_keys = set()
                for s in summaries:
                    all_keys.update(s.keys())
                
                # Order keys sensibly
                key_order = ['pipeline', 'dataset', 'num_examples', 'time_seconds']
                metric_keys = sorted(all_keys - set(key_order))
                ordered_keys = key_order + metric_keys
                
                # Header
                header = " | ".join(ordered_keys)
                lines.append(f"| {header} |")
                lines.append(f"| {' | '.join(['---'] * len(ordered_keys))} |")
                
                # Rows
                for summary in summaries:
                    row = " | ".join(str(summary.get(k, '-')) for k in ordered_keys)
                    lines.append(f"| {row} |")
                
                lines.append("")
        
        # Add notes
        lines.extend([
            "---",
            "",
            "## Metric Definitions",
            "",
            "**Retrieval Metrics**:",
            "- `recall@k`: Fraction of relevant items in top-k results",
            "- `ndcg@k`: Normalized Discounted Cumulative Gain at k",
            "- `mrr`: Mean Reciprocal Rank",
            "",
            "**Citation Metrics**:",
            "- `citation_precision`: Fraction of citations that are correct",
            "- `citation_recall`: Fraction of gold citations found",
            "- `evidence_coverage`: Fraction of claims with supporting evidence",
            "",
            "**Hallucination Metrics**:",
            "- `refusal_accuracy`: Correctly refused unanswerable questions",
            "- `hallucination_rate`: Answered without evidence when should refuse",
            "",
            "**LLM Judge Metrics** (0-5 scale):",
            "- `judge_correctness`: Factual correctness",
            "- `judge_completeness`: Answer completeness",
            "- `judge_overall`: Overall quality",
            ""
        ])
        
        return "\n".join(lines)
    
    def save_markdown_summary(
        self,
        matrix: BenchmarkMatrix,
        filename: Optional[str] = None
    ) -> Path:
        """Save markdown summary to file."""
        if not filename:
            filename = f"{matrix.run_id}_summary.md"
        
        output_path = self.output_dir / filename
        
        markdown = self.generate_markdown_summary(matrix)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        return output_path
    
    # ==========================================================================
    # HTML format
    # ==========================================================================
    
    def generate_html_report(self, matrix: BenchmarkMatrix) -> str:
        """
        Generate HTML report with interactive tables.
        
        Args:
            matrix: Benchmark matrix results
            
        Returns:
            HTML string
        """
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "  <meta charset='utf-8'>",
            f"  <title>Benchmark Results: {matrix.run_id}</title>",
            "  <style>",
            "    body { font-family: Arial, sans-serif; margin: 20px; }",
            "    h1 { color: #333; }",
            "    h2 { color: #666; margin-top: 30px; }",
            "    table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "    th { background-color: #4CAF50; color: white; }",
            "    tr:nth-child(even) { background-color: #f2f2f2; }",
            "    .metric-good { color: green; font-weight: bold; }",
            "    .metric-bad { color: red; }",
            "    .summary { background-color: #f9f9f9; padding: 15px; border-radius: 5px; }",
            "  </style>",
            "</head>",
            "<body>",
            f"  <h1>Benchmark Results: {matrix.run_id}</h1>",
            f"  <div class='summary'>",
            f"    <p><strong>Date:</strong> {matrix.created_at}</p>",
            f"    <p><strong>Pipelines:</strong> {matrix.total_pipelines}</p>",
            f"    <p><strong>Datasets:</strong> {matrix.total_datasets}</p>",
            f"    <p><strong>Total Examples:</strong> {matrix.total_examples}</p>",
            f"    <p><strong>Total Time:</strong> {matrix.total_time_seconds:.2f}s</p>",
            f"  </div>",
        ]
        
        # Group by dataset type
        by_dataset_type: Dict[str, List[BenchmarkResult]] = {}
        for result in matrix.results:
            ds_type = result.dataset_type
            if ds_type not in by_dataset_type:
                by_dataset_type[ds_type] = []
            by_dataset_type[ds_type].append(result)
        
        # Tables for each dataset type
        for ds_type in sorted(by_dataset_type.keys()):
            results = by_dataset_type[ds_type]
            html_parts.append(f"  <h2>{ds_type} Results</h2>")
            
            summaries = [self._extract_metric_summary(r) for r in results]
            
            if summaries:
                # Get all keys
                all_keys = set()
                for s in summaries:
                    all_keys.update(s.keys())
                
                key_order = ['pipeline', 'dataset', 'num_examples', 'time_seconds']
                metric_keys = sorted(all_keys - set(key_order))
                ordered_keys = key_order + metric_keys
                
                # Build table
                html_parts.append("  <table>")
                html_parts.append("    <thead>")
                html_parts.append("      <tr>")
                for key in ordered_keys:
                    html_parts.append(f"        <th>{key}</th>")
                html_parts.append("      </tr>")
                html_parts.append("    </thead>")
                html_parts.append("    <tbody>")
                
                for summary in summaries:
                    html_parts.append("      <tr>")
                    for key in ordered_keys:
                        value = summary.get(key, '-')
                        html_parts.append(f"        <td>{value}</td>")
                    html_parts.append("      </tr>")
                
                html_parts.append("    </tbody>")
                html_parts.append("  </table>")
        
        html_parts.extend([
            "</body>",
            "</html>"
        ])
        
        return "\n".join(html_parts)
    
    def save_html_report(
        self,
        matrix: BenchmarkMatrix,
        filename: Optional[str] = None
    ) -> Path:
        """Save HTML report to file."""
        if not filename:
            filename = f"{matrix.run_id}_report.html"
        
        output_path = self.output_dir / filename
        
        html = self.generate_html_report(matrix)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_path
    
    # ==========================================================================
    # CSV format
    # ==========================================================================
    
    def save_csv_summary(
        self,
        matrix: BenchmarkMatrix,
        filename: Optional[str] = None
    ) -> Path:
        """
        Save summary as CSV for analysis.
        
        Args:
            matrix: Benchmark matrix results
            filename: Output filename
            
        Returns:
            Path to saved CSV file
        """
        if not filename:
            filename = f"{matrix.run_id}_summary.csv"
        
        output_path = self.output_dir / filename
        
        # Extract all summaries
        summaries = [self._extract_metric_summary(r) for r in matrix.results]
        
        if not summaries:
            return output_path
        
        # Get all keys
        all_keys = set()
        for s in summaries:
            all_keys.update(s.keys())
        
        ordered_keys = sorted(all_keys)
        
        # Write CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_keys)
            writer.writeheader()
            for summary in summaries:
                writer.writerow(summary)
        
        return output_path
    
    # ==========================================================================
    # Complete report package
    # ==========================================================================
    
    def generate_complete_report(
        self,
        matrix: BenchmarkMatrix
    ) -> Dict[str, Path]:
        """
        Generate complete report package with all formats.
        
        Args:
            matrix: Benchmark matrix results
            
        Returns:
            Dict mapping format name to output path
        """
        print(f"\nGenerating reports for {matrix.run_id}...")
        
        outputs = {}
        
        # JSON (raw data)
        json_path = self.output_dir / f"{matrix.run_id}_full.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(matrix.model_dump(), f, indent=2, ensure_ascii=False)
        outputs['json'] = json_path
        print(f"  ✓ JSON: {json_path}")
        
        # Markdown
        md_path = self.save_markdown_summary(matrix)
        outputs['markdown'] = md_path
        print(f"  ✓ Markdown: {md_path}")
        
        # HTML
        html_path = self.save_html_report(matrix)
        outputs['html'] = html_path
        print(f"  ✓ HTML: {html_path}")
        
        # CSV
        csv_path = self.save_csv_summary(matrix)
        outputs['csv'] = csv_path
        print(f"  ✓ CSV: {csv_path}")
        
        print(f"\nReports saved to: {self.output_dir}")
        
        return outputs


def load_benchmark_result(path: Path) -> BenchmarkResult:
    """Load benchmark result from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return BenchmarkResult(**data)


def load_benchmark_matrix(path: Path) -> BenchmarkMatrix:
    """Load benchmark matrix from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return BenchmarkMatrix(**data)

