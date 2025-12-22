#!/usr/bin/env python3
"""
Generate example datasets for demonstration and testing.

This creates small example datasets (DS1-DS5) that demonstrate the schema
and can be used for testing the benchmark infrastructure.

In production, these would be replaced with:
- Manually curated examples by domain experts
- LLM-assisted generation with human review
- Extraction from real Q&A interactions
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.datasets.generator import DatasetGenerator
from benchmarks.datasets.loader import save_dataset
from benchmarks.datasets.validator import DatasetValidator


def generate_ds1_examples():
    """Generate example DS1 (Factual QA) dataset."""
    gen = DatasetGenerator()
    dataset = gen.create_empty_ds1(
        description="Example factual QA dataset about derivatives pricing"
    )
    
    # Add template examples
    examples = [
        gen.generate_ds1_example(
            example_id="ds1_ex001",
            question="What is the Black-Scholes formula for a European call option?",
            gold_answer="The Black-Scholes formula for a European call option is C = S₀N(d₁) - Ke⁻ʳᵀN(d₂), where d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T) and d₂ = d₁ - σ√T.",
            doc_id="arxiv_example_bs",
            start_page=5,
            end_page=5,
            quote="The Black-Scholes formula gives the price of a European call option...",
            topic="black_scholes",
            difficulty="medium"
        ),
        gen.generate_ds1_example(
            example_id="ds1_ex002",
            question="What is Delta in options trading?",
            gold_answer="Delta measures the rate of change of the option price with respect to changes in the underlying asset price. For a call option, Delta ranges from 0 to 1.",
            doc_id="arxiv_example_greeks",
            start_page=12,
            end_page=12,
            quote="Delta, the first derivative of the option price with respect to spot...",
            topic="greeks",
            difficulty="easy"
        ),
        gen.generate_ds1_example(
            example_id="ds1_ex003",
            question="What is the put-call parity relationship?",
            gold_answer="Put-call parity states that C - P = S - Ke⁻ʳᵀ for European options, where C is call price, P is put price, S is spot price, K is strike, r is risk-free rate, and T is time to maturity.",
            doc_id="arxiv_example_parity",
            start_page=8,
            end_page=8,
            quote="The put-call parity relationship holds for European options...",
            topic="parity",
            difficulty="medium"
        ),
    ]
    
    dataset.examples = examples
    
    # Re-initialize to compute statistics
    dataset.__init__(**dataset.model_dump())
    
    return dataset


def generate_ds2_examples():
    """Generate example DS2 (Retrieval Qrels) dataset."""
    gen = DatasetGenerator()
    dataset = gen.create_empty_ds2(
        description="Example retrieval evaluation dataset"
    )
    
    examples = [
        gen.generate_ds2_example(
            example_id="ds2_ex001",
            query="Black-Scholes formula components",
            qrels=[
                {'chunk_id': 'chunk_bs_001', 'relevance': 2, 'doc_id': 'arxiv_example_bs', 'start_page': 5, 'end_page': 5},
                {'chunk_id': 'chunk_bs_002', 'relevance': 1, 'doc_id': 'arxiv_example_bs', 'start_page': 6, 'end_page': 6},
                {'chunk_id': 'chunk_other_001', 'relevance': 0, 'doc_id': 'arxiv_other', 'start_page': 1, 'end_page': 1},
            ],
            query_type="formula"
        ),
        gen.generate_ds2_example(
            example_id="ds2_ex002",
            query="What are the Greeks in options?",
            qrels=[
                {'chunk_id': 'chunk_greeks_001', 'relevance': 2, 'doc_id': 'arxiv_example_greeks', 'start_page': 10, 'end_page': 10},
                {'chunk_id': 'chunk_greeks_002', 'relevance': 2, 'doc_id': 'arxiv_example_greeks', 'start_page': 11, 'end_page': 11},
            ],
            query_type="definition"
        ),
    ]
    
    dataset.examples = examples
    dataset.__init__(**dataset.model_dump())
    
    return dataset


def generate_ds3_examples():
    """Generate example DS3 (Unanswerable/Traps) dataset."""
    gen = DatasetGenerator()
    dataset = gen.create_empty_ds3(
        description="Example unanswerable questions and hallucination traps"
    )
    
    examples = [
        gen.generate_ds3_example(
            example_id="ds3_ex001",
            question="What was the closing price of SPX on December 22, 2023?",
            reason_unanswerable="Corpus contains theoretical models, not real-time market data",
            trap_type="out_of_scope",
            expected_behavior="refuse"
        ),
        gen.generate_ds3_example(
            example_id="ds3_ex002",
            question="How do I calculate the Vega of a plain vanilla swap?",
            reason_unanswerable="Plain vanilla swaps don't have Vega (that's for options)",
            trap_type="similar_term_confusion",
            expected_behavior="clarify",
            trap_answer="To calculate Vega of a swap, differentiate with respect to volatility..."
        ),
        gen.generate_ds3_example(
            example_id="ds3_ex003",
            question="What is the best strategy for making money in options?",
            reason_unanswerable="Question is too vague and asks for investment advice",
            trap_type="ambiguous",
            expected_behavior="clarify"
        ),
    ]
    
    dataset.examples = examples
    dataset.__init__(**dataset.model_dump())
    
    return dataset


def generate_ds4_examples():
    """Generate example DS4 (Multi-Hop) dataset."""
    gen = DatasetGenerator()
    dataset = gen.create_empty_ds4(
        description="Example multi-hop reasoning questions"
    )
    
    examples = [
        gen.generate_ds4_example(
            example_id="ds4_ex001",
            question="How do Delta and Gamma work together in dynamic hedging?",
            gold_answer="Delta measures first-order price sensitivity, while Gamma measures the rate of change of Delta. In dynamic hedging, traders rebalance Delta exposure, with Gamma indicating how frequently rebalancing is needed.",
            hops=[
                {
                    'hop_number': 1,
                    'sub_question': 'What is Delta?',
                    'required_doc_ids': ['arxiv_example_greeks'],
                    'required_concepts': ['delta', 'sensitivity']
                },
                {
                    'hop_number': 2,
                    'sub_question': 'What is Gamma?',
                    'required_doc_ids': ['arxiv_example_greeks'],
                    'required_concepts': ['gamma', 'second_derivative']
                },
                {
                    'hop_number': 3,
                    'sub_question': 'How are they used in hedging?',
                    'required_doc_ids': ['arxiv_example_hedging'],
                    'required_concepts': ['dynamic_hedging', 'rebalancing']
                }
            ],
            citations=[
                {'doc_id': 'arxiv_example_greeks', 'start_page': 12, 'end_page': 12, 'quote': 'Delta measures...'},
                {'doc_id': 'arxiv_example_greeks', 'start_page': 14, 'end_page': 14, 'quote': 'Gamma is the second derivative...'},
                {'doc_id': 'arxiv_example_hedging', 'start_page': 20, 'end_page': 20, 'quote': 'Dynamic hedging involves...'},
            ],
            reasoning_type="sequential",
            difficulty="hard"
        ),
    ]
    
    dataset.examples = examples
    dataset.__init__(**dataset.model_dump())
    
    return dataset


def generate_ds5_examples():
    """Generate example DS5 (Structured Extraction) dataset."""
    gen = DatasetGenerator()
    dataset = gen.create_empty_ds5(
        description="Example structured extraction dataset"
    )
    
    examples = [
        gen.generate_ds5_example(
            example_id="ds5_ex001",
            question="Extract the Black-Scholes formula and its components.",
            extraction_type="formula",
            output_schema={
                "type": "object",
                "properties": {
                    "formula": {"type": "string"},
                    "components": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string"},
                                "description": {"type": "string"}
                            }
                        }
                    }
                }
            },
            gold_output={
                "formula": "C = S₀N(d₁) - Ke⁻ʳᵀN(d₂)",
                "components": [
                    {"symbol": "S₀", "description": "Current stock price"},
                    {"symbol": "K", "description": "Strike price"},
                    {"symbol": "r", "description": "Risk-free rate"},
                    {"symbol": "T", "description": "Time to maturity"},
                    {"symbol": "σ", "description": "Volatility"},
                ]
            },
            citations=[
                {'doc_id': 'arxiv_example_bs', 'start_page': 5, 'end_page': 5, 'quote': 'The Black-Scholes formula...'}
            ],
            required_fields=["formula", "components"]
        ),
    ]
    
    dataset.examples = examples
    dataset.__init__(**dataset.model_dump())
    
    return dataset


def main():
    """Generate all example datasets."""
    print("Generating example datasets...")
    
    output_dir = project_root / "benchmarks" / "datasets" / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    validator = DatasetValidator()
    
    # DS1
    print("\n1. Generating DS1 (Factual QA)...")
    ds1 = generate_ds1_examples()
    save_dataset(ds1, output_dir / "ds1_factual_qa_example.json")
    report = validator.validate_ds1(ds1)
    print(f"   {report.get_summary()}")
    
    # DS2
    print("\n2. Generating DS2 (Retrieval Qrels)...")
    ds2 = generate_ds2_examples()
    save_dataset(ds2, output_dir / "ds2_retrieval_qrels_example.json")
    report = validator.validate_ds2(ds2)
    print(f"   {report.get_summary()}")
    
    # DS3
    print("\n3. Generating DS3 (Unanswerable/Traps)...")
    ds3 = generate_ds3_examples()
    save_dataset(ds3, output_dir / "ds3_unanswerable_traps_example.json")
    report = validator.validate_ds3(ds3)
    print(f"   {report.get_summary()}")
    
    # DS4
    print("\n4. Generating DS4 (Multi-Hop)...")
    ds4 = generate_ds4_examples()
    save_dataset(ds4, output_dir / "ds4_multihop_example.json")
    report = validator.validate_ds4(ds4)
    print(f"   {report.get_summary()}")
    
    # DS5
    print("\n5. Generating DS5 (Structured Extraction)...")
    ds5 = generate_ds5_examples()
    save_dataset(ds5, output_dir / "ds5_structured_extraction_example.json")
    report = validator.validate_ds5(ds5)
    print(f"   {report.get_summary()}")
    
    print(f"\n✓ All example datasets generated in: {output_dir}")
    print(f"\nThese are minimal examples for testing the infrastructure.")
    print(f"For production, you should create full datasets with:")
    print(f"  - DS1: 200 examples")
    print(f"  - DS2: 200 examples")
    print(f"  - DS3: 200 examples")
    print(f"  - DS4: 100 examples")
    print(f"  - DS5: 100 examples")


if __name__ == "__main__":
    main()

