"""
Dataset generator for creating benchmark examples.

Provides templates and utilities to generate examples for DS1-DS5.
In production, these would be combined with:
- Manual curation by domain experts
- LLM-assisted generation with human review
- Extraction from existing Q&A pairs

This module provides scaffolding and example generators.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from knowledge.models import PageSpan, Citation
from benchmarks.schemas import (
    DS1Dataset, DS1Example,
    DS2Dataset, DS2Example, DS2Qrel,
    DS3Dataset, DS3Example,
    DS4Dataset, DS4Example, DS4HopInfo,
    DS5Dataset, DS5Example, DS5StructuredOutput
)


class DatasetGenerator:
    """
    Generator for benchmark datasets.
    
    Provides templates and utilities for creating examples.
    """
    
    def __init__(self, corpus_profile: str = "public"):
        """
        Initialize generator.
        
        Args:
            corpus_profile: Corpus profile to use ('public')
        """
        self.corpus_profile = corpus_profile
        self.version = "1.0"
    
    # ==========================================================================
    # DS1: Factual QA
    # ==========================================================================
    
    def generate_ds1_example(
        self,
        example_id: str,
        question: str,
        gold_answer: str,
        doc_id: str,
        start_page: int,
        end_page: int,
        quote: str,
        topic: Optional[str] = None,
        difficulty: str = "medium"
    ) -> DS1Example:
        """
        Generate a DS1 example.
        
        Args:
            example_id: Unique ID
            question: Question text
            gold_answer: Gold standard answer
            doc_id: Document ID for citation
            start_page: Start page for citation
            end_page: End page for citation
            quote: Quote from document
            topic: Topic tag
            difficulty: Difficulty level
            
        Returns:
            DS1Example
        """
        citation = Citation(
            doc_id=doc_id,
            page_span=PageSpan(start_page=start_page, end_page=end_page),
            quote=quote,
            score=1.0,
            retriever_tag="gold"
        )
        
        return DS1Example(
            id=example_id,
            question=question,
            gold_answer=gold_answer,
            gold_citations=[citation],
            topic=topic,
            difficulty=difficulty
        )
    
    def create_empty_ds1(self, description: str = "") -> DS1Dataset:
        """Create empty DS1 dataset."""
        return DS1Dataset(
            version=self.version,
            name="FactualDerivativesQA",
            description=description or "Factual questions about derivatives with citations",
            examples=[]
        )
    
    # ==========================================================================
    # DS2: Retrieval Qrels
    # ==========================================================================
    
    def generate_ds2_example(
        self,
        example_id: str,
        query: str,
        qrels: List[Dict[str, Any]],
        query_type: Optional[str] = None
    ) -> DS2Example:
        """
        Generate a DS2 example.
        
        Args:
            example_id: Unique ID
            query: Query text
            qrels: List of dicts with chunk_id, relevance, doc_id
            query_type: Type of query
            
        Returns:
            DS2Example
        """
        qrel_objects = [
            DS2Qrel(
                chunk_id=q['chunk_id'],
                relevance=q['relevance'],
                doc_id=q['doc_id'],
                page_span=PageSpan(
                    start_page=q.get('start_page', 1),
                    end_page=q.get('end_page', 1)
                ) if 'start_page' in q else None
            )
            for q in qrels
        ]
        
        return DS2Example(
            id=example_id,
            query=query,
            qrels=qrel_objects,
            query_type=query_type
        )
    
    def create_empty_ds2(self, description: str = "") -> DS2Dataset:
        """Create empty DS2 dataset."""
        return DS2Dataset(
            version=self.version,
            name="RetrievalQrels",
            description=description or "Retrieval quality evaluation with relevance judgments",
            examples=[]
        )
    
    # ==========================================================================
    # DS3: Unanswerable and Traps
    # ==========================================================================
    
    def generate_ds3_example(
        self,
        example_id: str,
        question: str,
        reason_unanswerable: str,
        trap_type: str,
        expected_behavior: str = "refuse",
        trap_answer: Optional[str] = None
    ) -> DS3Example:
        """
        Generate a DS3 example.
        
        Args:
            example_id: Unique ID
            question: Question text
            reason_unanswerable: Why this should be refused
            trap_type: Type of trap
            expected_behavior: Expected system behavior
            trap_answer: Trap answer (if any)
            
        Returns:
            DS3Example
        """
        return DS3Example(
            id=example_id,
            question=question,
            reason_unanswerable=reason_unanswerable,
            trap_type=trap_type,
            expected_behavior=expected_behavior,
            trap_answer=trap_answer
        )
    
    def create_empty_ds3(self, description: str = "") -> DS3Dataset:
        """Create empty DS3 dataset."""
        return DS3Dataset(
            version=self.version,
            name="UnanswerableAndTraps",
            description=description or "Unanswerable questions and hallucination traps",
            examples=[]
        )
    
    # ==========================================================================
    # DS4: Multi-Hop
    # ==========================================================================
    
    def generate_ds4_example(
        self,
        example_id: str,
        question: str,
        gold_answer: str,
        hops: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        reasoning_type: str = "sequential",
        difficulty: str = "hard"
    ) -> DS4Example:
        """
        Generate a DS4 example.
        
        Args:
            example_id: Unique ID
            question: Question text
            gold_answer: Gold standard answer
            hops: List of hop information
            citations: List of citation dicts
            reasoning_type: Type of reasoning
            difficulty: Difficulty level
            
        Returns:
            DS4Example
        """
        hop_objects = [
            DS4HopInfo(
                hop_number=h['hop_number'],
                sub_question=h['sub_question'],
                required_doc_ids=h['required_doc_ids'],
                required_concepts=h['required_concepts']
            )
            for h in hops
        ]
        
        citation_objects = [
            Citation(
                doc_id=c['doc_id'],
                page_span=PageSpan(
                    start_page=c['start_page'],
                    end_page=c['end_page']
                ),
                quote=c['quote'],
                score=c.get('score', 1.0),
                retriever_tag=c.get('retriever_tag', 'gold')
            )
            for c in citations
        ]
        
        # Count unique doc_ids across all hops
        all_doc_ids = set()
        for h in hops:
            all_doc_ids.update(h['required_doc_ids'])
        
        return DS4Example(
            id=example_id,
            question=question,
            gold_answer=gold_answer,
            gold_citations=citation_objects,
            hops=hop_objects,
            reasoning_type=reasoning_type,
            min_required_sources=len(all_doc_ids),
            difficulty=difficulty
        )
    
    def create_empty_ds4(self, description: str = "") -> DS4Dataset:
        """Create empty DS4 dataset."""
        return DS4Dataset(
            version=self.version,
            name="MultiHopDerivatives",
            description=description or "Multi-hop reasoning questions",
            examples=[]
        )
    
    # ==========================================================================
    # DS5: Structured Extraction
    # ==========================================================================
    
    def generate_ds5_example(
        self,
        example_id: str,
        question: str,
        extraction_type: str,
        output_schema: Dict[str, Any],
        gold_output: Dict[str, Any],
        citations: List[Dict[str, Any]],
        required_fields: List[str]
    ) -> DS5Example:
        """
        Generate a DS5 example.
        
        Args:
            example_id: Unique ID
            question: Question text
            extraction_type: Type of extraction
            output_schema: JSON schema for output
            gold_output: Gold standard output
            citations: List of citation dicts
            required_fields: Required fields
            
        Returns:
            DS5Example
        """
        structured_output = DS5StructuredOutput(
            extraction_type=extraction_type,
            output_schema=output_schema,
            gold_output=gold_output
        )
        
        citation_objects = [
            Citation(
                doc_id=c['doc_id'],
                page_span=PageSpan(
                    start_page=c['start_page'],
                    end_page=c['end_page']
                ),
                quote=c['quote'],
                score=c.get('score', 1.0),
                retriever_tag=c.get('retriever_tag', 'gold')
            )
            for c in citations
        ]
        
        return DS5Example(
            id=example_id,
            question=question,
            structured_output=structured_output,
            gold_citations=citation_objects,
            required_fields=required_fields
        )
    
    def create_empty_ds5(self, description: str = "") -> DS5Dataset:
        """Create empty DS5 dataset."""
        return DS5Dataset(
            version=self.version,
            name="StructuredExtraction",
            description=description or "Structured extraction and classification",
            examples=[]
        )
    
    # ==========================================================================
    # Example templates
    # ==========================================================================
    
    def get_ds1_template_examples(self) -> List[DS1Example]:
        """Get template examples for DS1 (for demonstration)."""
        examples = [
            self.generate_ds1_example(
                example_id="ds1_template_001",
                question="What is the Black-Scholes formula for a European call option?",
                gold_answer="The Black-Scholes formula for a European call option is C = S₀N(d₁) - Ke⁻ʳᵀN(d₂)...",
                doc_id="arxiv_example",
                start_page=5,
                end_page=5,
                quote="The Black-Scholes formula gives the price of a European call...",
                topic="black_scholes",
                difficulty="medium"
            ),
            self.generate_ds1_example(
                example_id="ds1_template_002",
                question="What is Delta hedging?",
                gold_answer="Delta hedging is a risk management strategy...",
                doc_id="arxiv_example",
                start_page=10,
                end_page=10,
                quote="Delta hedging involves adjusting the portfolio...",
                topic="hedging",
                difficulty="easy"
            )
        ]
        return examples
    
    def get_ds3_template_examples(self) -> List[DS3Example]:
        """Get template examples for DS3."""
        examples = [
            self.generate_ds3_example(
                example_id="ds3_template_001",
                question="What was the implied volatility of SPX on March 15, 2023?",
                reason_unanswerable="Corpus contains theoretical models, not real-time market data",
                trap_type="out_of_scope",
                expected_behavior="refuse"
            ),
            self.generate_ds3_example(
                example_id="ds3_template_002",
                question="How do I calculate the Vega of a swap?",
                reason_unanswerable="Swaps typically don't have Vega (that's for options)",
                trap_type="similar_term_confusion",
                expected_behavior="clarify",
                trap_answer="To calculate Vega of a swap, use formula X..."
            )
        ]
        return examples

