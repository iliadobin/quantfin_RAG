"""
Pydantic schemas for all benchmark datasets (DS1-DS5).

Each dataset has a specific purpose in evaluating RAG system capabilities:
- DS1: Factual QA with citations
- DS2: Retrieval quality (qrels)
- DS3: Unanswerable and hallucination traps
- DS4: Multi-hop reasoning
- DS5: Structured extraction
"""
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from knowledge.models import PageSpan, Citation


# ==============================================================================
# DS1: Factual Derivatives QA
# ==============================================================================

class DS1Example(BaseModel):
    """
    Factual question-answer pair with required citations.
    
    Tests: factual accuracy, citation precision, answer completeness.
    """
    id: str
    question: str
    gold_answer: str
    gold_citations: List[Citation]
    
    # Optional metadata
    topic: Optional[str] = None  # e.g., "black_scholes", "greeks", "hedging"
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None
    requires_formula: bool = False
    requires_calculation: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "ds1_001",
                "question": "What is the Black-Scholes formula for a European call option?",
                "gold_answer": "The Black-Scholes formula for a European call option is C = S₀N(d₁) - Ke⁻ʳᵀN(d₂), where...",
                "gold_citations": [],
                "topic": "black_scholes",
                "difficulty": "medium"
            }
        }


class DS1Dataset(BaseModel):
    """Complete DS1 dataset."""
    version: str = "1.0"
    name: str = "FactualDerivativesQA"
    description: str
    examples: List[DS1Example]
    
    # Statistics
    total_examples: int = 0
    topics_distribution: Dict[str, int] = Field(default_factory=dict)
    
    def __init__(self, **data):
        super().__init__(**data)
        self.total_examples = len(self.examples)
        # Compute topic distribution
        for ex in self.examples:
            if ex.topic:
                self.topics_distribution[ex.topic] = self.topics_distribution.get(ex.topic, 0) + 1


# ==============================================================================
# DS2: Retrieval Quality (Qrels)
# ==============================================================================

class DS2Qrel(BaseModel):
    """
    Query-relevance judgment for a chunk.
    
    relevance: 0 (not relevant), 1 (somewhat relevant), 2 (highly relevant)
    """
    chunk_id: str
    relevance: int = Field(ge=0, le=2)
    doc_id: str
    page_span: Optional[PageSpan] = None


class DS2Example(BaseModel):
    """
    Query with gold-standard relevance judgments.
    
    Tests: retrieval quality (Recall@k, nDCG@k).
    """
    id: str
    query: str
    qrels: List[DS2Qrel]  # Gold-standard relevant chunks
    
    # Optional metadata
    query_type: Optional[str] = None  # 'definition', 'formula', 'comparison', 'procedure'
    expected_num_relevant: int = 0
    
    def __init__(self, **data):
        super().__init__(**data)
        self.expected_num_relevant = len([q for q in self.qrels if q.relevance > 0])


class DS2Dataset(BaseModel):
    """Complete DS2 dataset."""
    version: str = "1.0"
    name: str = "RetrievalQrels"
    description: str
    examples: List[DS2Example]
    
    total_examples: int = 0
    total_qrels: int = 0
    
    def __init__(self, **data):
        super().__init__(**data)
        self.total_examples = len(self.examples)
        self.total_qrels = sum(len(ex.qrels) for ex in self.examples)


# ==============================================================================
# DS3: Unanswerable and Hallucination Traps
# ==============================================================================

class DS3Example(BaseModel):
    """
    Question that should trigger refusal or is a hallucination trap.
    
    Tests: refusal accuracy, hallucination detection, robustness.
    """
    id: str
    question: str
    reason_unanswerable: str  # Why this should be refused/flagged
    trap_type: Literal[
        "out_of_scope",           # Question outside corpus coverage
        "ambiguous",              # Question is too vague/ambiguous
        "conflicting_assumptions", # Question has invalid assumptions
        "similar_term_confusion",  # Uses similar but wrong terminology
        "temporal_mismatch"       # Asks about events/data not in corpus
    ]
    
    # What should the system do?
    expected_behavior: Literal["refuse", "clarify", "flag_uncertainty"]
    
    # Optional: if there's a "trap answer" that looks plausible but is wrong
    trap_answer: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "ds3_001",
                "question": "What was the implied volatility of SPX options on March 15, 2023?",
                "reason_unanswerable": "Corpus contains theoretical models, not real-time market data",
                "trap_type": "out_of_scope",
                "expected_behavior": "refuse"
            }
        }


class DS3Dataset(BaseModel):
    """Complete DS3 dataset."""
    version: str = "1.0"
    name: str = "UnanswerableAndTraps"
    description: str
    examples: List[DS3Example]
    
    total_examples: int = 0
    trap_types_distribution: Dict[str, int] = Field(default_factory=dict)
    
    def __init__(self, **data):
        super().__init__(**data)
        self.total_examples = len(self.examples)
        for ex in self.examples:
            self.trap_types_distribution[ex.trap_type] = self.trap_types_distribution.get(ex.trap_type, 0) + 1


# ==============================================================================
# DS4: Multi-Hop Reasoning
# ==============================================================================

class DS4HopInfo(BaseModel):
    """Information about one reasoning hop."""
    hop_number: int
    sub_question: str
    required_doc_ids: List[str]
    required_concepts: List[str]


class DS4Example(BaseModel):
    """
    Multi-hop question requiring information from multiple sources.
    
    Tests: complex reasoning, information integration, cross-reference handling.
    """
    id: str
    question: str
    gold_answer: str
    gold_citations: List[Citation]
    
    # Multi-hop structure
    hops: List[DS4HopInfo]
    reasoning_type: Literal[
        "sequential",      # Hop 1 → Hop 2 → Answer
        "comparative",     # Compare info from multiple sources
        "synthesizing"     # Combine/synthesize multiple pieces
    ]
    
    # Metadata
    min_required_sources: int = 2
    difficulty: Optional[Literal["medium", "hard", "very_hard"]] = None


class DS4Dataset(BaseModel):
    """Complete DS4 dataset."""
    version: str = "1.0"
    name: str = "MultiHopDerivatives"
    description: str
    examples: List[DS4Example]
    
    total_examples: int = 0
    reasoning_types_distribution: Dict[str, int] = Field(default_factory=dict)
    
    def __init__(self, **data):
        super().__init__(**data)
        self.total_examples = len(self.examples)
        for ex in self.examples:
            self.reasoning_types_distribution[ex.reasoning_type] = \
                self.reasoning_types_distribution.get(ex.reasoning_type, 0) + 1


# ==============================================================================
# DS5: Structured Extraction and Classification
# ==============================================================================

class DS5StructuredOutput(BaseModel):
    """Expected structured output (schema varies by extraction type)."""
    extraction_type: Literal[
        "formula",           # Extract formula with components
        "assumptions",       # Extract assumptions/conditions
        "parameter_ranges",  # Extract valid parameter ranges
        "classification"     # Classify scenario/case
    ]
    output_schema: Dict[str, Any]  # JSON schema for expected output
    gold_output: Dict[str, Any]  # Gold-standard structured answer


class DS5Example(BaseModel):
    """
    Question requiring structured output (JSON, classification, etc.).
    
    Tests: structured extraction, schema compliance, precision.
    """
    id: str
    question: str
    structured_output: DS5StructuredOutput
    gold_citations: List[Citation]
    
    # Evaluation criteria
    required_fields: List[str]  # Fields that must be present
    optional_fields: List[str] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "ds5_001",
                "question": "Extract the Black-Scholes formula components and their meanings.",
                "structured_output": {
                    "extraction_type": "formula",
                    "output_schema": {
                        "formula": "string",
                        "components": "array"
                    },
                    "gold_output": {
                        "formula": "C = S₀N(d₁) - Ke⁻ʳᵀN(d₂)",
                        "components": []
                    }
                },
                "gold_citations": [],
                "required_fields": ["formula", "components"]
            }
        }


class DS5Dataset(BaseModel):
    """Complete DS5 dataset."""
    version: str = "1.0"
    name: str = "StructuredExtraction"
    description: str
    examples: List[DS5Example]
    
    total_examples: int = 0
    extraction_types_distribution: Dict[str, int] = Field(default_factory=dict)
    
    def __init__(self, **data):
        super().__init__(**data)
        self.total_examples = len(self.examples)
        for ex in self.examples:
            et = ex.structured_output.extraction_type
            self.extraction_types_distribution[et] = self.extraction_types_distribution.get(et, 0) + 1


# ==============================================================================
# Unified Dataset Container
# ==============================================================================

class BenchmarkDatasets(BaseModel):
    """
    Container for all benchmark datasets.
    
    Allows loading/saving all datasets together.
    """
    ds1: Optional[DS1Dataset] = None
    ds2: Optional[DS2Dataset] = None
    ds3: Optional[DS3Dataset] = None
    ds4: Optional[DS4Dataset] = None
    ds5: Optional[DS5Dataset] = None
    
    # Metadata
    version: str = "1.0"
    created_at: Optional[str] = None
    corpus_profile: str = "public"
    notes: Optional[str] = None
    
    def get_dataset(self, dataset_name: str):
        """Get dataset by name (ds1, ds2, etc.)."""
        return getattr(self, dataset_name.lower(), None)
    
    def get_total_examples(self) -> int:
        """Get total number of examples across all datasets."""
        total = 0
        for ds_name in ['ds1', 'ds2', 'ds3', 'ds4', 'ds5']:
            ds = self.get_dataset(ds_name)
            if ds:
                total += ds.total_examples
        return total

