"""
Core data models for QA Assistant knowledge base.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class Document(BaseModel):
    """A source document in the corpus."""
    id: str
    title: str
    source_url: str
    source_type: str  # 'arxiv', 'regulatory', 'industry'
    license: str
    corpus_profile: str = "public"
    
    # File info
    pdf_path: Optional[str] = None
    checksum_sha256: Optional[str] = None
    file_size_bytes: Optional[int] = None
    page_count: Optional[int] = None
    
    # Metadata
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    arxiv_id: Optional[str] = None
    primary_category: Optional[str] = None
    
    # Timestamps
    retrieved_at: Optional[datetime] = None
    ingested_at: Optional[datetime] = None
    
    # Description
    description: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "arxiv_2005_02347",
                "title": "Differential Machine Learning",
                "source_url": "https://arxiv.org/abs/2005.02347",
                "source_type": "arxiv",
                "license": "arXiv.org perpetual non-exclusive license",
                "arxiv_id": "2005.02347",
                "primary_category": "q-fin.CP"
            }
        }


class CorpusManifest(BaseModel):
    """Manifest of all documents in the corpus."""
    version: str = "1.0"
    created_at: datetime
    total_documents: int
    total_pages: int
    documents: List[Document]
    
    # Build info
    build_config: dict = Field(default_factory=dict)
    notes: Optional[str] = None


class PageSpan(BaseModel):
    """
    Span within a document, primarily for citation.

    Pages are 1-indexed. Character offsets are optional and are relative to the
    normalized page text (after normalization pipeline).
    """
    start_page: int
    end_page: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None


class ParsedPage(BaseModel):
    """A single parsed (and normalized) PDF page."""
    doc_id: str
    page_number: int  # 1-indexed
    text: str
    char_count: int

    # Optional diagnostics
    raw_char_count: Optional[int] = None
    extraction_method: Optional[str] = None  # 'pymupdf' | 'pdfplumber'


class Chunk(BaseModel):
    """
    Text chunk to be indexed and retrieved.

    `page_span` is required for citations; `section_path` is optional and may
    be empty for non-structural strategies.
    """
    id: str
    doc_id: str
    strategy: str  # 'fixed' | 'section_aware' | ...
    text: str
    page_span: PageSpan
    section_path: List[str] = Field(default_factory=list)
    token_count: int
    char_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk retrieved by a retriever with score and metadata."""
    chunk: Chunk
    score: float
    retriever_tag: str  # e.g., 'bm25', 'dense', 'hybrid'
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """
    Citation linking answer claims to source document spans.
    
    Used for evidence tracking and user transparency.
    """
    doc_id: str
    page_span: PageSpan
    quote: str  # Actual text excerpt supporting the claim
    score: float
    retriever_tag: str
    chunk_id: Optional[str] = None
    
    def format_page_reference(self) -> str:
        """Format citation as 'doc_id, pp. X-Y'."""
        if self.page_span.start_page == self.page_span.end_page:
            return f"{self.doc_id}, p. {self.page_span.start_page}"
        return f"{self.doc_id}, pp. {self.page_span.start_page}-{self.page_span.end_page}"


class RetrievalTrace(BaseModel):
    """
    Trace of retrieval process for debugging and transparency.
    
    Captures all intermediate steps in the RAG pipeline.
    """
    query: str
    expanded_queries: List[str] = Field(default_factory=list)
    retrieved_chunks_count: int = 0
    reranked_chunks_count: int = 0
    final_chunks_count: int = 0
    retrieval_time_ms: Optional[float] = None
    generation_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Answer(BaseModel):
    """
    Final answer with citations, confidence, and optional refusal.
    
    All claims in the answer should be supported by citations.
    If the system cannot answer with confidence, it should refuse
    and provide a reason.
    """
    text: str
    citations: List[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    refusal_reason: Optional[str] = None
    trace: Optional[RetrievalTrace] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def is_refused(self) -> bool:
        """Check if this is a refusal answer."""
        return self.refusal_reason is not None
    
    def has_citations(self) -> bool:
        """Check if answer has supporting citations."""
        return len(self.citations) > 0
    
    def format_with_citations(self) -> str:
        """Format answer with inline citation references."""
        result = self.text
        if self.citations:
            result += "\n\nReferences:\n"
            for i, cit in enumerate(self.citations, 1):
                result += f"[{i}] {cit.format_page_reference()}: \"{cit.quote[:100]}...\"\n"
        return result

