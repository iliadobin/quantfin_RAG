"""
RAGv4: Parent-Child retrieval.

Retrieves small "child" chunks for precision, then expands to "parent" 
context (section or page) for better answer generation.
"""
import time
import logging
from typing import Optional, List, Dict

from knowledge.models import Answer, RetrievalTrace, RetrievedChunk, Chunk, PageSpan
from llm.deepseek_client import DeepSeekClient
from rag.retrievers.dense_retriever import DenseRetriever
from rag.retrievers.hybrid_retriever import HybridRetriever
from rag.generators.citation_generator import CitationGenerator
from rag.guardrails.unanswerable_detector import UnanswerableDetector


logger = logging.getLogger(__name__)


class RAGv4ParentChild:
    """
    RAG v4: Parent-Child retrieval.
    
    Pipeline:
    1. Retrieve small chunks ("children") for precision
    2. Expand to parent context (page or section) for comprehensiveness
    3. Generate answer from parent context with child-level citations
    """
    
    def __init__(
        self,
        child_retriever,  # DenseRetriever or HybridRetriever
        llm_client: DeepSeekClient,
        parent_strategy: str = "page",  # 'page' or 'section'
        use_unanswerable_detection: bool = True,
        child_top_k: int = 15,
        parent_top_k: int = 5,
        model: Optional[str] = None
    ):
        """
        Initialize RAGv4 pipeline.
        
        Args:
            child_retriever: Retriever for child chunks (dense or hybrid)
            llm_client: DeepSeek client
            parent_strategy: How to expand to parent ('page' or 'section')
            use_unanswerable_detection: Whether to detect unanswerable questions
            child_top_k: Number of child chunks to retrieve
            parent_top_k: Number of parent contexts to use
            model: LLM model to use
        """
        self.child_retriever = child_retriever
        self.llm_client = llm_client
        self.parent_strategy = parent_strategy
        self.child_top_k = child_top_k
        self.parent_top_k = parent_top_k
        self.model = model
        
        # Components
        self.generator = CitationGenerator(llm_client, model=model)
        
        if use_unanswerable_detection:
            self.unanswerable_detector = UnanswerableDetector()
        else:
            self.unanswerable_detector = None
        
        # Store all chunks for parent lookup
        # In real implementation, this would be a proper chunk store/database
        self.chunk_store: Dict[str, Chunk] = {}
        if hasattr(child_retriever, 'chunks_dict'):
            self.chunk_store = child_retriever.chunks_dict
        elif hasattr(child_retriever, 'dense_retriever'):
            # For hybrid retriever
            self.chunk_store = child_retriever.dense_retriever.chunks_dict
    
    def _get_parent_key(self, chunk: Chunk) -> str:
        """
        Get parent key for grouping chunks.
        
        Args:
            chunk: Child chunk
            
        Returns:
            Parent key (e.g., 'doc_id:page_num' or 'doc_id:section_path')
        """
        if self.parent_strategy == "page":
            # Use start page as parent key
            return f"{chunk.doc_id}:page_{chunk.page_span.start_page}"
        else:  # section
            # Use section path as parent key
            section = ":".join(chunk.section_path) if chunk.section_path else "root"
            return f"{chunk.doc_id}:section_{section}"
    
    def _expand_to_parents(
        self,
        child_chunks: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        """
        Expand child chunks to parent contexts.
        
        Groups children by parent, then creates parent chunks by merging text.
        
        Args:
            child_chunks: Retrieved child chunks
            
        Returns:
            Parent chunks (expanded contexts)
        """
        # Group children by parent
        parent_groups: Dict[str, List[RetrievedChunk]] = {}
        
        for child in child_chunks:
            parent_key = self._get_parent_key(child.chunk)
            if parent_key not in parent_groups:
                parent_groups[parent_key] = []
            parent_groups[parent_key].append(child)
        
        # Create parent chunks
        parent_chunks = []
        
        for parent_key, children in parent_groups.items():
            # Sort children by position in document (by page, then char offset)
            children.sort(key=lambda x: (
                x.chunk.page_span.start_page,
                x.chunk.page_span.start_char or 0
            ))
            
            # Aggregate score (max of children)
            parent_score = max(c.score for c in children)
            
            # Merge text
            merged_text = "\n\n".join(c.chunk.text for c in children)
            
            # Get page span (min start to max end)
            min_page = min(c.chunk.page_span.start_page for c in children)
            max_page = max(c.chunk.page_span.end_page for c in children)
            
            # Create parent chunk
            parent_chunk = Chunk(
                id=f"parent_{parent_key}",
                doc_id=children[0].chunk.doc_id,
                strategy=f"parent_{self.parent_strategy}",
                text=merged_text,
                page_span=PageSpan(start_page=min_page, end_page=max_page),
                section_path=children[0].chunk.section_path,
                token_count=sum(c.chunk.token_count for c in children),
                char_count=sum(c.chunk.char_count for c in children),
                metadata={
                    "parent_strategy": self.parent_strategy,
                    "num_children": len(children),
                    "child_ids": [c.chunk.id for c in children]
                }
            )
            
            parent_chunks.append(RetrievedChunk(
                chunk=parent_chunk,
                score=parent_score,
                retriever_tag="parent_child",
                metadata={
                    "num_children": len(children),
                    "child_scores": [c.score for c in children]
                }
            ))
        
        # Sort by score and take top-k
        parent_chunks.sort(key=lambda x: x.score, reverse=True)
        return parent_chunks[:self.parent_top_k]
    
    def run(
        self,
        query: str,
        corpus_profile: str = "public",
        **kwargs
    ) -> Answer:
        """
        Run RAGv4 pipeline.
        
        Args:
            query: User query
            corpus_profile: Corpus to use
            **kwargs: Additional parameters
            
        Returns:
            Answer with citations and trace
        """
        start_time = time.time()
        
        # Override defaults
        child_top_k = kwargs.get('child_top_k', self.child_top_k)
        parent_top_k = kwargs.get('parent_top_k', self.parent_top_k)
        
        # Pre-retrieval check
        if self.unanswerable_detector:
            refusal_reason = self.unanswerable_detector.detect_pre_retrieval(query)
            if refusal_reason:
                logger.info(f"Query refused pre-retrieval: {refusal_reason}")
                return Answer(
                    text="I cannot answer this question as it appears to be outside the scope of the available documents.",
                    citations=[],
                    confidence=0.0,
                    refusal_reason=refusal_reason,
                    trace=RetrievalTrace(
                        query=query,
                        metadata={"pipeline": "rag_v4", "stage": "pre_retrieval_check"}
                    )
                )
        
        # Step 1: Retrieve child chunks
        retrieval_start = time.time()
        child_chunks = self.child_retriever.retrieve(query, top_k=child_top_k)
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        logger.info(f"Retrieved {len(child_chunks)} child chunks in {retrieval_time:.1f}ms")
        
        # Post-retrieval check
        if self.unanswerable_detector:
            refusal_reason = self.unanswerable_detector.detect_post_retrieval(query, child_chunks)
            if refusal_reason:
                logger.info(f"Query refused post-retrieval: {refusal_reason}")
                return Answer(
                    text="I cannot find sufficient relevant information in the documents to answer this question confidently.",
                    citations=[],
                    confidence=0.0,
                    refusal_reason=refusal_reason,
                    trace=RetrievalTrace(
                        query=query,
                        retrieved_chunks_count=len(child_chunks),
                        retrieval_time_ms=retrieval_time,
                        metadata={"pipeline": "rag_v4", "stage": "post_retrieval_check"}
                    )
                )
        
        # Step 2: Expand to parents
        expansion_start = time.time()
        parent_chunks = self._expand_to_parents(child_chunks)
        expansion_time = (time.time() - expansion_start) * 1000
        
        logger.info(f"Expanded to {len(parent_chunks)} parent contexts in {expansion_time:.1f}ms")
        
        # Step 3: Generate answer from parent contexts
        generation_start = time.time()
        answer = self.generator.generate(query, parent_chunks)
        generation_time = (time.time() - generation_start) * 1000
        
        logger.info(f"Generated answer in {generation_time:.1f}ms with {len(answer.citations)} citations")
        
        # Add trace
        total_time = (time.time() - start_time) * 1000
        answer.trace = RetrievalTrace(
            query=query,
            retrieved_chunks_count=len(child_chunks),
            final_chunks_count=len(parent_chunks),
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            metadata={
                "pipeline": "rag_v4",
                "corpus_profile": corpus_profile,
                "parent_strategy": self.parent_strategy,
                "expansion_time_ms": expansion_time,
                "total_time_ms": total_time
            }
        )
        
        return answer
    
    @property
    def name(self) -> str:
        """Pipeline name."""
        return "RAGv4_ParentChild"
    
    @property
    def version(self) -> str:
        """Pipeline version."""
        return "4.0"

