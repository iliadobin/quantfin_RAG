"""
Pipeline factory for creating RAG pipelines on demand.

Creates properly configured pipeline instances based on user preferences.
"""
import logging
import os
from pathlib import Path
from typing import Optional

from llm.deepseek_client import DeepSeekClient

from rag.retrievers.dense_retriever import DenseRetriever
from rag.retrievers.bm25_retriever import BM25Retriever
from rag.retrievers.hybrid_retriever import HybridRetriever
from rag.retrievers.multi_query_retriever import MultiQueryRetriever

from rag.rerankers.cross_encoder_reranker import CrossEncoderReranker

from rag.pipelines import (
    RAGv1Dense,
    RAGv2Hybrid,
    RAGv3MultiQuery,
    RAGv4ParentChild,
    RAGv5Evidence
)

from apps.telegram_bot.config import BotConfig


logger = logging.getLogger(__name__)


class PipelineFactory:
    """
    Factory for creating RAG pipeline instances.
    
    Manages shared resources (indices, LLM client) and creates
    pipeline instances based on configuration.
    """
    
    def __init__(self, corpus_profile: str = "public", index_strategy: str = "fixed"):
        """
        Initialize factory.
        
        Args:
            corpus_profile: Default corpus profile to use
            index_strategy: Indexing strategy directory name (e.g. "fixed", "section_aware").
                If the expected layout is not found, the factory will auto-detect a usable one.
        """
        self.corpus_profile = corpus_profile
        self.index_strategy = index_strategy
        
        # Paths
        self.indices_dir = os.path.join(BotConfig.INDICES_DIR, corpus_profile)
        self.parsed_dir = BotConfig.PARSED_DIR
        
        # Initialize shared resources lazily
        self._llm_client: Optional[DeepSeekClient] = None
        self._dense_retriever: Optional[DenseRetriever] = None
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._hybrid_retriever: Optional[HybridRetriever] = None
        self._multi_query_retriever: Optional[MultiQueryRetriever] = None
        self._reranker: Optional[CrossEncoderReranker] = None
        
        logger.info(f"PipelineFactory initialized for corpus: {corpus_profile}")

    def _candidate_index_dirs(self) -> list[Path]:
        """
        Return candidate index directories in priority order.

        Supported layouts:
          - data/indices/<profile>/ (legacy)
          - data/indices/<profile>/<strategy>/ (current build_indices.py output)
        """
        base = Path(self.indices_dir)
        candidates: list[Path] = [base]

        # Preferred strategy first (current layout)
        if self.index_strategy:
            candidates.append(base / self.index_strategy)

        # Common strategies as fallback
        candidates.append(base / "fixed")
        candidates.append(base / "section_aware")

        # Any other subdirectories
        if base.exists() and base.is_dir():
            for p in sorted(base.iterdir()):
                if p.is_dir():
                    candidates.append(p)

        # De-dup while preserving order
        seen = set()
        uniq: list[Path] = []
        for c in candidates:
            if c in seen:
                continue
            seen.add(c)
            uniq.append(c)
        return uniq

    def _resolve_chunks_path(self, index_dir: Path) -> Path:
        """
        Resolve chunk store file name across legacy/current outputs.
        """
        for name in ("chunks.jsonl", "chunks_store.jsonl", "chunks_store.jsonl.gz"):
            p = index_dir / name
            if p.exists():
                return p
        # Default (for error messages)
        return index_dir / "chunks.jsonl"

    def _resolve_index_dir_for(self, *, require: str) -> Path:
        """
        Pick an index directory that contains required artifact.
        """
        checked: list[Path] = []
        for d in self._candidate_index_dirs():
            checked.append(d / require)
            if (d / require).exists():
                return d
        checked_str = "\n".join(f"- {p}" for p in checked[:12])
        if len(checked) > 12:
            checked_str += f"\n- ... ({len(checked) - 12} more)"
        raise FileNotFoundError(
            f"Required index artifact not found: {require}\n"
            f"Searched:\n{checked_str}\n"
            f"Build indices with: python scripts/build_indices.py --profile {self.corpus_profile} --strategy {self.index_strategy}"
        )
    
    def get_llm_client(self) -> DeepSeekClient:
        """Get or create LLM client."""
        if self._llm_client is None:
            self._llm_client = DeepSeekClient()
            logger.info("Created DeepSeek LLM client")
        return self._llm_client
    
    def get_dense_retriever(self) -> DenseRetriever:
        """Get or create dense retriever."""
        if self._dense_retriever is None:
            index_dir = self._resolve_index_dir_for(require="faiss.index")
            chunks_path = self._resolve_chunks_path(index_dir)
            self._dense_retriever = DenseRetriever(
                index_dir=str(index_dir),
                chunks_jsonl_path=str(chunks_path)
            )
            logger.info(f"Created dense retriever from {index_dir}")
        
        return self._dense_retriever
    
    def get_bm25_retriever(self) -> BM25Retriever:
        """Get or create BM25 retriever."""
        if self._bm25_retriever is None:
            index_dir = self._resolve_index_dir_for(require="bm25.pkl")
            chunks_path = self._resolve_chunks_path(index_dir)
            self._bm25_retriever = BM25Retriever(
                index_dir=str(index_dir),
                chunks_jsonl_path=str(chunks_path)
            )
            logger.info(f"Created BM25 retriever from {index_dir}")
        
        return self._bm25_retriever
    
    def get_hybrid_retriever(self) -> HybridRetriever:
        """Get or create hybrid retriever."""
        if self._hybrid_retriever is None:
            dense = self.get_dense_retriever()
            bm25 = self.get_bm25_retriever()
            self._hybrid_retriever = HybridRetriever(dense, bm25)
            logger.info("Created hybrid retriever")
        
        return self._hybrid_retriever

    def get_multi_query_retriever(self) -> MultiQueryRetriever:
        """Get or create multi-query retriever."""
        if self._multi_query_retriever is None:
            base = self.get_hybrid_retriever()
            self._multi_query_retriever = MultiQueryRetriever(
                base_retriever=base,
                expansion_strategy="pricing",
                max_queries=3,
                fusion_method="rrf",
            )
            logger.info("Created multi-query retriever")
        return self._multi_query_retriever

    def get_reranker(self) -> CrossEncoderReranker:
        """Get or create cross-encoder reranker."""
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(device="cpu")
            logger.info("Created cross-encoder reranker")
        return self._reranker
    
    def create_pipeline(
        self,
        pipeline_key: str,
        model_key: str = "chat"
    ):
        """
        Create pipeline instance.
        
        Args:
            pipeline_key: Pipeline identifier (v1-v5)
            model_key: LLM model key (chat, reasoner)
            
        Returns:
            Configured pipeline instance
        """
        logger.info(f"Creating pipeline: {pipeline_key} with model: {model_key}")
        
        # Get model name
        model = BotConfig.LLM_MODELS[model_key]
        
        # Get shared resources
        llm_client = self.get_llm_client()
        
        if pipeline_key == "v1":
            # RAGv1: Dense only
            dense_retriever = self.get_dense_retriever()
            
            return RAGv1Dense(
                dense_retriever=dense_retriever,
                llm_client=llm_client,
                model=model,
                use_unanswerable_detection=True
            )
        
        elif pipeline_key == "v2":
            # RAGv2: Hybrid with rerank
            hybrid_retriever = self.get_hybrid_retriever()
            reranker = self.get_reranker()
            
            return RAGv2Hybrid(
                hybrid_retriever=hybrid_retriever,
                reranker=reranker,
                llm_client=llm_client,
                model=model,
                use_unanswerable_detection=True
            )
        
        elif pipeline_key == "v3":
            # RAGv3: Multi-query
            multi_query_retriever = self.get_multi_query_retriever()
            
            return RAGv3MultiQuery(
                multi_query_retriever=multi_query_retriever,
                llm_client=llm_client,
                model=model
            )
        
        elif pipeline_key == "v4":
            # RAGv4: Parent-child
            dense_retriever = self.get_dense_retriever()
            
            return RAGv4ParentChild(
                child_retriever=dense_retriever,
                llm_client=llm_client,
                model=model
            )
        
        elif pipeline_key == "v5":
            # RAGv5: Evidence validation
            hybrid_retriever = self.get_hybrid_retriever()
            reranker = self.get_reranker()
            
            return RAGv5Evidence(
                hybrid_retriever=hybrid_retriever,
                reranker=reranker,
                llm_client=llm_client,
                model=model
            )
        
        else:
            raise ValueError(f"Unknown pipeline: {pipeline_key}")

