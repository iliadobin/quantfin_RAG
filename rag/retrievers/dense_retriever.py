"""
Dense (vector) retriever using FAISS and sentence transformers.
"""
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from knowledge.models import Chunk, RetrievedChunk
from ingest.indexing.chunk_store import load_chunks_from_parsed


logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Dense retriever using sentence transformers + FAISS.
    
    Uses intfloat/e5-small-v2 for local CPU embedding.
    """
    
    def __init__(
        self,
        index_dir: str,
        chunks_jsonl_path: str,
        model_name: str = "intfloat/e5-small-v2",
        device: str = "cpu"
    ):
        """
        Initialize dense retriever.
        
        Args:
            index_dir: Directory containing faiss.index and chunk_ids.npy
            chunks_jsonl_path: Path to chunk store JSONL
            model_name: Sentence transformer model
            device: Device for embeddings ('cpu' or 'cuda')
        """
        self.index_dir = Path(index_dir)
        self.chunks_jsonl_path = Path(chunks_jsonl_path)
        self.model_name = model_name
        self.device = device
        
        # Load FAISS index
        index_path = self.index_dir / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        self.index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors")
        
        # Load chunk IDs
        chunk_ids_path = self.index_dir / "chunk_ids.npy"
        if not chunk_ids_path.exists():
            raise FileNotFoundError(f"Chunk IDs not found: {chunk_ids_path}")
        self.chunk_ids = np.load(chunk_ids_path, allow_pickle=True).tolist()
        
        # Load chunks
        self.chunks_dict = self._load_chunks()
        logger.info(f"Loaded {len(self.chunks_dict)} chunks")
        
        # Load embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name, device=device)
    
    def _load_chunks(self) -> Dict[str, Chunk]:
        """Load chunks from JSONL into dict."""
        chunks = {}
        if not self.chunks_jsonl_path.exists():
            raise FileNotFoundError(f"Chunk store not found: {self.chunks_jsonl_path}")
        
        import json
        with open(self.chunks_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                chunk = Chunk.model_validate(obj)
                chunks[chunk.id] = chunk
        
        return chunks
    
    def _embed_query(self, query: str) -> np.ndarray:
        """Embed query with e5 prefix."""
        # E5 models require "query: " prefix for queries
        prefixed = f"query: {query}"
        embedding = self.encoder.encode(
            [prefixed],
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]
        return embedding.astype(np.float32)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[RetrievedChunk]:
        """
        Retrieve top-k chunks using dense retrieval.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            **kwargs: Additional parameters (unused)
            
        Returns:
            List of retrieved chunks with scores
        """
        # Embed query
        query_embedding = self._embed_query(query)
        
        # Search
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1),
            k=min(top_k, self.index.ntotal)
        )
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty results
                break
            
            chunk_id = self.chunk_ids[idx]
            chunk = self.chunks_dict.get(chunk_id)
            
            if chunk:
                results.append(RetrievedChunk(
                    chunk=chunk,
                    score=float(score),
                    retriever_tag="dense",
                    metadata={"index": int(idx)}
                ))
        
        return results
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
        **kwargs
    ) -> List[List[RetrievedChunk]]:
        """
        Batch retrieve for multiple queries.
        
        Args:
            queries: List of queries
            top_k: Number of chunks per query
            **kwargs: Additional parameters
            
        Returns:
            List of retrieval results
        """
        # Embed all queries
        prefixed = [f"query: {q}" for q in queries]
        query_embeddings = self.encoder.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype(np.float32)
        
        # Batch search
        scores, indices = self.index.search(
            query_embeddings,
            k=min(top_k, self.index.ntotal)
        )
        
        # Build results for each query
        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for score, idx in zip(query_scores, query_indices):
                if idx == -1:
                    break
                
                chunk_id = self.chunk_ids[idx]
                chunk = self.chunks_dict.get(chunk_id)
                
                if chunk:
                    results.append(RetrievedChunk(
                        chunk=chunk,
                        score=float(score),
                        retriever_tag="dense",
                        metadata={"index": int(idx)}
                    ))
            
            all_results.append(results)
        
        return all_results

