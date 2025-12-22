"""
BM25 (lexical) retriever.
"""
import logging
import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from rank_bm25 import BM25Okapi

from knowledge.models import Chunk, RetrievedChunk


logger = logging.getLogger(__name__)


_RE_TOK = re.compile(r"\b\w+\b", flags=re.UNICODE)


def simple_tokenize(text: str) -> List[str]:
    """Simple tokenizer matching index build."""
    return _RE_TOK.findall((text or "").lower())


class BM25Retriever:
    """
    BM25 lexical retriever.
    
    Uses rank_bm25 for efficient term-based retrieval.
    """
    
    def __init__(
        self,
        index_dir: str,
        chunks_jsonl_path: str
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            index_dir: Directory containing bm25.pkl and chunk_ids.npy
            chunks_jsonl_path: Path to chunk store JSONL
        """
        self.index_dir = Path(index_dir)
        self.chunks_jsonl_path = Path(chunks_jsonl_path)
        
        # Load BM25 index
        bm25_path = self.index_dir / "bm25.pkl"
        if not bm25_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {bm25_path}")
        
        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)
        logger.info(f"Loaded BM25 index")
        
        # Load chunk IDs
        chunk_ids_path = self.index_dir / "chunk_ids.npy"
        if not chunk_ids_path.exists():
            raise FileNotFoundError(f"Chunk IDs not found: {chunk_ids_path}")
        self.chunk_ids = np.load(chunk_ids_path, allow_pickle=True).tolist()
        
        # Load chunks
        self.chunks_dict = self._load_chunks()
        logger.info(f"Loaded {len(self.chunks_dict)} chunks for BM25")
    
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
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[RetrievedChunk]:
        """
        Retrieve top-k chunks using BM25.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            **kwargs: Additional parameters (unused)
            
        Returns:
            List of retrieved chunks with BM25 scores
        """
        # Tokenize query
        query_tokens = simple_tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            chunk_id = self.chunk_ids[idx]
            chunk = self.chunks_dict.get(chunk_id)
            score = float(scores[idx])
            
            if chunk and score > 0:  # Only include positive scores
                results.append(RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    retriever_tag="bm25",
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
        return [self.retrieve(q, top_k=top_k, **kwargs) for q in queries]

