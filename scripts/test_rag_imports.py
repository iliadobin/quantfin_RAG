#!/usr/bin/env python3
"""
Quick test to verify all RAG components can be imported.

This is a sanity check to ensure no import errors or missing dependencies.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test all RAG component imports."""
    
    print("Testing RAG component imports...")
    
    # Core models
    print("✓ Testing knowledge models...")
    from knowledge.models import (
        Document, Chunk, PageSpan, Citation, Answer, 
        RetrievalTrace, RetrievedChunk
    )
    
    # Contracts
    print("✓ Testing contracts...")
    from rag.contracts import Retriever, Reranker, Generator, Guardrail, Pipeline
    
    # LLM client
    print("✓ Testing LLM client...")
    from llm.deepseek_client import DeepSeekClient
    
    # Retrievers
    print("✓ Testing retrievers...")
    from rag.retrievers import (
        DenseRetriever,
        BM25Retriever,
        HybridRetriever,
        MultiQueryRetriever
    )
    
    # Rerankers
    print("✓ Testing rerankers...")
    from rag.rerankers import CrossEncoderReranker
    
    # Generators
    print("✓ Testing generators...")
    from rag.generators import CitationGenerator
    from rag.generators import prompts
    
    # Guardrails
    print("✓ Testing guardrails...")
    from rag.guardrails import EvidenceValidator, UnanswerableDetector
    
    # Pipelines
    print("✓ Testing pipelines...")
    from rag.pipelines import (
        RAGv1Dense,
        RAGv2Hybrid,
        RAGv3MultiQuery,
        RAGv4ParentChild,
        RAGv5Evidence
    )
    
    # Main rag package
    print("✓ Testing main rag package...")
    import rag
    
    print("\n✅ All imports successful!")
    print(f"RAG package version: {rag.__version__}")
    
    # Print available pipelines
    print("\nAvailable pipelines:")
    for name in ['RAGv1Dense', 'RAGv2Hybrid', 'RAGv3MultiQuery', 
                 'RAGv4ParentChild', 'RAGv5Evidence']:
        print(f"  - {name}")
    
    return True


def check_dependencies():
    """Check required dependencies."""
    
    print("\nChecking dependencies...")
    
    required = [
        ('numpy', 'numpy'),
        ('faiss', 'faiss-cpu or faiss-gpu'),
        ('sentence_transformers', 'sentence-transformers'),
        ('rank_bm25', 'rank-bm25'),
        ('openai', 'openai'),
        ('pydantic', 'pydantic'),
        ('tenacity', 'tenacity'),
    ]
    
    missing = []
    for module, package in required:
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True


if __name__ == "__main__":
    try:
        deps_ok = check_dependencies()
        imports_ok = test_imports()
        
        if deps_ok and imports_ok:
            print("\n🎉 RAG system ready!")
            sys.exit(0)
        else:
            print("\n❌ Setup incomplete")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

