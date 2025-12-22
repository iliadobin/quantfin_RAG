#!/usr/bin/env python3
"""
Quick test to verify project structure and imports.
Run this before collecting corpus to check dependencies.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")
    
    try:
        import yaml
        print("  ✓ yaml")
    except ImportError as e:
        print(f"  ✗ yaml: {e}")
        return False
    
    try:
        import arxiv
        print("  ✓ arxiv")
    except ImportError as e:
        print(f"  ✗ arxiv: {e}")
        return False
    
    try:
        import requests
        print("  ✓ requests")
    except ImportError as e:
        print(f"  ✗ requests: {e}")
        return False
    
    try:
        import fitz
        print("  ✓ fitz (PyMuPDF)")
    except ImportError as e:
        print(f"  ✗ fitz (PyMuPDF): {e}")
        return False
    
    try:
        from pydantic import BaseModel
        print("  ✓ pydantic")
    except ImportError as e:
        print(f"  ✗ pydantic: {e}")
        return False
    
    try:
        from tqdm import tqdm
        print("  ✓ tqdm")
    except ImportError as e:
        print(f"  ✗ tqdm: {e}")
        return False
    
    return True

def test_structure():
    """Test project structure."""
    print("\nTesting project structure...")
    
    required_dirs = [
        "configs",
        "ingest/sources",
        "knowledge",
        "scripts",
        "data/pdf",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ (missing)")
            all_exist = False
    
    return all_exist

def test_configs():
    """Test config files."""
    print("\nTesting config files...")
    
    required_configs = [
        "configs/anchors.yaml",
        "configs/arxiv_search_config.yaml",
    ]
    
    all_exist = True
    for config_path in required_configs:
        full_path = project_root / config_path
        if full_path.exists():
            print(f"  ✓ {config_path}")
        else:
            print(f"  ✗ {config_path} (missing)")
            all_exist = False
    
    return all_exist

def main():
    print("="*80)
    print("QA Assistant - Structure Test")
    print("="*80 + "\n")
    
    imports_ok = test_imports()
    structure_ok = test_structure()
    configs_ok = test_configs()
    
    print("\n" + "="*80)
    if imports_ok and structure_ok and configs_ok:
        print("✓ ALL TESTS PASSED")
        print("\nYou can now run: python scripts/collect_corpus.py")
    else:
        print("✗ SOME TESTS FAILED")
        if not imports_ok:
            print("\nInstall dependencies: pip install -r requirements.txt")
        if not structure_ok:
            print("\nRun: mkdir -p data/pdf data/parsed data/indices")
    print("="*80)

if __name__ == "__main__":
    main()

