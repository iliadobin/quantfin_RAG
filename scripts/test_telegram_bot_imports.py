#!/usr/bin/env python3
"""
Test script to verify all telegram bot imports work correctly.

Usage:
    python scripts/test_telegram_bot_imports.py
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Skip config validation for this test
os.environ["SKIP_CONFIG_VALIDATION"] = "1"


def test_imports():
    """Test all bot module imports."""
    
    print("Testing telegram bot imports...\n")
    
    tests = []
    
    # Test 1: Core modules
    try:
        from apps.telegram_bot import config, state, keyboards, formatter
        print("✅ Core modules (config, state, keyboards, formatter)")
        tests.append(True)
    except ImportError as e:
        print(f"❌ Core modules failed: {e}")
        tests.append(False)
    
    # Test 2: Bot and handlers
    try:
        from apps.telegram_bot import bot, handlers
        print("✅ Bot and handlers")
        tests.append(True)
    except ImportError as e:
        print(f"❌ Bot/handlers failed: {e}")
        tests.append(False)
    
    # Test 3: Pipeline factory
    try:
        from apps.telegram_bot.pipeline_factory import PipelineFactory
        print("✅ Pipeline factory")
        tests.append(True)
    except ImportError as e:
        print(f"❌ Pipeline factory failed: {e}")
        tests.append(False)
    
    # Test 4: RAG dependencies
    try:
        from rag.pipelines import (
            RAGv1Dense, RAGv2Hybrid, RAGv3MultiQuery,
            RAGv4ParentChild, RAGv5Evidence
        )
        print("✅ RAG pipelines")
        tests.append(True)
    except ImportError as e:
        print(f"❌ RAG pipelines failed: {e}")
        tests.append(False)
    
    # Test 5: Retrievers
    try:
        from rag.retrievers import (
            DenseRetriever, BM25Retriever, HybridRetriever
        )
        print("✅ Retrievers")
        tests.append(True)
    except ImportError as e:
        print(f"❌ Retrievers failed: {e}")
        tests.append(False)
    
    # Test 6: Generators and guardrails
    try:
        from rag.generators.citation_generator import CitationGenerator
        from rag.guardrails.unanswerable_detector import UnanswerableDetector
        print("✅ Generators and guardrails")
        tests.append(True)
    except ImportError as e:
        print(f"❌ Generators/guardrails failed: {e}")
        tests.append(False)
    
    # Test 7: Knowledge models
    try:
        from knowledge.models import Answer, Citation, RetrievalTrace
        print("✅ Knowledge models")
        tests.append(True)
    except ImportError as e:
        print(f"❌ Knowledge models failed: {e}")
        tests.append(False)
    
    # Test 8: LLM client
    try:
        from llm.deepseek_client import DeepSeekClient
        print("✅ LLM client")
        tests.append(True)
    except ImportError as e:
        print(f"❌ LLM client failed: {e}")
        tests.append(False)
    
    # Test 9: Telegram library
    try:
        from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
        from telegram.ext import (
            Application, CommandHandler, MessageHandler, CallbackQueryHandler
        )
        print("✅ Telegram library")
        tests.append(True)
    except ImportError as e:
        print(f"❌ Telegram library failed: {e}")
        tests.append(False)
    
    # Summary
    print(f"\n{'='*50}")
    passed = sum(tests)
    total = len(tests)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All imports successful!")
        return 0
    else:
        print(f"❌ {total - passed} test(s) failed")
        return 1


def test_config():
    """Test configuration."""
    print("\nTesting configuration...\n")
    
    from apps.telegram_bot.config import BotConfig
    
    print(f"Pipelines: {list(BotConfig.PIPELINES.keys())}")
    print(f"Models: {list(BotConfig.LLM_MODELS.keys())}")
    print(f"Corpus profiles: {BotConfig.CORPUS_PROFILES}")
    print(f"Defaults: pipeline={BotConfig.DEFAULT_PIPELINE}, model={BotConfig.DEFAULT_MODEL}")
    print("✅ Configuration OK")


def test_state_manager():
    """Test state manager."""
    print("\nTesting state manager...\n")
    
    from apps.telegram_bot.state import StateManager
    
    manager = StateManager()
    
    # Create user state
    state = manager.get_state(12345, "test_user")
    print(f"Created state for user {state.user_id}")
    
    # Update settings
    manager.update_pipeline(12345, "v2")
    manager.update_model(12345, "reasoner")
    manager.toggle_debug(12345)
    
    # Check state
    state = manager.get_state(12345)
    assert state.pipeline == "v2"
    assert state.model == "reasoner"
    assert state.show_debug == True
    
    print(f"Settings: pipeline={state.pipeline}, model={state.model}, debug={state.show_debug}")
    print("✅ State manager OK")


def main():
    """Run all tests."""
    print("="*50)
    print("Telegram Bot Import Test")
    print("="*50 + "\n")
    
    # Test imports
    result = test_imports()
    
    if result == 0:
        # Test configuration
        try:
            test_config()
        except Exception as e:
            print(f"❌ Config test failed: {e}")
            return 1
        
        # Test state manager
        try:
            test_state_manager()
        except Exception as e:
            print(f"❌ State manager test failed: {e}")
            return 1
        
        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED!")
        print("="*50)
        print("\nBot is ready to run. Set up your .env and run:")
        print("  python scripts/run_telegram_bot.py")
        return 0
    else:
        return result


if __name__ == "__main__":
    sys.exit(main())

